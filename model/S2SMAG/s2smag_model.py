"""S2SMAG：双向 GRU 编码器、多头注意力及自回归 GRU 解码器。

来源为原 seq2seq_attention.py。外部统一采用 [批量, 时间, 特征]；
网络内部保留原实现的时间优先顺序与参数名称，便于逐层核对。
未来真值只允许在训练模式下按教师强制概率使用，测试时逐步反馈预测值。
"""

import random
from typing import Optional

import torch
from torch import nn


class Encoder(nn.Module):
    """读取历史沉降量、沉降速率及真空度，生成编码序列和解码初始状态。"""

    def __init__(self, input_size, enc_hidden_dim, num_layers, dec_hid_dim, dropout_prob=0.2):
        super().__init__()
        self.enc_hidden_dim = enc_hidden_dim
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, enc_hidden_dim, num_layers, bidirectional=True)
        # 原编码器只在最终隐状态拼接后 Dropout，不在 GRU 层间使用 Dropout。
        self.dropout = nn.Dropout(dropout_prob)
        self.fc = nn.Linear(enc_hidden_dim * 2, dec_hid_dim)

    def forward(self, inputs):
        # 不显式传入 h0 时，PyTorch 在输入设备和精度上自动创建全零初始状态。
        outputs, hidden = self.gru(inputs)
        last_hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        return outputs, self.fc(self.dropout(last_hidden))


class MultiHeadAttention(nn.Module):
    """由解码状态查询完整历史编码序列的缩放点积注意力。"""

    def __init__(self, enc_hid_dim, dec_hid_dim, n_heads=4):
        super().__init__()
        if n_heads < 1 or dec_hid_dim % n_heads:
            raise ValueError("decoder_hidden_size 必须能被 num_heads 整除，且 num_heads > 0。")
        self.n_heads = n_heads
        self.d_k = dec_hid_dim // n_heads
        self.W_q = nn.Linear(dec_hid_dim, dec_hid_dim)
        self.W_k = nn.Linear(enc_hid_dim * 2, dec_hid_dim)
        self.W_v = nn.Linear(enc_hid_dim * 2, dec_hid_dim)
        self.fc = nn.Linear(dec_hid_dim, dec_hid_dim)

    def forward(self, state, encoder_outputs):
        batch_size = encoder_outputs.shape[1]
        query = self.W_q(state).view(batch_size, self.n_heads, self.d_k)
        memory = encoder_outputs.transpose(0, 1)
        keys = self.W_k(memory).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        values = self.W_v(memory).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        scores = torch.matmul(query.unsqueeze(2), keys.transpose(-1, -2)) / self.d_k ** 0.5
        weights = torch.softmax(scores, dim=-1)
        context = torch.matmul(weights, values).squeeze(2)
        # 注意：保留原代码的 transpose 后再展平顺序；改成直接展平会改变既有网络语义。
        context = context.transpose(1, 2).contiguous().view(batch_size, -1)
        return self.fc(context)


class Decoder(nn.Module):
    """融合上一时刻沉降量、注意力上下文以及本时刻未来真空度。"""

    def __init__(self, future_input_size, output_size, enc_hid_dim, dec_hid_dim, dropout, attention,
                 num_layers=1, dropout_enabled=False):
        super().__init__()
        self.output_dim = output_size
        self.MultiHeadAttention = attention
        self.num_layers = num_layers
        self.dropout_enabled = dropout_enabled
        self.rnn = nn.GRU(enc_hid_dim, dec_hid_dim, num_layers=num_layers)
        self.fc_out = nn.Linear(dec_hid_dim + output_size, output_size)
        self.fc = nn.Linear(dec_hid_dim + output_size + future_input_size, enc_hid_dim)
        # 本地旧版没有调用该层；作者 GitHub 多层版在 fc 后调用，由配置明确选择。
        self.dropout = nn.Dropout(dropout)

    def forward(self, future_input, state, encoder_outputs, previous_target):
        # 多层解码器以最后一层状态查询注意力，完整 state 送入 GRU 继续递推。
        context = self.MultiHeadAttention(state[-1], encoder_outputs)
        fused = torch.cat((previous_target, context, future_input), dim=-1)
        rnn_input = self.fc(fused.unsqueeze(0))
        if self.dropout_enabled:
            rnn_input = self.dropout(rnn_input)
        output, hidden = self.rnn(rnn_input, state)
        prediction = self.fc_out(torch.cat((output.squeeze(0), previous_target), dim=-1))
        return prediction, hidden


class S2SMAG(nn.Module):
    """统一接口输出 [批量, 预测步数, 1]，不再保留原来的首个空白输出步。"""

    def __init__(self, input_size=3, future_input_size=1, output_size=1, hidden_size=256,
                 decoder_hidden_size=256, num_layers=2, dropout=0.2, num_heads=4, horizon=10,
                 decoder_num_layers=1, encoder_dropout=None, decoder_dropout_enabled=False,
                 teacher_forcing_rng="torch"):
        super().__init__()
        if output_size != 1:
            raise ValueError("本项目预测第一列沉降量，output_size 必须为 1。")
        if min(input_size, future_input_size, hidden_size, decoder_hidden_size, num_layers, decoder_num_layers, horizon) < 1:
            raise ValueError("网络维度、层数和 horizon 均必须为正整数。")
        if teacher_forcing_rng not in {"torch", "python"}:
            raise ValueError("teacher_forcing_rng 必须为 torch 或 python。")
        self.input_size = input_size
        self.future_input_size = future_input_size
        self.horizon = horizon
        self.teacher_forcing_rng = teacher_forcing_rng
        encoder_dropout = dropout if encoder_dropout is None else encoder_dropout
        self.encoder = Encoder(input_size, hidden_size, num_layers, decoder_hidden_size, encoder_dropout)
        attention = MultiHeadAttention(hidden_size, decoder_hidden_size, num_heads)
        self.decoder = Decoder(future_input_size, output_size, hidden_size, decoder_hidden_size, dropout, attention,
                               num_layers=decoder_num_layers, dropout_enabled=decoder_dropout_enabled)

    def forward(self, encoder_inputs, future_inputs, targets: Optional[torch.Tensor] = None,
                teacher_forcing_ratio=0.0):
        """训练可提供 targets；eval 模式始终自回归，避免使用未来沉降真值。"""
        if encoder_inputs.ndim != 3 or encoder_inputs.shape[-1] != self.input_size:
            raise ValueError(f"encoder_inputs 应为 [batch, history, {self.input_size}]。")
        batch_size = encoder_inputs.shape[0]
        if future_inputs.shape != (batch_size, self.horizon, self.future_input_size):
            raise ValueError(f"future_inputs 应为 [batch, {self.horizon}, {self.future_input_size}]。")
        if encoder_inputs.shape[1] < 1:
            raise ValueError("历史序列不能为空。")
        if not 0.0 <= teacher_forcing_ratio <= 1.0:
            raise ValueError("teacher_forcing_ratio 必须在 [0, 1] 内。")
        if targets is not None and targets.shape != (batch_size, self.horizon, 1):
            raise ValueError(f"targets 应为 [batch, {self.horizon}, 1]。")
        if self.training and teacher_forcing_ratio > 0 and targets is None:
            raise ValueError("启用教师强制训练时必须提供 targets。")

        memory, state = self.encoder(encoder_inputs.transpose(0, 1))
        # 作者多层实现把编码器投影后的同一状态复制给解码器的每一层。
        state = state.unsqueeze(0).repeat(self.decoder.num_layers, 1, 1)
        previous_target = encoder_inputs[:, -1, :1]
        predictions = []
        for step in range(self.horizon):
            prediction, state = self.decoder(future_inputs[:, step], state, memory, previous_target)
            predictions.append(prediction)
            use_teacher = False
            if self.training and targets is not None and teacher_forcing_ratio > 0:
                # 作者模式用 Python RNG，避免每一步 CUDA rand.item() 强制同步；
                # 默认 torch 模式保留本地旧版训练的随机数来源。
                draw = (random.random() if self.teacher_forcing_rng == "python"
                        else torch.rand((), device=encoder_inputs.device).item())
                use_teacher = draw < teacher_forcing_ratio
            previous_target = targets[:, step] if use_teacher else prediction
        return torch.stack(predictions, dim=1)

def build_model(cfg, data=None) -> S2SMAG:
    """本地配置驱动架构创建，不在架构模块读取表格或启动训练。

    data 保留与共享运行时的工厂接口兼容；输入维度以 cfg 为准，
    运行时会核对实际数据维度，避免从数据静默覆盖用户设置。
    """
    return S2SMAG(
        input_size=cfg.INPUT_SIZE,
        future_input_size=cfg.FUTURE_INPUT_SIZE,
        output_size=cfg.OUTPUT_SIZE,
        hidden_size=cfg.HIDDEN_DIM,
        num_layers=cfg.NUM_LAYERS,
        dropout=cfg.DROPOUT,
        horizon=cfg.HORIZON,
        decoder_hidden_size=cfg.DECODER_HIDDEN_DIM,
        num_heads=cfg.ATTENTION_HEADS,
        decoder_num_layers=getattr(cfg, "DECODER_NUM_LAYERS", 1),
        encoder_dropout=getattr(cfg, "ENCODER_DROPOUT", cfg.DROPOUT),
        decoder_dropout_enabled=getattr(cfg, "DECODER_DROPOUT_ENABLED", False),
        teacher_forcing_rng=getattr(cfg, "TEACHER_FORCING_RNG", "torch"),
    )
