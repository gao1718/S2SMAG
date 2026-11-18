import torch
from torch import nn
import torch.nn.functional as F
import random


class Encoder(nn.Module):
    def __init__(self, input_size, enc_hidden_dim, num_layers, dec_hid_dim, dropout_prob=0.2):
        super().__init__()
        self.input_size = input_size
        self.enc_hidden_dim = enc_hidden_dim
        self.num_layers = num_layers  # 编码器层数（双向）
        self.dec_hid_dim = dec_hid_dim
        self.dropout_prob = dropout_prob

        # 双向GRU（层数=num_layers）
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=enc_hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=False  # 保持时序优先：[seq_len, batch_size, dim]
        )
        self.dropout = nn.Dropout(dropout_prob)
        # 将双向GRU的输出映射到解码器隐藏维度
        self.fc = nn.Linear(enc_hidden_dim * 2, dec_hid_dim)

    def forward(self, init_input, h0):
        # init_input: [seq_len, batch_size, input_size]
        # h0: [num_layers*2, batch_size, enc_hidden_dim]（双向GRU的初始隐藏状态）
        enc_output, hidden = self.gru(init_input, h0)  # enc_output: [seq_len, batch_size, 2*enc_hid_dim]
                                                      # hidden: [num_layers*2, batch_size, enc_hidden_dim]

        # 拼接双向GRU最后一层的正向和反向隐藏状态
        h_m = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)  # [batch_size, 2*enc_hid_dim]
        h_m = self.dropout(h_m)
        s0 = self.fc(h_m)  # [batch_size, dec_hid_dim]（解码器初始隐藏状态的基础）

        return enc_output, s0


class MultiHeadAttention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim, n_heads=4):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = dec_hid_dim // n_heads  # 每个头的维度
        assert self.d_k * n_heads == dec_hid_dim, "dec_hid_dim必须是n_heads的整数倍"

        # 注意力映射矩阵（Q来自解码器隐藏状态，K/V来自编码器输出）
        self.W_q = nn.Linear(dec_hid_dim, dec_hid_dim)
        self.W_k = nn.Linear(enc_hid_dim * 2, dec_hid_dim)
        self.W_v = nn.Linear(enc_hid_dim * 2, dec_hid_dim)
        self.fc = nn.Linear(dec_hid_dim, dec_hid_dim)

        self.attn_weights = None  # 存储注意力权重：[batch_size, n_heads, 1, src_len]

    def forward(self, dec_hidden, enc_output):
        # dec_hidden: [batch_size, dec_hid_dim]（解码器最后一层的隐藏状态）
        # enc_output: [seq_len, batch_size, 2*enc_hid_dim]
        batch_size = enc_output.size(1)
        src_len = enc_output.size(0)

        # 生成Q、K、V并拆分多头
        q = self.W_q(dec_hidden).view(batch_size, self.n_heads, self.d_k)  # [batch_size, n_heads, d_k]
        k = self.W_k(enc_output.permute(1, 0, 2)).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)  # [batch_size, n_heads, src_len, d_k]
        v = self.W_v(enc_output.permute(1, 0, 2)).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)  # [batch_size, n_heads, src_len, d_k]

        # 计算注意力分数
        attn_scores = torch.matmul(q.unsqueeze(2), k.transpose(-1, -2)) / (self.d_k ** 0.5)  # [batch_size, n_heads, 1, src_len]
        self.attn_weights = F.softmax(attn_scores, dim=-1)  # 注意力权重

        # 加权求和并拼接多头
        context = torch.matmul(self.attn_weights, v).squeeze(2)  # [batch_size, n_heads, d_k]
        context = context.transpose(1, 2).contiguous().view(batch_size, -1)  # [batch_size, dec_hid_dim]

        return self.fc(context)  # [batch_size, dec_hid_dim]


class Decoder(nn.Module):
    def __init__(self, decoder_input_dim, output_dim, enc_hid_dim, dec_hid_dim, dropout, multi_head_attention, num_layers):
        super().__init__()
        self.output_dim = output_dim
        self.multi_head_attention = multi_head_attention
        self.num_layers = num_layers  # 解码器层数（新增参数）
        self.dec_hid_dim = dec_hid_dim

        # 解码器GRU（单向，层数=num_layers）
        self.rnn = nn.GRU(
            input_size=enc_hid_dim,  # 输入维度由fc层保证为enc_hid_dim
            hidden_size=dec_hid_dim,
            num_layers=num_layers,
            batch_first=False  # 时序优先：[seq_len, batch_size, dim]
        )
        # 输出层：融合GRU输出和当前输入特征
        self.fc_out = nn.Linear(dec_hid_dim + output_dim, output_dim)
        # 特征映射层：将拼接特征映射到GRU输入维度
        self.fc = nn.Linear(dec_hid_dim + output_dim + decoder_input_dim, enc_hid_dim)
        self.dropout = nn.Dropout(dropout)

    def get_attention_weights(self):
        return self.multi_head_attention.attn_weights

    def forward(self, dec_input, hidden, enc_output, j):
        # dec_input: [batch_size, decoder_input_dim]（解码器输入）
        # hidden: [num_layers, batch_size, dec_hid_dim]（解码器当前隐藏状态）
        # enc_output: [seq_len, batch_size, 2*enc_hid_dim]（编码器输出）
        # j: [batch_size, output_dim]（上一时间步的输出或目标）

        # 扩展维度以适应GRU的时序优先格式（添加seq_len=1的维度）
        j = j.unsqueeze(0)  # [1, batch_size, output_dim]
        dec_input = dec_input.unsqueeze(0)  # [1, batch_size, decoder_input_dim]

        # 用解码器最后一层的隐藏状态计算注意力
        last_hidden = hidden[-1, :, :]  # [batch_size, dec_hid_dim]（最后一层隐藏状态）
        context = self.multi_head_attention(last_hidden, enc_output).unsqueeze(0)  # [1, batch_size, dec_hid_dim]

        # 拼接特征并映射到GRU输入维度
        rnn_input = torch.cat((j, context, dec_input), dim=2)  # [1, batch_size, output_dim + dec_hid_dim + decoder_input_dim]
        rnn_input = self.fc(rnn_input)  # [1, batch_size, enc_hid_dim]（匹配GRU输入维度）
        rnn_input = self.dropout(rnn_input)

        # GRU前向计算
        dec_output, new_hidden = self.rnn(rnn_input, hidden)  # dec_output: [1, batch_size, dec_hid_dim]
                                                              # new_hidden: [num_layers, batch_size, dec_hid_dim]

        # 调整输出维度并生成预测
        dec_output = dec_output.squeeze(0)  # [batch_size, dec_hid_dim]
        j = j.squeeze(0)  # [batch_size, output_dim]
        combined = torch.cat([dec_output, j], dim=1)  # [batch_size, dec_hid_dim + output_dim]
        pred = self.fc_out(combined)  # [batch_size, output_dim]

        return pred, new_hidden


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.all_attn_weights = None  # [trg_len, batch_size, n_heads, 1, src_len]

    def forward(self, src, trg, Y, is_training=True):
        # src: [src_len, batch_size, input_size]（编码器输入）
        # trg: [trg_len, batch_size, decoder_input_dim]（解码器输入）
        # Y: [trg_len, batch_size, output_dim]（解码器目标）
        batch_size = trg.shape[1]
        trg_len = trg.shape[0]
        output_dim = self.decoder.output_dim
        src_len = src.shape[0]

        # 初始化输出存储张量
        outputs = torch.zeros(trg_len + 1, batch_size, output_dim).to(self.device)

        # 编码器初始隐藏状态（双向GRU需要2*num_layers层）
        enc_num_layers = self.encoder.num_layers
        enc_hidden_dim = self.encoder.enc_hidden_dim
        h0 = torch.zeros(enc_num_layers * 2, batch_size, enc_hidden_dim).to(self.device)

        # 编码器前向计算
        enc_output, s0 = self.encoder(src, h0)  # enc_output: [src_len, batch_size, 2*enc_hid_dim]
                                                # s0: [batch_size, dec_hid_dim]

        # 初始化解码器隐藏状态（扩展为num_layers层）
        dec_num_layers = self.decoder.num_layers
        hidden = s0.unsqueeze(0).repeat(dec_num_layers, 1, 1)  # [num_layers, batch_size, dec_hid_dim]

        # 解码器初始输入和解码条件
        dec_input = trg[0, :, :]  # [batch_size, decoder_input_dim]（第一个时间步输入）
        j = src[-1, :, 0].unsqueeze(1).repeat(1, output_dim)  # [batch_size, output_dim]（初始条件，适配output_dim）

        # 存储注意力权重
        self.all_attn_weights = []

        # 解码循环
        for t in range(1, trg_len + 1):
            # 解码器前向计算
            pred, hidden = self.decoder(dec_input, hidden, enc_output, j)

            # 记录注意力权重
            current_attn = self.decoder.get_attention_weights()  # [batch_size, n_heads, 1, src_len]
            self.all_attn_weights.append(current_attn)

            # 教师强制（训练时）
            if is_training and random.random() < 0.8:
                j = Y[t - 1, :, :]  # 使用目标值
            else:
                j = pred  # 使用预测值

            # 存储输出
            outputs[t] = pred

            # 更新下一时间步的输入
            if t < trg_len:
                dec_input = trg[t, :, :]

        # 拼接所有时间步的注意力权重
        self.all_attn_weights = torch.stack(self.all_attn_weights, dim=0)  # [trg_len, batch_size, n_heads, 1, src_len]
        return outputs  # [trg_len+1, batch_size, output_dim]

    def get_all_attention_weights(self):
        return self.all_attn_weights