# compute_shap_and_save.py
import os
import numpy as np
import pandas as pd
import torch
import shap_Interpretability
import json

import seq2seq_attention  # 模型定义模块
import Data_handing  # 数据处理模块

# ==========================
# 基本设置 & 超参数（与训练一致）
# ==========================

# 强制使用CPU规避cuDNN RNN限制
if torch.cuda.is_available():
    print("检测到GPU，SHAP解释阶段强制使用CPU以避免cuDNN限制。")
device = torch.device('cpu')
print(f"SHAP使用设备: {device}")

# 目标数据集索引
target_index = 0

# 模型结构超参数（必须与训练时一致）
enc_hid_dim = 432
dec_hid_dim = 352
enc_num_layers = 2
dec_num_layers = 3
n_heads = 2
dropout = 0.1

# 模型 checkpoint 路径
model_root = r""
checkpoint_epoch = 15000
checkpoint_path = os.path.join(model_root, f"{checkpoint_epoch}.pth")

# 结果保存路径
shap_save_root = r"G:"
os.makedirs(shap_save_root, exist_ok=True)

# 固定随机种子（复现性）
torch.manual_seed(42)
np.random.seed(42)

# ==========================
# 1. 载入数据（固定划分）
# ==========================
print("加载数据...")

X1 = Data_handing.all_encoder_inputs[target_index]  # [N, src_len, input_size]
X2 = Data_handing.all_decoder_inputs[target_index]  # [N, trg_len, decoder_input_dim]
Y = Data_handing.all_decoder_targets[target_index]  # [N, trg_len, output_dim]

test_X1 = Data_handing.all_test_encoder_inputs_list[target_index]  # [N_test, src_len, input_size]
test_X2 = Data_handing.all_test_decoder_inputs_list[target_index]  # [N_test, trg_len, decoder_input_dim]

if not (isinstance(X1, torch.Tensor) and isinstance(X2, torch.Tensor) and isinstance(Y, torch.Tensor)):
    raise TypeError("输入数据必须为torch.Tensor")

N_total = X1.shape[0]
src_len, input_size = X1.shape[1], X1.shape[2]
trg_len, decoder_input_dim = X2.shape[1], X2.shape[2]
output_dim = Y.shape[2]

print(f"数据维度：")
print(f"  训练集样本数: {N_total}")
print(f"  编码器输入: src_len={src_len}, input_size={input_size}")
print(f"  解码器输入: trg_len={trg_len}, decoder_input_dim={decoder_input_dim}")
print(f"  输出维度: output_dim={output_dim}")

background_src = X1.to(device)
background_trg = X2.to(device)

explain_src = test_X1.to(device)
explain_trg = test_X2.to(device)

print(f"背景数据: {background_src.shape[0]} 条样本")
print(f"解释数据: {explain_src.shape[0]} 条样本")

# ==========================
# 2. 构建并加载模型
# ==========================
print("构建模型结构...")

encoder = seq2seq_attention.Encoder(
    input_size=input_size,
    enc_hidden_dim=enc_hid_dim,
    num_layers=enc_num_layers,
    dec_hid_dim=dec_hid_dim
).to(device)

attention = seq2seq_attention.MultiHeadAttention(
    enc_hid_dim=enc_hid_dim,
    dec_hid_dim=dec_hid_dim,
    n_heads=n_heads
).to(device)

decoder = seq2seq_attention.Decoder(
    decoder_input_dim=decoder_input_dim,
    output_dim=output_dim,
    enc_hid_dim=enc_hid_dim,
    dec_hid_dim=dec_hid_dim,
    dropout=dropout,
    multi_head_attention=attention,
    num_layers=dec_num_layers
).to(device)

model = seq2seq_attention.Seq2Seq(encoder, decoder, device).to(device)

print(f"加载模型权重：{checkpoint_path}")
ckpt = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()





# ==========================
class EncoderDecoderModel(torch.nn.Module):

    def __init__(self, seq2seq_model, output_dim=0, device=None):
        super().__init__()
        self.model = seq2seq_model
        self.output_dim = output_dim  # 输出维度（沉降为单维度）
        self.device = device or next(seq2seq_model.parameters()).device

    def forward(self, src_batch, trg_batch):
        """
        src_batch: [batch_size, src_len, input_size]
        trg_batch: [batch_size, trg_len, decoder_input_dim]
        返回: [batch_size, trg_len] 所有未来时间步的预测
        """
        if src_batch.dim() != 3 or trg_batch.dim() != 3:
            raise ValueError("src_batch和trg_batch必须为3维张量")

        batch_size = src_batch.shape[0]

        # 转换形状为模型所需格式
        src = src_batch.permute(1, 0, 2).to(self.device)  # [src_len, batch, input_size]
        trg = trg_batch.permute(1, 0, 2).to(self.device)  # [trg_len, batch, decoder_input_dim]

        # 创建占位目标Y（推理时不使用，但接口需要）
        dummy_Y = torch.zeros(
            (trg.shape[0], batch_size, self.model.decoder.output_dim),
            device=self.device
        )

        # 推理模式（自回归，不使用teacher forcing）
        outputs = self.model(src, trg, dummy_Y, is_training=False)  # [trg_len+1, batch, output_dim]

        # 提取所有未来时间步（去掉起始符），并取目标输出维度
        out = outputs[1:, :, self.output_dim].permute(1, 0)  # [batch, trg_len]

        return out



# 初始化包装模型
wrapper_model = EncoderDecoderModel(
    seq2seq_model=model,
    device=device
).to(device)
wrapper_model.eval()

# 验证输出形状
print("验证包装模型...")
with torch.no_grad():
    test_out = wrapper_model(explain_src[:2], explain_trg[:2])
    assert test_out.shape == (2, trg_len), f"输出形状错误，预期(2, {trg_len})，实际{test_out.shape}"
print(f"包装模型验证通过，输出形状：{test_out.shape}（样本数×未来时间步数）")

# ==========================
# 4. 准备SHAP输入数据（列表形式）
# ==========================
print("准备SHAP输入数据...")

# 背景数据和解释数据用列表包装（不是元组）
background_data = [background_src, background_trg]
explain_data = [explain_src, explain_trg]

# ==========================
# 5. 计算多时间步SHAP值
# ==========================
print("构建DeepExplainer...")
explainer = shap_Interpretability.DeepExplainer(wrapper_model, background_data)

print("计算SHAP值（多时间步输出，可能耗时较长）...")
shap_values_list = explainer.shap_values(explain_data, check_additivity=False)

# 处理SHAP返回值
# shap_values_list 是一个列表，每个元素对应一个输出时间步
# 对于多输入情况，每个时间步的SHAP值又是一个列表，包含每个输入的SHAP值
if isinstance(shap_values_list, list) and len(shap_values_list) > 0:
    # 如果是多输出（多个时间步），取第一个时间步的结构来判断
    if isinstance(shap_values_list[0], list):
        # 多输入多输出情况
        # shap_values_list[时间步][输入索引][样本, 时间步, 特征]
        # 需要重组为 [输入索引][样本, 时间步, 特征, 输出时间步]
        num_outputs = len(shap_values_list)
        num_inputs = len(shap_values_list[0])

        # 重组SHAP值
        shap_values_src = []
        shap_values_trg = []

        for sample_idx in range(explain_src.shape[0]):
            sample_shap_src = []
            sample_shap_trg = []
            for out_step in range(num_outputs):
                sample_shap_src.append(shap_values_list[out_step][0][sample_idx])
                sample_shap_trg.append(shap_values_list[out_step][1][sample_idx])
            shap_values_src.append(np.stack(sample_shap_src, axis=-1))
            shap_values_trg.append(np.stack(sample_shap_trg, axis=-1))

        shap_values_src = np.array(shap_values_src)
        shap_values_trg = np.array(shap_values_trg)
    else:
        # 单输出情况
        shap_values_src = shap_values_list[0]
        shap_values_trg = shap_values_list[1]
else:
    shap_values_src = shap_values_list[0]
    shap_values_trg = shap_values_list[1]

print(f"编码器SHAP值形状：{np.array(shap_values_src).shape}")
print(f"解码器SHAP值形状：{np.array(shap_values_trg).shape}")

# ==========================
# 6. 计算并保存基准值和预测值
# ==========================
print("计算基准值和解释样本预测值...")
with torch.no_grad():
    # 1. 计算基准值：背景数据在每个未来时间步的平均预测值
    bg_preds = wrapper_model(background_src, background_trg)
    base_values = bg_preds.mean(dim=0).cpu().numpy()  # [trg_len]

    # 2. 计算解释样本预测值
    explain_preds = wrapper_model(explain_src, explain_trg).cpu().numpy()  # [10, trg_len]

# 保存到文件

np.save(os.path.join(shap_save_root, "base_values.npy"), base_values)
np.save(os.path.join(shap_save_root, "explain_preds.npy"), explain_preds)
print(f"基准值保存路径：{os.path.join(shap_save_root, 'base_values.npy')}")
print(f"解释样本预测值保存路径：{os.path.join(shap_save_root, 'explain_preds.npy')}")

# ==========================
# 7. 分析编码器输入的SHAP值
# ==========================
print("\n分析编码器输入的多时间步特征重要性...")

shap_values_src_np = np.array(shap_values_src)
print(f"编码器SHAP值形状：{shap_values_src_np.shape}")

# 确保形状正确：[样本数, src_len, input_size, trg_len]
if shap_values_src_np.ndim == 4:
    # 每个"时间步×特征"的综合重要性（对样本和未来时间步求平均）
    comprehensive_importance_src = np.mean(
        np.abs(shap_values_src_np),
        axis=(0, 3)  # 对样本维度和未来时间步维度取平均
    )  # [src_len, input_size]

    # 特征重要性（对所有时间步、未来时间步和样本平均）
    feature_importance_src = np.mean(np.abs(shap_values_src_np), axis=(0, 1, 3))  # [input_size]

    # 时间步重要性（对所有特征、未来时间步和样本平均）
    time_importance_src = np.mean(np.abs(shap_values_src_np), axis=(0, 2, 3))  # [src_len]
else:
    # 如果形状不是4维，尝试其他处理方式
    print(f"警告：编码器SHAP值形状不是预期的4维，实际为{shap_values_src_np.ndim}维")
    comprehensive_importance_src = np.mean(np.abs(shap_values_src_np), axis=0)
    feature_importance_src = np.mean(np.abs(shap_values_src_np),
                                     axis=(0, 1)) if shap_values_src_np.ndim >= 3 else np.mean(
        np.abs(shap_values_src_np), axis=0)
    time_importance_src = np.mean(np.abs(shap_values_src_np), axis=(0, 2)) if shap_values_src_np.ndim >= 3 else np.mean(
        np.abs(shap_values_src_np), axis=0)

# 生成名称
feature_labels = ["历史累计沉降", "沉降速率", "真空度"]
feature_names = feature_labels[:input_size]  # 确保长度匹配
time_steps = [f"历史时间步{t + 1}" for t in range(src_len)]

# 综合重要性表（编码器）
importance_data_src = []
if comprehensive_importance_src.ndim == 2:
    for t in range(min(src_len, comprehensive_importance_src.shape[0])):
        for f in range(min(input_size, comprehensive_importance_src.shape[1])):
            importance_data_src.append({
                "时间步索引": t,
                "时间步名称": time_steps[t],
                "特征索引": f,
                "特征名称": feature_names[f],
                "综合重要性": comprehensive_importance_src[t, f]
            })

df_comprehensive_src = pd.DataFrame(importance_data_src).sort_values(
    by="综合重要性", ascending=False
) if importance_data_src else pd.DataFrame()

# 特征重要性表（编码器）
df_feature_src = pd.DataFrame({
    "特征索引": list(range(len(feature_importance_src))),
    "特征名称": feature_names[:len(feature_importance_src)],
    "重要性": feature_importance_src
}).sort_values(by="重要性", ascending=False)

# 时间步重要性表（编码器）
df_time_src = pd.DataFrame({
    "时间步索引": list(range(len(time_importance_src))),
    "时间步名称": time_steps[:len(time_importance_src)],
    "重要性": time_importance_src
}).sort_values(by="时间步索引")

# ==========================
# 8. 分析解码器输入的SHAP值
# ==========================
print("分析解码器输入的多时间步特征重要性...")

shap_values_trg_np = np.array(shap_values_trg)
print(f"解码器SHAP值形状：{shap_values_trg_np.shape}")

# 确保形状正确：[样本数, trg_len, decoder_input_dim, trg_len]
if shap_values_trg_np.ndim == 4:
    # 解码器特征重要性
    feature_importance_trg = np.mean(np.abs(shap_values_trg_np), axis=(0, 1, 3))  # [decoder_input_dim]

    # 解码器时间步重要性
    time_importance_trg = np.mean(np.abs(shap_values_trg_np), axis=(0, 2, 3))  # [trg_len]
else:
    print(f"警告：解码器SHAP值形状不是预期的4维，实际为{shap_values_trg_np.ndim}维")
    feature_importance_trg = np.mean(np.abs(shap_values_trg_np),
                                     axis=(0, 1)) if shap_values_trg_np.ndim >= 3 else np.mean(
        np.abs(shap_values_trg_np), axis=0)
    time_importance_trg = np.mean(np.abs(shap_values_trg_np), axis=(0, 2)) if shap_values_trg_np.ndim >= 3 else np.mean(
        np.abs(shap_values_trg_np), axis=0)

# 解码器特征名称（根据实际情况修改）
decoder_feature_names = [f"解码器特征{i + 1}" for i in range(decoder_input_dim)]
decoder_time_steps = [f"未来时间步{t + 1}" for t in range(trg_len)]

# 解码器特征重要性表
df_feature_trg = pd.DataFrame({
    "特征索引": list(range(len(feature_importance_trg))),
    "特征名称": decoder_feature_names[:len(feature_importance_trg)],
    "重要性": feature_importance_trg
}).sort_values(by="重要性", ascending=False)

# 解码器时间步重要性表
df_time_trg = pd.DataFrame({
    "时间步索引": list(range(len(time_importance_trg))),
    "时间步名称": decoder_time_steps[:len(time_importance_trg)],
    "重要性": time_importance_trg
}).sort_values(by="时间步索引")

# ==========================
# 9. 保存所有结果
# ==========================
print("保存用于画图的数据...")

# 9.1 保存原始SHAP值
shap_src_path = os.path.join(shap_save_root, "shap_values_encoder_multi_step.npy")
shap_trg_path = os.path.join(shap_save_root, "shap_values_decoder_multi_step.npy")
np.save(shap_src_path, shap_values_src_np)
np.save(shap_trg_path, shap_values_trg_np)

# 9.2 保存解释样本的输入
explain_src_path = os.path.join(shap_save_root, "explain_src.npy")
explain_trg_path = os.path.join(shap_save_root, "explain_trg.npy")
np.save(explain_src_path, explain_src.cpu().numpy())
np.save(explain_trg_path, explain_trg.cpu().numpy())

# 9.3 保存编码器重要性分析结果
comprehensive_src_path = os.path.join(shap_save_root, "encoder_comprehensive_importance.csv")
feature_src_path = os.path.join(shap_save_root, "encoder_feature_importance.csv")
time_src_path = os.path.join(shap_save_root, "encoder_time_step_importance.csv")
df_comprehensive_src.to_csv(comprehensive_src_path, index=False, encoding="utf-8-sig")
df_feature_src.to_csv(feature_src_path, index=False, encoding="utf-8-sig")
df_time_src.to_csv(time_src_path, index=False, encoding="utf-8-sig")

# 9.4 保存解码器重要性分析结果
feature_trg_path = os.path.join(shap_save_root, "decoder_feature_importance.csv")
time_trg_path = os.path.join(shap_save_root, "decoder_time_step_importance.csv")
df_feature_trg.to_csv(feature_trg_path, index=False, encoding="utf-8-sig")
df_time_trg.to_csv(time_trg_path, index=False, encoding="utf-8-sig")

# 9.5 保存元信息
meta = {
    "src_len": int(src_len),
    "input_size": int(input_size),
    "trg_len": int(trg_len),
    "decoder_input_dim": int(decoder_input_dim),
    "output_dim": int(output_dim),
    "encoder_feature_names": feature_names,
    "encoder_time_step_names": time_steps,
    "decoder_feature_names": decoder_feature_names,
    "decoder_time_step_names": decoder_time_steps
}
meta_path = os.path.join(shap_save_root, "meta.json")
with open(meta_path, "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)


