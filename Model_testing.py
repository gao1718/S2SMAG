import torch
import os
import Data_handing
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seq2seq_attention

# 取消中文字体设置，默认使用英文字体
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题
plt.rcParams['font.size'] = 10  # 基础字体大小

# -------------------------- 1. 数据准备与模型初始化 --------------------------
target = 0
X1 = Data_handing.all_test_encoder_inputs_list[target]
X2 = Data_handing.all_test_decoder_inputs_list[target]
Y = Data_handing.all_test_decoder_targets_list[target]

input_size = X1.shape[2]
output_dim = Y.shape[2]
decoder_input_dim = X2.shape[2]

# 训练时的最优超参数
enc_hid_dim = 432
dec_hid_dim = 352
enc_num_layers = 2
dec_num_layers = 3
n_heads = 2
dropout = 0.1

# 确定设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 初始化模型结构并移到设备
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

test_model = seq2seq_attention.Seq2Seq(encoder, decoder, device=device)

# -------------------------- 2. 加载模型参数 --------------------------
folder_path = r'G:'
file_name = '.pth'
full_path = os.path.join(folder_path, file_name)

checkpoint = torch.load(full_path, weights_only=False)
test_model.load_state_dict(checkpoint['model_state_dict'])
test_model.eval()  # 设为评估模式

# -------------------------- 3. 数据加载与设备适配 --------------------------
test_X1 = X1  # [108, 10, 3]
test_X2 = X2  # [108, 10, 1]
test_Y = Y  # [108, 10, 1]

# 调整维度并移到设备
test_X1 = test_X1.transpose(0, 1).to(device)  # [10, 108, 3]
test_X2 = test_X2.transpose(0, 1).to(device)  # [10, 108, 1]
test_Y = test_Y.transpose(0, 1).to(device)  # [10, 108, 1]

# -------------------------- 4. 模型预测 --------------------------
with torch.no_grad():
    outputs = test_model(test_X1, test_X2, test_Y, is_training=False)
    outputs = outputs[1:]  # 忽略第一个无效时间步

# -------------------------- 5. 反归一化处理 --------------------------
scaler_1 = Data_handing.all_scalers_1[0]

# 转换为[sample, seq_len, dim]格式
outputs_np = outputs.cpu().numpy().transpose(1, 0, 2)  # [108, 10, 1]
test_Y_np = test_Y.cpu().numpy().transpose(1, 0, 2)  # [108, 10, 1]

# 反归一化
original_shape_outputs = outputs_np.shape
original_shape_test_Y = test_Y_np.shape

outputs_np_reshaped = outputs_np.reshape(-1, outputs_np.shape[-1])
test_Y_np_reshaped = test_Y_np.reshape(-1, test_Y_np.shape[-1])

if scaler_1.scale_[0] == 0:
    print("Warning: Constant feature value, skipping inverse normalization")
    outputs_np_inverse = outputs_np_reshaped
    test_Y_np_inverse = test_Y_np_reshaped
else:
    outputs_np_inverse = scaler_1.inverse_transform(outputs_np_reshaped)
    test_Y_np_inverse = scaler_1.inverse_transform(test_Y_np_reshaped)

# 恢复形状
outputs = outputs_np_inverse.reshape(original_shape_outputs)  # [108, 10, 1]
test_Y = test_Y_np_inverse.reshape(original_shape_test_Y)  # [108, 10, 1]

# -------------------------- 6. 评估指标计算 --------------------------
outputs_tensor = torch.tensor(outputs, dtype=torch.float32).to(device)
test_Y_tensor = torch.tensor(test_Y, dtype=torch.float32).to(device)

# 整体指标
criterion = nn.MSELoss()
mse = criterion(outputs_tensor, test_Y_tensor).item()
rmse = np.sqrt(mse) if not np.isnan(mse) else np.nan

y_mean = torch.mean(test_Y_tensor)
ss_total = torch.sum((test_Y_tensor - y_mean) ** 2).item()
ss_residual = torch.sum((test_Y_tensor - outputs_tensor) ** 2).item()

r2 = 1 - (ss_residual / ss_total) if ss_total != 0 else np.nan

print(f"Overall - RMSE: {rmse:.6f}, R²: {r2:.6f}")
