import torch
import torch.nn as nn
import seq2seq_attention
import matplotlib.pyplot as plt
import pandas as pd
import os
import time
from torch.utils.data import DataLoader, TensorDataset
import Data_handing
from Lookahead_optimizer import Lookahead



# 基础设置
# ----------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 图表字体设置
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'axes.unicode_minus': False,
    'font.size': 10,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'legend.fontsize': 10
})

# 保存路径
save_root = r'G:'
os.makedirs(save_root, exist_ok=True)

# ----------------------------
# 数据准备（数据留在CPU，训练时再移到GPU）
# ----------------------------
target = 0
X1 = Data_handing.all_encoder_inputs[target]
X2 = Data_handing.all_decoder_inputs[target]
Y = Data_handing.all_decoder_targets[target]

input_size = X1.shape[2]
output_dim = Y.shape[2]
decoder_input_dim = X2.shape[2]



# 最优超参数
# ----------------------------
enc_hid_dim = 432
dec_hid_dim = 352
enc_num_layers = 2
dec_num_layers = 3
n_heads = 2
batch_size_train = 16
lookahead_k = 2
lookahead_alpha = 0.7
lr = 9.628021160532742e-05
dropout = 0.1
num_epochs = 15000

# 强制保存的指定epoch
save_epochs = [1000, 3000, 5000, 10000]


# ----------------------------
# 模型定义
# ----------------------------
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


# 优化器
# ----------------------------
adamw_optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=lr,
    betas=(0.9, 0.99),
    eps=1e-08,
    weight_decay=0.001
)
optimizer = Lookahead(adamw_optimizer, k=lookahead_k, alpha=lookahead_alpha)
criterion = nn.MSELoss()

# 数据加载器
# ----------------------------
train_dataset = TensorDataset(X1, X2, Y)
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size_train,
    shuffle=True,
    pin_memory=True
)

# ----------------------------
train_losses = []
best_loss = float('inf')
best_epoch = 0
start_time = time.time()

for epoch in range(num_epochs):
    model.train()
    epoch_train_loss = 0.0

    for batch_idx, (src, trg, Y_batch) in enumerate(train_loader):
        # 批次数据移到GPU
        src = src.transpose(0, 1).to(device)
        trg = trg.transpose(0, 1).to(device)
        Y_batch = Y_batch.transpose(0, 1).to(device)

        optimizer.zero_grad()
        outputs = model(src, trg, Y_batch)
        outputs = outputs[1:]

        loss = criterion(outputs, Y_batch)
        # 打印每个batch的损失
        print(f"Epoch {epoch+1}, Batch {batch_idx+1}, Loss: {loss.item():.7f}")
        epoch_train_loss += loss.item()
        loss.backward()
        optimizer.step()

    # 计算并打印当前epoch的平均损失
    avg_train_loss = epoch_train_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    print(f'Epoch {epoch+1}/{num_epochs}, Average Train Loss: {avg_train_loss:.7f}')

    current_epoch = epoch + 1

    # 1. 强制保存指定epoch的模型（无论损失是否最优）
    if current_epoch in save_epochs:
        timed_model_path = os.path.join(save_root, f'timed_model_epoch{current_epoch}_loss{avg_train_loss:.6f}.pth')
        torch.save({
            'epoch': current_epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_train_loss
        }, timed_model_path)
        print(f"已保存指定轮次模型（第{current_epoch}轮）至：{timed_model_path}")

# ----------------------------
# 训练总结
# ----------------------------
training_time = time.time() - start_time
print(f"\n训练完成！总耗时: {training_time/3600:.2f} 小时")
print(f"历史最优模型位于第 {best_epoch} 轮，损失: {best_loss:.7f}")
print(f"指定轮次模型已保存：{save_epochs} 轮")


# ----------------------------
# 损失曲线与数据保存
# ----------------------------
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_epochs+1), train_losses, color='blue', linewidth=1.5, label='Training Loss')
# 标记最优模型和指定保存轮次
plt.scatter(best_epoch, best_loss, color='red', s=80, marker='*', label=f'Best Epoch ({best_epoch})')
for epoch in save_epochs:
    if epoch <= num_epochs:
        loss_at_epoch = train_losses[epoch-1]
        plt.scatter(epoch, loss_at_epoch, color='green', s=60, marker='s', label=f'Timed Save ({epoch})')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Training Loss Curve')
plt.legend(loc='upper right')
plt.grid(True, linestyle='--', alpha=0.3)
plt.xlim(0, num_epochs)
plt.ylim(0, max(train_losses) * 1.1)

# 保存损失曲线
loss_fig_png = os.path.join(save_root, 'training_loss_curve.png')
loss_fig_pdf = os.path.join(save_root, 'training_loss_curve.pdf')
plt.tight_layout()
plt.savefig(loss_fig_png, dpi=300, bbox_inches='tight')
plt.savefig(loss_fig_pdf, dpi=300, bbox_inches='tight')
plt.show()

# 保存损失数据
loss_data = pd.DataFrame({
    'Epoch': range(1, num_epochs+1),
    'Training_Loss': train_losses
})
loss_excel_path = os.path.join(save_root, 'training_losses.xlsx')
loss_data.to_excel(loss_excel_path, index=False)
print(f"损失数据已保存至：{loss_excel_path}")