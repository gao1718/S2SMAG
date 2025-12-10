import torch
import torch.nn as nn
import seq2seq_attention
import pandas as pd
import numpy as np
import os
import time
import optuna
import random
from optuna.trial import TrialState
from torch.utils.data import DataLoader, TensorDataset
import Data_handing
from Lookahead_optimizer import Lookahead

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 数据准备 - 和原始代码保持一致
target = 0
X1 = Data_handing.all_encoder_inputs[target]
X2 = Data_handing.all_decoder_inputs[target]
Y = Data_handing.all_decoder_targets[target]

# 固定的维度参数
input_size = X1.shape[2]
output_dim = Y.shape[2]
decoder_input_dim = X2.shape[2]

# 定义目标函数 - 保持原有逻辑
def objective(trial):
    # 定义超参数搜索空间
    enc_hid_dim = trial.suggest_int("enc_hid_dim", 32, 512, step=16)
    dec_hid_dim = trial.suggest_int("dec_hid_dim", 32, 512, step=16)
    enc_num_layers = trial.suggest_int("enc_num_layers", 1, 4, step=1)
    dec_num_layers = trial.suggest_int("dec_num_layers", 1, 4, step=1)
    batch_size = trial.suggest_int("batch_size", 16, 128, step=16)
    n_heads = trial.suggest_categorical("n_heads", [1, 2, 4, 8, 16])
    weight_decay = 0.001
    lookahead_alpha = trial.suggest_float("lookahead_alpha", 0.2, 0.8, step=0.1)
    lookahead_k = trial.suggest_int("lookahead_k", 2, 20, step=2)
    lr = trial.suggest_float("lr", 1e-7, 1e-4, log=True)

    max_epochs = 1000
    dropout = 0.2
    patience = 200
    no_improve_epochs = 0

    # 创建模型
    encoder = seq2seq_attention.Encoder(
        input_size, enc_hid_dim, enc_num_layers, dec_hid_dim
    ).to(device)
    attention = seq2seq_attention.MultiHeadAttention(enc_hid_dim, dec_hid_dim, n_heads).to(device)
    decoder = seq2seq_attention.Decoder(
        decoder_input_dim, output_dim, enc_hid_dim, dec_hid_dim,
        dropout, attention, dec_num_layers
    ).to(device)
    model = seq2seq_attention.Seq2Seq(encoder, decoder, device).to(device)

    # 优化器
    adamw_optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        betas=(0.9, 0.99),
        eps=1e-08,
        weight_decay=weight_decay
    )
    optimizer = Lookahead(adamw_optimizer, k=lookahead_k, alpha=lookahead_alpha)
    criterion = nn.MSELoss()

    # 数据加载器
    train_dataset = TensorDataset(X1, X2, Y)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # 训练模型
    best_loss = float('inf')
    start_time = time.time()

    for epoch in range(max_epochs):
        model.train()
        epoch_train_loss = 0

        for src, trg, Y_batch in train_loader:
            src = src.transpose(0, 1).to(device)
            trg = trg.transpose(0, 1).to(device)
            Y_batch = Y_batch.transpose(0, 1).to(device)

            optimizer.zero_grad()
            outputs = model(src, trg, Y_batch)
            outputs = outputs[1:]

            loss = criterion(outputs, Y_batch)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()

        avg_train_loss = epoch_train_loss / len(train_loader)

        # 早停检查
        if avg_train_loss < best_loss:
            best_loss = avg_train_loss
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= patience:
                print(f"早停在第 {epoch + 1} 轮，最佳损失: {best_loss:.7f}")
                break

        if (epoch + 1) % 100 == 0:
            print(f"Trial {trial.number}, Epoch {epoch + 1}/{max_epochs}, Loss: {avg_train_loss:.7f}, LR: {lr:.8f}")

        trial.report(avg_train_loss, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    training_time = time.time() - start_time
    print(f"Trial {trial.number} 完成，最佳损失: {best_loss:.7f}，训练时间: {training_time:.2f}秒")

    # 保存最佳模型（修改为新路径）
    frozen_trial = next(t for t in trial.study.trials if t.number == trial.number)
    if frozen_trial.state == TrialState.COMPLETE:
        completed_trials = [t for t in trial.study.trials if t.state == TrialState.COMPLETE]
        if len(completed_trials) > 0:
            current_best = min(completed_trials, key=lambda t: t.value)
            if trial.number == current_best.number:
                folder_path = r'G:'
                os.makedirs(folder_path, exist_ok=True)
                file_name = f'best_optuna_model_trial_{trial.number}.pth'
                full_path = os.path.join(folder_path, file_name)
                torch.save(model, full_path)
                print(f"保存最佳模型到: {full_path}")

    return best_loss

if __name__ == "__main__":
    seed = 42
    # 固定随机种子
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    # 修改输出目录为 G:（包含所有可视化和数据文件）
    output_dir = r'G:'
    os.makedirs(output_dir, exist_ok=True)

    # 创建Optuna研究
    study = optuna.create_study(
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=200),
        sampler=optuna.samplers.TPESampler(seed=seed)
    )

    # 运行优化
    study.optimize(objective, n_trials=50, show_progress_bar=True)

    # 输出结果统计
    pruned_trials = [t for t in study.trials if t.state == TrialState.PRUNED]
    complete_trials = [t for t in study.trials if t.state == TrialState.COMPLETE]

    print("\n优化完成!")
    print(f"试验总数: {len(study.trials)}")
    print(f"剪枝试验: {len(pruned_trials)}")
    print(f"完成试验: {len(complete_trials)}")

    if len(complete_trials) > 0:
        print("\n最佳试验:")
        best_trial = study.best_trial
        print(f"  损失: {best_trial.value:.7f}")
        print("  超参数:")
        for key, value in best_trial.params.items():
            print(f"    {key}: {value}")

        # 保存所有试验结果到Excel
        results = []
        for trial in study.trials:
            results.append({
                "trial_number": trial.number,
                "value": trial.value,
                "state": trial.state.name,** trial.params
            })
        df = pd.DataFrame(results)
        excel_path = os.path.join(output_dir, "optuna_results.xlsx")
        df.to_excel(excel_path, index=False)
        print(f"\n所有试验结果已保存到: {excel_path}")