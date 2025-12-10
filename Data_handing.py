import torch
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np


# 读取数据
data1 = pd.read_excel(r"G:.xlsx")  #土体累积沉降
data2 = pd.read_excel(r"G:.xlsx")  #土体沉降速率
data3 = pd.read_excel(r"G:.xlsx")  #真空度


data1 = data1.iloc[:, 1:].values
data2 = data2.iloc[:, 1:].values
data3 = data3.iloc[:, 1:].values


# 提取特征
feature_11 = data1[:, 0]
feature_12 = data2[:, 0]
feature_13 = data3[:, 0]

feature_21 = data1[:, 1]
feature_22 = data2[:, 1]
feature_23 = data3[:, 1]

feature_31 = data1[:, 2]
feature_32 = data2[:, 2]
feature_33 = data3[:, 2]

feature_41 = data1[:, 3]
feature_42 = data2[:, 3]
feature_43 = data3[:, 5]

feature_51 = data1[:, 4]
feature_52 = data2[:, 4]
feature_53 = data3[:, 7]

feature_61 = data1[:, 5]
feature_62 = data2[:, 5]
feature_63 = data3[:, 8]

feature_71 = data1[:, 6]
feature_72 = data2[:, 6]
feature_73 = data3[:, 9]

feature_81 = data1[:, 7]
feature_82 = data2[:, 7]
feature_83 = data3[:, 10]

feature_91 = data1[:, 8]
feature_92 = data2[:, 8]
feature_93 = data3[:, 11]

# 将特征分组
features_1 = [feature_11, feature_21, feature_31, feature_41, feature_51, feature_61, feature_71, feature_81, feature_91]
features_2 = [feature_12, feature_22, feature_32, feature_42, feature_52, feature_62, feature_72, feature_82, feature_92]
features_3 = [feature_13, feature_23, feature_33, feature_43, feature_53, feature_63, feature_73, feature_83, feature_93]

# 用于存储训练数据集的结果
all_encoder_inputs_list = []
all_decoder_inputs_list = []
all_decoder_targets_list = []

# 用于存储测试数据集的结果
all_test_encoder_inputs_list = []
all_test_decoder_inputs_list = []
all_test_decoder_targets_list = []



# # 用于存储每次循环的归一化器
all_scalers_1 = []
all_scalers_2 = []
all_scalers_3 = []

# 进行九次交叉验证的数据归一化
for i in range(9):

    # 准备训练集和测试集
    train_features_1 = np.column_stack(features_1[:i] + features_1[i + 1:])
    test_feature_1 = features_1[i].reshape(-1, 1)

    train_features_2 = np.column_stack(features_2[:i] + features_2[i + 1:])
    test_feature_2 = features_2[i].reshape(-1, 1)

    train_features_3 = np.column_stack(features_3[:i] + features_3[i + 1:])
    test_feature_3 = features_3[i].reshape(-1, 1)

    all_features_1=np.column_stack(features_1).reshape(-1, 1)
    all_features_2=np.column_stack(features_2).reshape(-1, 1)
    all_features_3=np.column_stack(features_3).reshape(-1, 1)


    # 调整训练集形状为 (148 * 8, 1)
    train_features_1 = train_features_1.reshape(-1, 1)
    train_features_2 = train_features_2.reshape(-1, 1)
    train_features_3 = train_features_3.reshape(-1, 1)

    # 创建归一化器
    scaler_1 = MinMaxScaler()
    scaler_2 = MinMaxScaler()
    scaler_3 = MinMaxScaler()

    # 对训练集进行归一化
    all_train_features_1_scaled = scaler_1.fit_transform(all_features_1)
    all_train_features_2_scaled = scaler_2.fit_transform(all_features_2)
    all_train_features_3_scaled = scaler_3.fit_transform(all_features_3)

    # 对训练集进行归一化
    train_features_1_scaled = scaler_1.transform(train_features_1)
    train_features_2_scaled = scaler_2.transform(train_features_2)
    train_features_3_scaled = scaler_3.transform(train_features_3)

    # 保存本次循环的归一化器
    all_scalers_1.append(scaler_1)
    all_scalers_2.append(scaler_2)
    all_scalers_3.append(scaler_3)

    # 使用训练集的归一化参数对测试集进行归一化
    test_feature_1_scaled = scaler_1.transform(test_feature_1)
    test_feature_2_scaled = scaler_2.transform(test_feature_2)
    test_feature_3_scaled = scaler_3.transform(test_feature_3)

    # 进行测试集数据生成
    scaled_feature_x = np.column_stack([test_feature_1_scaled, test_feature_2_scaled, test_feature_3_scaled])
    # 构建解码器输入
    feature_decoder_x = scaled_feature_x[:, 2]
    # 标签数据
    label_x = scaled_feature_x[:, 0]
    # 张量形式
    tensor_feature_x = torch.tensor(scaled_feature_x, dtype=torch.float32)  # 确保是浮点型
    tensor_feature_decoder_x = torch.tensor(feature_decoder_x, dtype=torch.float32)  # 确保是浮点型
    tensor_label_x = torch.tensor(label_x, dtype=torch.float32)  # 确保是浮点型

    # 定义序列长度
    encoder_seq_len = 10  # 编码器输入序列长度
    decoder_seq_len = 10  # 解码器目标序列长度
    total_seq_len = encoder_seq_len + decoder_seq_len  # 总序列长度

    # 计算新样本数
    n_samples = tensor_feature_x.shape[0] - total_seq_len + 1

    # 初始化编码器输入和解码器目标列表
    encoder_inputs = []
    decoder_inputs = []
    decoder_targets = []

    # 滑动窗口采样
    for k in range(n_samples):
        # 提取编码器输入
        encoder_input = tensor_feature_x[k:k + encoder_seq_len]
        # 提取解码器输入
        decoder_input = tensor_feature_decoder_x[k + encoder_seq_len:k + total_seq_len]
        # 提取解码器输出
        decoder_target = tensor_label_x[k + encoder_seq_len:k + total_seq_len]

        encoder_inputs.append(encoder_input)
        decoder_inputs.append(decoder_input)
        decoder_targets.append(decoder_target)
    # 将列表转换为张量
    test_encoder_inputs = torch.stack(encoder_inputs)  # [sample, encoder_seq_len, input_size]
    test_decoder_inputs = torch.stack(decoder_inputs)  # [sample, encoder_seq_len]
    test_decoder_targets = torch.stack(decoder_targets)  # [sample, decoder_seq_len]

    # 在最后一个维度上插入新的维度
    test_decoder_inputs = test_decoder_inputs.unsqueeze(-1)
    test_decoder_targets = test_decoder_targets.unsqueeze(-1)

    all_test_encoder_inputs_list.append(test_encoder_inputs)
    all_test_decoder_inputs_list.append(test_decoder_inputs)
    all_test_decoder_targets_list.append(test_decoder_targets)

    # 存储本次循环的所有结果
    encoder_inputs_list = []
    decoder_inputs_list = []
    decoder_targets_list = []

    # 进行训练集数据生成
    for j in range(1, 10):
        if j == i + 1:
            continue  # 只对训练集做数据整理
        # 只处理测试集生成时间窗口数据
        scaled_feature_x = np.column_stack((scaler_1.transform(globals()[f"feature_{j}1"].reshape(-1, 1)),
                                            scaler_2.transform(globals()[f"feature_{j}2"].reshape(-1, 1)),
                                            scaler_3.transform(globals()[f"feature_{j}3"].reshape(-1, 1))))

        # 构建解码器输入
        feature_decoder_x = scaled_feature_x[:, 2]
        # 标签数据
        label_x = scaled_feature_x[:, 0]
        # 张量形式
        tensor_feature_x = torch.tensor(scaled_feature_x, dtype=torch.float32)  # 确保是浮点型
        tensor_feature_decoder_x = torch.tensor(feature_decoder_x, dtype=torch.float32)  # 确保是浮点型
        tensor_label_x = torch.tensor(label_x, dtype=torch.float32)  # 确保是浮点型

        # 计算新样本数
        n_samples = tensor_feature_x.shape[0] - total_seq_len + 1

        # 初始化编码器输入和解码器目标列表
        encoder_inputs = []
        decoder_inputs = []
        decoder_targets = []

        # 滑动窗口采样
        for k in range(n_samples):
            # 提取编码器输入
            encoder_input = tensor_feature_x[k:k + encoder_seq_len]
            # 提取解码器输入
            decoder_input = tensor_feature_decoder_x[k + encoder_seq_len:k + total_seq_len]
            # 提取解码器输出
            decoder_target = tensor_label_x[k + encoder_seq_len:k + total_seq_len]

            encoder_inputs.append(encoder_input)
            decoder_inputs.append(decoder_input)
            decoder_targets.append(decoder_target)

        # 将列表转换为张量
        encoder_inputs = torch.stack(encoder_inputs)  # [sample, encoder_seq_len, input_size]
        decoder_inputs = torch.stack(decoder_inputs)  # [sample, encoder_seq_len]
        decoder_targets = torch.stack(decoder_targets)  # [sample, decoder_seq_len]

        # 在最后一个维度上插入新的维度
        decoder_inputs = decoder_inputs.unsqueeze(-1)
        decoder_targets = decoder_targets.unsqueeze(-1)

        encoder_inputs_list.append(encoder_inputs)
        decoder_inputs_list.append(decoder_inputs)
        decoder_targets_list.append(decoder_targets)

    # 拼接本次循环的所有结果
    encoder_inputs_combined = torch.cat(encoder_inputs_list, dim=0)
    decoder_inputs_combined = torch.cat(decoder_inputs_list, dim=0)
    decoder_targets_combined = torch.cat(decoder_targets_list, dim=0)

    all_encoder_inputs_list.append(encoder_inputs_combined)
    all_decoder_inputs_list.append(decoder_inputs_combined)
    all_decoder_targets_list.append(decoder_targets_combined)

# 在所有循环结束后，在新的维度上进行堆叠，形成四维张量   训练集
all_encoder_inputs = torch.stack(all_encoder_inputs_list, dim=0)
all_decoder_inputs = torch.stack(all_decoder_inputs_list, dim=0)
all_decoder_targets = torch.stack(all_decoder_targets_list, dim=0)

# 在所有循环结束后，在新的维度上进行堆叠，形成四维张量   测试集
all_test_encoder_inputs_list = torch.stack(all_test_encoder_inputs_list, dim=0)
all_test_decoder_inputs_list = torch.stack(all_test_decoder_inputs_list, dim=0)
all_test_decoder_targets_list = torch.stack(all_test_decoder_targets_list, dim=0)

print("all_test_encoder_inputs_list:", all_test_encoder_inputs_list.shape)
print("all_test_decoder_inputs_list:", all_test_decoder_inputs_list.shape)
print("all_test_decoder_targets_list:", all_test_decoder_targets_list.shape)

print("all_encoder_inputs shape:", all_encoder_inputs.shape)
print("all_decoder_inputs shape:", all_decoder_inputs.shape)
print("all_decoder_targets shape:", all_decoder_targets.shape)