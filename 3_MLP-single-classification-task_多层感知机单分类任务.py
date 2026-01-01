import torch
import torch.nn as nn
import numpy as np

class Model(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim):
        super().__init__()
        self.Linear_layer0 = nn.Linear(input_dim, hidden_dim)
        self.Linear_layer1 = nn.Linear(hidden_dim, hidden_dim)
        self.Linear_layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.Linear_layer3 = nn.Linear(hidden_dim, hidden_dim)
        self.Linear_layer4 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.Linear_layer0(x)
        x = self.relu(x)
        x = self.Linear_layer1(x)
        x = self.relu(x)
        x = self.Linear_layer2(x)
        x = self.relu(x)
        x = self.Linear_layer3(x)
        x = self.relu(x)
        x = self.Linear_layer4(x)
        x = self.sigmoid(x)
        return x

def get_data(data_size):
    n_samples = data_size if data_size % 2 == 0 else data_size - 1
    n_pos = n_neg = n_samples // 2

    # --------------------- 1. 生成正例（圆内：x1² + x2² ≤ 4） ---------------------
    # 极坐标生成圆内均匀样本（避免随机采样的密度不均）
    pos_r = np.random.uniform(0, 2, n_pos)  # 半径0~2
    pos_theta = np.random.uniform(0, 2*np.pi, n_pos)  # 角度0~2π
    pos_x1 = pos_r * np.cos(pos_theta)
    pos_x2 = pos_r * np.sin(pos_theta)
    pos_x = np.vstack([pos_x1, pos_x2]).T  # (n_pos, 2)

    # --------------------- 2. 生成负例（圆环区：4 < x1² + x2² ≤ 25） ---------------------
    neg_r = np.random.uniform(2, 5, n_neg)  # 半径2~5
    neg_theta = np.random.uniform(0, 2*np.pi, n_neg)
    neg_x1 = neg_r * np.cos(neg_theta)
    neg_x2 = neg_r * np.sin(neg_theta)
    neg_x = np.vstack([neg_x1, neg_x2]).T  # (n_neg, 2)

    # --------------------- 3. 合并并打乱 ---------------------
    all_x = np.vstack([pos_x, neg_x])
    all_y = np.vstack([np.ones((n_pos, 1)), np.zeros((n_neg, 1))])

    # 随机打乱样本顺序
    shuffle_idx = np.random.permutation(n_samples)
    all_x = all_x[shuffle_idx]
    all_y = all_y[shuffle_idx]

    # 转换为PyTorch张量（float32，适配MLP）
    x = torch.tensor(all_x, dtype=torch.float32)
    y = torch.tensor(all_y, dtype=torch.float32)

    return x, y


# 划分批次
def split_batch(data, batch_size):
    # 核心操作：沿第一个维度（dim=0）分割，保留后续所有维度
    split_tensors = torch.split(data, batch_size, dim=0)
    # 转为列表返回（torch.split返回tuple，列表更易操作）
    return list(split_tensors)

# 训练数据
batch_size = 128
train_x, train_y = get_data(1024)
train_x_batch = split_batch(train_x, batch_size)
train_y_batch = split_batch(train_y, batch_size)
# 验证数据
val_x, val_y = get_data(128)
# 测试数据
test_x, test_y = get_data(6)
print('输入数据形状:', train_x.shape)
print('输入批次数量:', len(train_x_batch), '\t批次形状:', train_x_batch[0].shape)
print('标签数据形状:', train_y.shape)
print('输入批次数量:', len(train_y_batch), '\t批次形状:', train_y_batch[0].shape)

model = Model(2, 1, 32)
criterion = nn.BCELoss()  # 匹配带sigmoid的模型输出
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

epochs = 1000
for epoch in range(epochs):
    loss = None
    for i in range(len(train_x_batch)):
        x = train_x_batch[i]
        y = train_y_batch[i]
        model.train()
        # 前向传播，得到预测值
        output = model(x)
        # 计算损失
        loss = criterion(output, y)
        # 梯度清零，因为在每次反向传播前都要清除之前累积的梯度
        optimizer.zero_grad()
        # 反向传播，计算梯度
        loss.backward()
        # 更新权重和偏置
        optimizer.step()

    model.eval()
    output = model(val_x)
    val_loss = criterion(output, val_y).item()

    # 更改验证逻辑为适合分类任务的准确率和召回率
    ace = torch.sum((output > 0.5) == val_y).item() / val_y.shape[0]
    recall = (torch.sum((output > 0.5) & (val_y == 1))).item() / max(torch.sum(val_y).item(), 1)
    if (epoch + 1) % 200 == 0:
        print(f'[epoch {epoch+1}]loss:', loss.item())
        print(f'\t val loss:', val_loss)
        print(f'\t val ace: {ace * 100:.2f}%\trecall: {recall * 100:.2f}%')

model.eval()
output = model(test_x)

for i in range(len(test_x)):
    print('输入数据:', test_x[i].tolist())
    print('目标结果:', test_y[i].item())
    print('预测结果:', (output[i] > 0.5).item())