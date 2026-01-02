import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim):
        super().__init__()
        self.Linear_layer0 = nn.Linear(input_dim, hidden_dim)
        self.Linear_layer1 = nn.Linear(hidden_dim, hidden_dim)
        self.Linear_layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.Linear_layer3 = nn.Linear(hidden_dim, hidden_dim)
        self.Linear_layer4 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

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
        return x

import numpy as np

def get_data(data_size):
    # 确保总样本数是5的倍数，每类样本数相等
    total_samples = data_size if data_size % 5 == 0 else data_size - (data_size % 5)
    samples_per_class = total_samples // 5

    # --------------------- 1. 生成类别0：中心圆形区（x1² + x2² ≤ 1） ---------------------
    r0 = np.random.uniform(0, 1, samples_per_class)  # 半径0~1
    theta0 = np.random.uniform(0, 2 * np.pi, samples_per_class)  # 角度0~2π
    x1_0 = r0 * np.cos(theta0)
    x2_0 = r0 * np.sin(theta0)
    class0 = np.vstack([x1_0, x2_0]).T

    # --------------------- 2. 生成类别1：第一象限环区（x1>0, x2>0, 1<r≤6） ---------------------
    r1 = np.random.uniform(1, 6, samples_per_class)
    theta1 = np.random.uniform(0, np.pi/2, samples_per_class)  # 第一象限角度范围
    x1_1 = r1 * np.cos(theta1)
    x2_1 = r1 * np.sin(theta1)
    class1 = np.vstack([x1_1, x2_1]).T

    # --------------------- 3. 生成类别2：第二象限环区（x1<0, x2>0, 1<r≤6） ---------------------
    r2 = np.random.uniform(1, 6, samples_per_class)
    theta2 = np.random.uniform(np.pi/2, np.pi, samples_per_class)  # 第二象限角度范围
    x1_2 = r2 * np.cos(theta2)
    x2_2 = r2 * np.sin(theta2)
    class2 = np.vstack([x1_2, x2_2]).T

    # --------------------- 4. 生成类别3：第三象限环区（x1<0, x2<0, 1<r≤6） ---------------------
    r3 = np.random.uniform(1, 6, samples_per_class)
    theta3 = np.random.uniform(np.pi, 3*np.pi/2, samples_per_class)  # 第三象限角度范围
    x1_3 = r3 * np.cos(theta3)
    x2_3 = r3 * np.sin(theta3)
    class3 = np.vstack([x1_3, x2_3]).T

    # --------------------- 5. 生成类别4：第四象限环区（x1>0, x2<0, 1<r≤6） ---------------------
    r4 = np.random.uniform(1, 6, samples_per_class)
    theta4 = np.random.uniform(3*np.pi/2, 2*np.pi, samples_per_class)  # 第四象限角度范围
    x1_4 = r4 * np.cos(theta4)
    x2_4 = r4 * np.sin(theta4)
    class4 = np.vstack([x1_4, x2_4]).T

    # --------------------- 合并所有类别并生成标签 ---------------------
    # 合并特征
    all_x = np.vstack([class0, class1, class2, class3, class4])
    # 生成标签（0~4各samples_per_class个）
    all_y = np.hstack([
        np.zeros(samples_per_class, dtype=np.int64),   # 类别0
        np.ones(samples_per_class, dtype=np.int64),    # 类别1
        2 * np.ones(samples_per_class, dtype=np.int64),# 类别2
        3 * np.ones(samples_per_class, dtype=np.int64),# 类别3
        4 * np.ones(samples_per_class, dtype=np.int64) # 类别4
    ])

    # --------------------- 随机打乱样本顺序 ---------------------
    shuffle_idx = np.random.permutation(total_samples)
    all_x = all_x[shuffle_idx]
    all_y = all_y[shuffle_idx]

    # --------------------- 转换为PyTorch张量 ---------------------
    x = torch.tensor(all_x, dtype=torch.float32)  # 特征为float32（MLP输入要求）
    y = torch.tensor(all_y, dtype=torch.long)     # 标签为long（CrossEntropyLoss要求）

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

model = Model(2, 5, 32)
# 定义损失函数（这里使用交叉熵损失，适用于分类任务）
criterion = nn.CrossEntropyLoss()
# 定义优化器（这里使用Adam优化器，学习率等参数可调整）
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

epochs = 200
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
    if (epoch + 1) % 40 == 0:
        print(f'[epoch {epoch+1}]loss:', loss.item())
        print(f'\t val loss:', val_loss)

model.eval()
output = model(test_x)

for i in range(len(test_x)):
    print('输入数据:', test_x[i].tolist())
    print('目标结果:', test_y[i].item())
    print('预测结果:', torch.argmax(output, dim=1)[i].item())
