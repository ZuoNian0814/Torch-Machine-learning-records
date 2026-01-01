import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim):
        super().__init__()
        self.Linear_layer0 = nn.Linear(input_dim, hidden_dim)
        self.Linear_layer1 = nn.Linear(hidden_dim, hidden_dim)
        self.Linear_layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.Linear_layer3 = nn.Linear(hidden_dim, output_dim)
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
        return x

def get_data(data_size):
    # 2. 生成输入特征 [batch_size, 5]（5个特征）
    T_set = torch.rand(data_size, 1) * 14 + 16  # 16~30℃
    T_env = torch.rand(data_size, 1) * 18 + 20  # 20~38℃
    mode = torch.randint(0, 3, (data_size, 1))  # 0/1/2（运行模式）
    time_ratio = torch.rand(data_size, 1)       # 0~1（运行时长占比）
    people = torch.randint(1, 6, (data_size, 1))# 1~5人
    X = torch.cat([T_set, T_env, mode, time_ratio, people], dim=1)  # 拼接为[batch,5]

    # 3. 生成目标变量（耗电量）[batch_size, 1]
    base_power = 0.2  # 基础能耗
    # 温度差影响（分模式）
    temp_effect = torch.where(mode == 0, 0.08 * torch.max(T_env - T_set, torch.zeros_like(T_env)) * time_ratio,
                              torch.where(mode == 1, 0.1 * torch.max(T_set - T_env, torch.zeros_like(T_env)) * time_ratio,
                                          0.01 * time_ratio))
    # 人数影响
    people_effect = 0.05 * people * time_ratio
    # 随机噪声（-0.1~0.1）
    noise = (torch.rand(data_size, 1) - 0.5) * 0.1
    # 最终耗电量
    y = base_power + temp_effect + people_effect + noise

    # 最终数据形状：X→[128,5]（输入），y→[128,1]（目标），适配MLP输入格式
    return X, y

# 划分批次
def split_batch(data, batch_size):
    # 核心操作：沿第一个维度（dim=0）分割，保留后续所有维度
    split_tensors = torch.split(data, batch_size, dim=0)
    # 转为列表返回（torch.split返回tuple，列表更易操作）
    return list(split_tensors)

# 训练数据
train_x, train_y = get_data(1024)
train_x_batch = split_batch(train_x, 128)
train_y_batch = split_batch(train_y, 128)
# 验证数据
val_x, val_y = get_data(128)
# 测试数据
test_x, test_y = get_data(8)
print('输入数据形状:', train_x.shape)
print('输入批次数量:', len(train_x_batch), '\t批次形状:', train_x_batch[0].shape)
print('标签数据形状:', train_y.shape)
print('输入批次数量:', len(train_y_batch), '\t批次形状:', train_y_batch[0].shape)

# 线性回归的输入和输出均只有一个维度
model = Model(5, 1, 128)
# 定义损失函数，这里使用均方误差损失（MSELoss）
criterion = nn.MSELoss()
# 定义优化器，使用随机梯度下降（SGD）来更新权重和偏置
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

epochs = 1000
for epoch in range(1000):
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
    error = torch.sum((val_y - output) ** 2).item()
    if (epoch + 1) % 200 == 0:
        print(f'[epoch {epoch+1}]loss:', loss.item())
        print(f'\t val loss:', val_loss)
        print(f'\t val error:', error)


model.eval()
output = model(test_x)

for i in range(len(test_x)):
    print('输入数据:', test_x[i].tolist())
    print('目标结果:', test_y[i].item())
    print('预测结果:', output[i].item())