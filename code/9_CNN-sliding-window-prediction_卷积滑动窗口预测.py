import torch
import torch.nn as nn
import copy
import matplotlib.pyplot as plt
import numpy as np

def get_data(data_size, test_mode=False, step=0.1, seq_len=20):
    # ===================== 1. 定义核心参数 =====================
    num_samples = data_size  # 总数据条数
    total_points = seq_len + 1  # 每个序列的总点数（用于生成X+Y）
    if test_mode:
        total_points = seq_len * 2

    # ===================== 2. 生成随机起点 =====================
    # 随机起点范围：0 ~ 10
    start_points = torch.rand(num_samples) * 10

    # ===================== 3. 生成21个等步长的数值序列 =====================
    # 生成步长序列
    step_sequence = torch.arange(0, total_points * step, step)  # 避免浮点精度问题，用arange直接生成
    # 扩展维度，便于和起点广播计算
    step_sequence = step_sequence.unsqueeze(0)
    start_points = start_points.unsqueeze(1)
    # 生成所有序列的21个点
    all_points = start_points + step_sequence

    # ===================== 4. 计算正弦值 =====================
    sin_values = torch.sin(all_points)

    # ===================== 5. 拆分X和Y =====================
    X = sin_values[:, :seq_len]
    Y = sin_values[:, seq_len:]

    if test_mode:
        return X, sin_values
    return X, Y

# 划分批次
def split_batch(data, batch_size):
    # 核心操作：沿第一个维度（dim=0）分割，保留后续所有维度
    split_tensors = torch.split(data, batch_size, dim=0)
    # 转为列表返回（torch.split返回tuple，列表更易操作）
    return list(split_tensors)

# 训练数据
seq_len = 24
batch_size = 128
train_x, train_y = get_data(240, seq_len=seq_len)
train_x_batch = split_batch(train_x, batch_size)
train_y_batch = split_batch(train_y, batch_size)
# 验证数据
val_x, val_y = get_data(32, seq_len=seq_len)
# 测试数据
test_x, test_y = get_data(3, test_mode=True, seq_len=seq_len)
print('输入数据形状:', train_x.shape)
print('输入批次数量:', len(train_x_batch), '\t批次形状:', train_x_batch[0].shape)
print('标签数据形状:', train_y.shape)
print('输入批次数量:', len(train_y_batch), '\t批次形状:', train_y_batch[0].shape)

class Model(nn.Module):
    def __init__(self, input_dim=1, seg_dim=24, output_dim=1, hidden_dim=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Flatten(),
        )

        self.output_layer = nn.Linear(hidden_dim * seg_dim, output_dim)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.encoder(x)
        x = self.output_layer(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Model(input_dim=1, seg_dim=24, output_dim=1, hidden_dim=32).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

epochs = 100
for epoch in range(epochs):
    loss = None
    model.train()
    for i in range(len(train_x_batch)):
        x = train_x_batch[i].to(device)
        y = train_y_batch[i].to(device)

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

    val_x = val_x.to(device)
    val_y = val_y.to(device)
    model.eval()
    output = model(val_x)
    val_loss = criterion(output, val_y).item()

    # 更改验证逻辑为适合分类任务的准确率和召回率
    if (epoch + 1) % 30 == 0:
        print(f'[epoch {epoch+1}]loss:', loss.item())
        print(f'\t val loss:', val_loss)

model.eval()
test_x = test_x.to(device)
output = model(test_x)

input_x = copy.deepcopy(test_x)
data_list = [test_x]
for i in range(seq_len):
    input_data = input_x[:, i:i+seq_len]
    output_num = model(input_data)
    data_list.append(output_num)
    input_x = torch.cat([input_x, output_num], dim=1)

output_data = torch.cat(data_list, dim=1)

output_data = output_data.detach().to('cpu').numpy()
test_data = test_y.detach().to('cpu').numpy()

plt.rcParams['font.sans-serif'] = ['SimHei']  # 黑体
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题
plt.figure(figsize=(8, 12))

# 按样本拆分到3个子图（i=0/i=1/i=2）
for idx, i in enumerate(range(min(3, len(test_x)))):  # 取前3个样本
    plt.subplot(3, 1, idx+1)
    output_data_i = output_data[i]
    test_data_i = test_data[i]
    test_x_np = np.arange(len(test_data_i))
    output_x_np = np.arange(len(output_data_i))

    plt.scatter(test_x_np, test_data_i, c='blue', label='真值', s=20)
    plt.scatter(output_x_np[seq_len:], output_data_i[seq_len:], c='red', label='预测值', s=10)

    plt.axvline(
        x=seq_len,          # 指定x坐标
        linestyle='--',      # 虚线样式（--虚线，-.点划线，:点线）
        color='green',       # 线条颜色
        linewidth=1.5,       # 线条宽度
        alpha=0.8            # 透明度
    )

    plt.title(f'样本{i} - 真实vs预测', fontsize=12)
    plt.xlabel('索引'), plt.ylabel('数值')
    plt.legend(), plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()