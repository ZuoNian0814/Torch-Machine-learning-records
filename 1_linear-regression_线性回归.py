import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.Linear_layer = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.Linear_layer(x)

w, b, noise_range = 2.3, -8, 0.1
batch_size = 100
# 输入 x
x = torch.randint(low=-50, high=51, size=(batch_size, 1), dtype=torch.float32)
# 输出 y
y = w * x + b
# 噪声
noise = 2 * (torch.rand_like(y) * noise_range) - 1
# 4. 整数叠加噪声，得到最终结果
y = y + noise

# 线性回归的输入和输出均只有一个维度
model = Model(1, 1)
# 定义损失函数，这里使用均方误差损失（MSELoss）
criterion = nn.MSELoss()
# 定义优化器，使用随机梯度下降（SGD）来更新权重和偏置
optimizer = torch.optim.SGD(model.parameters(), lr=0.0008)

epochs = 1000
for epoch in range(1000):
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

    if (epoch + 1) % 50 == 0:
        print(f'[epoch {epoch+1}]loss:', loss.item())

model.eval()
x1 = torch.randint(low=-100, high=101, size=(1, 1), dtype=torch.float32)
x2 = torch.randint(low=-100, high=101, size=(1, 1), dtype=torch.float32)

y_p1 = model(x1)
y_p2 = model(x2)
y1 = w * x1 + b
y2 = w * x2 + b

print('真实值1:', y1.item())
print('预测值1:', y_p1.item())

print('真实值2:', y2.item())
print('预测值2:', y_p2.item())

weight = model.Linear_layer.weight.data  # 获取权重
bias = model.Linear_layer.bias.data      # 获取偏置
print(f"训练权重(w): {weight.item()}, 训练偏置(b): {bias.item()}")
print(f"真实权重(w): {w}, 真实偏置(b): {b}")
