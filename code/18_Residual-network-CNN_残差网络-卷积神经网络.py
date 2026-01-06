import torch
import torch.nn as nn

class ResBlock(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=None):
        super().__init__()
        if not hidden_dim:
            hidden_dim = input_dim

        self.layer = nn.Sequential(
            nn.Conv2d(input_dim, hidden_dim, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
        )
        self.output_layer = nn.Conv2d(hidden_dim, output_dim, kernel_size=3, padding=1, stride=1)

    def forward(self, x):
        out = self.layer(x)
        out = out + x       # 残差连接
        out = self.output_layer(out)
        return out

class MainModel(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim):
        super().__init__()
        self.layer = nn.Sequential(
            ResBlock(input_dim, hidden_dim),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            ResBlock(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            ResBlock(hidden_dim, hidden_dim),
            nn.MaxPool2d(2, 2),
            nn.Flatten()
        )
        self.output_layer = nn.Linear(1024, output_dim)

    def forward(self, x):
        out = self.layer(x)
        out = self.output_layer(out)
        return out

batch_size = 4
input_tensor = torch.rand((batch_size, 3, 64, 64))

model = MainModel(3, 5, 16)

# 前向传播过程
output_tensor = model(input_tensor)

print('input_tensor:', input_tensor.shape)
print('output_tensor:', output_tensor.shape)
