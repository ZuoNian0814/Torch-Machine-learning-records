import torch
import torch.nn as nn

class ResBlock(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=None):
        super().__init__()
        if not hidden_dim:
            hidden_dim = input_dim

        self.layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )
        self.output_layer = nn.Linear(hidden_dim, output_dim)

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
            ResBlock(hidden_dim, hidden_dim),
            nn.ReLU(),
            ResBlock(hidden_dim, hidden_dim)
        )
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out = self.layer(x)
        out = self.output_layer(out)
        return out

batch_size = 8
input_tensor = torch.rand((batch_size, 3))

model = MainModel(3, 5, 16)

# 前向传播过程
output_tensor = model(input_tensor)

print('input_tensor:', input_tensor.shape)
print('output_tensor:', output_tensor.shape)
