import torch
import torch.nn as nn
from torch_geometric.utils import scatter  # 用于批次图池化
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

class Model(nn.Module):
    def __init__(self, num_nodes, embedding_dim, output_dim, hidden_dim):
        super(Model, self).__init__()
        # 嵌入层，将节点映射到嵌入空间
        self.embedding = nn.Embedding(num_nodes, embedding_dim)
        self.conv1 = GCNConv(embedding_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

        self.relu = nn.ReLU()

    def forward(self, data):
        x, edge_index, batch_idx = data.x, data.edge_index, data.batch
        device = next(self.parameters()).device
        x, edge_index, batch_idx = x.to(device), edge_index.to(device), batch_idx.to(device)

        # 使用嵌入层将节点索引映射为节点特征
        x = self.embedding(x)
        x = self.conv1(x, edge_index)
        x = self.relu(x)
        x = self.conv2(x, edge_index)
        x = self.relu(x)

        x = scatter(x, batch_idx, dim=0, reduce='mean')  # 关键：区分不同图的节点
        # 边选择
        edge_score = self.fc(x)
        return edge_score

batch_size = 8
num_nodes = 5
output_dim = 1
embedding_dim = 16
hidden_dim = 32

input_tensor = torch.arange(num_nodes, dtype=torch.long)

edge_index = torch.tensor([
    [0, 1, 2, 3, 4, 0],
    [4, 3, 3, 1, 0, 2],
], dtype=torch.long)

model = Model(num_nodes=5, embedding_dim=16, output_dim=1, hidden_dim=16)

data = Data(
    x=input_tensor,
    edge_index=edge_index,
)
loader = DataLoader([data for _ in range(batch_size)], batch_size=batch_size)
for batch in loader:
    output_tensor = model(batch)

    print('输入张量：', batch.x.shape)
    print('边索引张量：', batch.edge_index.shape)
    print('输出张量：', output_tensor.shape)