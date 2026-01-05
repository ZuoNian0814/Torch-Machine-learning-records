import torch
from torch_geometric.nn import GCNConv
import torch.nn as nn
from torch_geometric.utils import scatter  # 用于批次图池化
import random
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

batch_size = 1
num_nodes = 5
input_dim = 4
output_dim = 16

# 节点特征张量
input_tensor = torch.rand((batch_size, num_nodes, input_dim))
# 边索引张量（这里只展示了10种连接，实际上每个节点都可以相互连接）
# 0→9，1→8，2→7，是一种有向连接，索引对应的是节点中的对应位置
edge_index = torch.tensor([
    [0, 1, 2, 3, 4],
    [4, 3, 2, 1, 0],
], dtype=torch.long)

conv = GCNConv(input_dim, output_dim)

output_tensor = conv(input_tensor, edge_index)

print('输入张量：', input_tensor)
print('边索引张量：', edge_index)
print('输出张量：', output_tensor.shape)

class Model(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(Model, self).__init__()
        self.conv1 = GCNConv(input_dim, 16)
        self.conv2 = GCNConv(16, 32)
        self.relu = nn.ReLU()
        self.output_layer = nn.Linear(32, num_classes)
        self.softmax = nn.Softmax()

    def forward(self, batch):
        x, edge_index, batch_idx = batch.x, batch.edge_index, batch.batch
        device = next(self.parameters()).device
        x, edge_index, batch_idx = x.to(device), edge_index.to(device), batch_idx.to(device)
        x = self.conv1(x, edge_index)
        x = self.relu(x)
        x = self.conv2(x, edge_index)
        x = self.relu(x)
        x = scatter(x, batch_idx, dim=0, reduce='mean')  # 关键：区分不同图的节点
        x = self.output_layer(x)
        return x

def get_data(data_size):
    graph_datas = []
    progress = 0
    for _ in range(data_size):
        progress += 1
        a = []
        while True:
            a.append(random.randint(0, 5) + random.random())
            if sum(a) >= 500:
                break

        b = []
        while True:
            b.append(random.randint(0, 10) + random.random())
            if sum(b) >= 1000:
                break

        c = []
        while True:
            c.append(random.randint(0, 10) + random.random())
            if sum(c) >= 1000:
                break

        y = random.randint(0, 1)
        if y:
            a_0, b_0, c_0 = sum(random.choices(a, k=int(len(a) * 0.6))), sum(random.choices(b, k=int(len(b) * 0.6))), sum(random.choices(c, k=int(len(c) * 0.4))),
            a_1, b_1, c_1 = sum(random.choices(a, k=int(len(a) * 0.4))), sum(random.choices(b, k=int(len(b) * 0.4))), sum(random.choices(c, k=int(len(c) * 0.4))),
        else:
            a_1, b_1, c_1 = sum(random.choices(a, k=int(len(a) * 0.6))), sum(random.choices(b, k=int(len(b) * 0.4))), sum(random.choices(c, k=int(len(c) * 0.4))),
            a_0, b_0, c_0 = sum(random.choices(a, k=int(len(a) * 0.4))), sum(random.choices(b, k=int(len(b) * 0.4))), sum(random.choices(c, k=int(len(c) * 0.4))),

        x = []
        weights = [0.6, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05]
        for weight in weights:
            a, b, c = a_0 * weight, b_0 * weight, c_0 * weight
            x.append([a, b, c])
        for weight in weights:
            a, b, c = a_1 * weight, b_1 * weight, c_1 * weight
            x.append([a, b, c])

        x = torch.tensor(x, dtype=torch.float)
        edge_index = torch.tensor([
            [0, 0, 0, 1, 1, 2, 2, 7, 7, 7, 8, 8, 9, 9],
            [1, 2, 7, 3, 4, 5, 6, 0, 8, 9, 10, 11, 12, 13]
        ], dtype=torch.long)
        y = torch.tensor([y], dtype=torch.long)

        data = Data(x=x, edge_index=edge_index, y=y)
        graph_datas.append(data)
    return graph_datas

# 训练数据
batch_size = 32
dataset = get_data(300)

cut_num = int(len(dataset)*0.7)
# 分离训练集和验证集
train_dataset = dataset[:cut_num]
val_dataset = dataset[cut_num:]
train_loader = DataLoader(train_dataset, batch_size=batch_size)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

for batch in train_loader:
    print("批次数据信息：", batch)
    break

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

input_dim = dataset[0].num_node_features
num_classes = 2

model = Model(input_dim, num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0008)

epochs = 16
for epoch in range(epochs):
    model.train()
    train_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        out = model(batch)
        y = batch.y.to(device)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    train_loss /= len(train_loader)

    # 验证
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    model.eval()
    for batch in val_loader:
        out = model(batch)
        y = batch.y.to(device)
        loss = criterion(out, y)
        val_loss += loss.item()
        pred = out.argmax(dim=1)
        correct += int((pred == y).sum())
        total += batch.y.size(0)
    val_loss /= len(val_loader)
    val_acc = correct / total

    if (epoch + 1) % 4 == 0:
        print(f'Epoch[{epoch + 1}] Train Loss: {train_loss}')
        print(f'\tVal Loss: {val_loss}, Val Acc: {val_acc*100:.2f}%')