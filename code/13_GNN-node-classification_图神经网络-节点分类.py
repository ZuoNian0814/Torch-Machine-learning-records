import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
import random
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

class ModelNode(nn.Module):
    def __init__(self, num_node_features, num_classes):
        super(ModelNode, self).__init__()
        self.conv1 = GCNConv(num_node_features, 16)
        self.conv2 = GCNConv(16, 32)
        self.conv3 = GCNConv(32, num_classes)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        device = next(self.parameters()).device
        x, edge_index = x.to(device), edge_index.to(device)
        x = self.conv1(x, edge_index)
        x = self.relu(x)
        x = self.conv2(x, edge_index)
        x = self.relu(x)
        x = self.conv3(x, edge_index)
        return x

def get_data(data_size):
    graph_datas = []
    for _ in range(data_size):
        x = []
        y = []
        # 数据生成
        node_num = 1200
        for n in range(node_num):
            max_num = random.randint(0, 5)

            num_1 = random.randint(0, 100)
            num_2 = random.randint(0, 100 - num_1)
            num_3 = random.randint(0, 100 - num_1 - num_2)
            num_4 = random.randint(0, 100 - num_1 - num_2 - num_3)
            num_5 = random.randint(0, 100 - num_1 - num_2 - num_3 - num_4)
            num_6 = 100 - num_1 - num_2 - num_3 - num_4 - num_5
            if max_num == 0:
                x_list = [num_1, num_2, num_3, num_4, num_5, num_6]
            elif max_num == 1:
                x_list = [num_2, num_1, num_3, num_4, num_5, num_6]
            elif max_num == 2:
                x_list = [num_3, num_2, num_1, num_4, num_5, num_6]
            elif max_num == 3:
                x_list = [num_4, num_2, num_3, num_1, num_5, num_6]
            elif max_num == 4:
                x_list = [num_5, num_2, num_3, num_4, num_1, num_6]
            else:
                x_list = [num_6, num_2, num_3, num_4, num_5, num_1]

            x.append(x_list)
            y_num = x_list.index(max(x_list))
            x_list[y_num] += 50
            y.append(y_num)

        x = torch.tensor(x, dtype=torch.float)
        y = torch.tensor(y, dtype=torch.long)
        edge = [[], []]
        for n in range(120):
            a, b = random.choices(list(range(0, node_num-1)), k=2)
            edge[0].append(a)
            edge[1].append(b)
        edge_index = torch.tensor(edge, dtype=torch.long)

        data2 = Data(x=x, edge_index=edge_index, y=y)
        graph_datas.append(data2)
    return graph_datas

# 训练数据
batch_size = 32
dataset = get_data(640)
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
num_node_features = dataset[0].num_node_features
num_classes = 6

model = ModelNode(num_node_features, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 60
# 训练模型
model.train()
for epoch in range(epochs):
    train_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        out = model(batch)
        loss = criterion(out, batch.y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)
    # 验证
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in val_loader:
            out = model(batch)
            loss = criterion(out.view(-1, num_classes), batch.y.view(-1))
            val_loss += loss.item()
            pred = out.argmax(dim=1)
            correct += int((pred == batch.y).sum())
            total += batch.y.size(0)
    val_loss /= len(val_loader)
    val_acc = correct / total

    if (epoch + 1) % 12 == 0:
        print(f'Epoch[{epoch + 1}] Train Loss: {train_loss}')
        print(f'\tVal Loss: {val_loss} \tVal Acc: {val_acc}')