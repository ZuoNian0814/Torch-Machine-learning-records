import torch
import torch.nn as nn
import random

# 嵌入位置编码
class PosEncoding(nn.Module):
    def __init__(self, seq_num, embedding_dim):
        super().__init__()
        self.pos_embedding = nn.Embedding(num_embeddings=seq_num, embedding_dim=embedding_dim)

    def forward(self, x):
        batch, seq_num, embedding_dim = x.shape
        idx_tensor = torch.arange(seq_num).unsqueeze(0).repeat(batch, 1)
        pos_encoding = self.pos_embedding(idx_tensor)
        print('- 位置编码层输入：', x.shape)
        print('- 位置编码：', pos_encoding.shape)
        x += pos_encoding
        return x

class MainModel0(nn.Module):
    def __init__(self, seq_num, num_embeddings, embedding_dim, output_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim)
        self.pos_encoding = PosEncoding(seq_num=seq_num, embedding_dim=embedding_dim)

        self.layer = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, output_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.embedding(x)
        x = self.pos_encoding(x)
        print('- 添加位置编码后的张量：', x.shape)
        x = self.layer(x)
        return x

# 正余弦位置编码
class SinCosPosEncoding(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim  # 仅需记录维度，无需初始化参数

    def forward(self, x):
        # 获取输入形状：[batch, seq_num, embedding_dim]
        batch, seq_num, embedding_dim = x.shape

        # 1. 生成位置索引 [seq_num]
        pos = torch.arange(seq_num, device=x.device)
        # 2. 生成维度索引（步长2，对应偶数维度）[embedding_dim//2]
        dim_idx = torch.arange(0, embedding_dim, 2, device=x.device)

        # 3. 计算正弦余弦的分母项：10000^(2i/d_model)
        denom = torch.pow(10000, 2 * dim_idx / self.embedding_dim)

        # 4. 初始化位置编码张量 [seq_num, embedding_dim]
        pos_encoding = torch.zeros(seq_num, embedding_dim, device=x.device)
        # 5. 填充偶数维度（sin）、奇数维度（cos）
        pos_encoding[:, 0::2] = torch.sin(pos.unsqueeze(1) / denom)  # 偶数位：sin
        pos_encoding[:, 1::2] = torch.cos(pos.unsqueeze(1) / denom)  # 奇数位：cos

        print('- 位置编码层输入：', x.shape)
        print('- 位置编码：', pos_encoding.shape)
        # 6. 扩展batch维度 [1, seq_num, embedding_dim] → [batch, seq_num, embedding_dim]
        pos_encoding = pos_encoding.unsqueeze(0).repeat(batch, 1, 1)
        # 7. 叠加位置编码（保持输入输出形状一致）
        x += pos_encoding
        return x

class MainModel1(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, output_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim)
        self.pos_encoding = SinCosPosEncoding(embedding_dim=embedding_dim)

        self.layer = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, output_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.embedding(x)
        x = self.pos_encoding(x)
        print('- 添加位置编码后的张量：', x.shape)
        x = self.layer(x)
        return x

embedding_dim = 16
batch_size = 2
seq_num = 16

vocab_dict = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4}
vocab_size = len(vocab_dict)

batch_list = []
for _ in range(batch_size):
    # 随机生成一个长度为seq_num的句子
    sentence = random.choices(list(vocab_dict.keys()), k=seq_num)
    # 转换成索引
    sentence_index = [vocab_dict[char] for char in sentence]
    # 转换成张量
    sentence_tensor = torch.tensor(sentence_index)
    # 添加批次维度
    sentence_tensor = sentence_tensor.unsqueeze(dim=0)
    batch_list.append(sentence_tensor)

# 拼接成一整个批次
input_tensor = torch.cat(batch_list, dim=0)

model0 = MainModel0(seq_num, num_embeddings=vocab_size, embedding_dim=embedding_dim, output_dim=8, hidden_dim=16)
model1 = MainModel1(num_embeddings=vocab_size, embedding_dim=embedding_dim, output_dim=8, hidden_dim=16)

print('嵌入位置编码：')
output_tensor = model0(input_tensor)
print(f'输入张量：{input_tensor.shape}')
print(f'输出张量：{output_tensor.shape}')

print('\n正余弦位置编码：')
output_tensor = model1(input_tensor)
print(f'输入张量：{input_tensor.shape}')
print(f'输出张量：{output_tensor.shape}')