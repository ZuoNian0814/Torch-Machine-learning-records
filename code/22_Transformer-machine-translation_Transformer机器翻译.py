import random
import thulac
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

test_size = 5
thu1 = thulac.thulac(seg_only=True)

def split_Chinese(text, strat_mark, end_mark):
    global output_index_data, output_list
    split_list = []
    num = len(output_index_data)
    for i in thu1.cut(text):
        split_list.append(i[0])
        if i[0] not in output_index_data:
            output_index_data[i[0]] = num
            num += 1
    return [strat_mark] + split_list + [end_mark]

def split_English(text):
    global input_index_data
    text = text.lower()
    result = []
    start = 0
    num = len(input_index_data)
    for index in range(len(text)):
        if text[index].isspace() or not text[index].isalnum():
            if start!= index:
                result.append(text[start:index])
            result.append(text[index])

            start = index + 1
    if start < len(text):
        result.append(text[start:])

    while ' ' in result:
        result.remove(' ')

    for word in result:
        if word not in input_index_data:
            input_index_data[word] = num
            num += 1
    return result


data = [
    ['This is apple.', '这个是苹果。'],
    ['This apple is sweet.', '这个苹果很甜。'],
    ["I like apples.", "我喜欢苹果。"],
    ["I do not like apples.", "我不喜欢苹果。"],
    ['Do you like apples?', "你喜欢苹果吗？"],
    ['Do you like this sweet apple?', '你喜欢这个甜的苹果吗？'],
    ['Do you like this apple?', '你喜欢这个苹果吗？'],
    ['I like this apple.', '我喜欢这个苹果。'],
    ['This apple is very sweet.', '这个苹果非常甜。'],
    ['I very like this apple.', '我非常喜欢这个苹果。'],
    ['This red apple looks delicious.', '这个红苹果看起来很美味。'],
    ['Apples are rich in vitamins.', '苹果富含维生素。'],
    ['I picked some apples in the orchard.', '我在果园里摘了一些苹果。'],
    ['The apples in this basket are fresh.', '这个篮子里的苹果很新鲜。'],
    ['Would you like to have an apple?', '你想吃个苹果吗？'],
    ['She made an apple pie for dessert.', '她做了一个苹果派当甜点。'],
    ['These apples are a bit sour.', '这些苹果有点酸。'],
    ['The apple fell from the tree.', '苹果从树上掉了下来。'],
    ['This sweet apple fell from the tree.', '这个甜苹果从树上掉了下来。'],
    ['I like apples that are very sweet.', '我喜欢非常甜的苹果。'],
    ['Do you like this apple that looks sweet?', '你喜欢这个看起来很甜的苹果吗？'],
    ['The apples in this orchard are very fresh and rich in vitamins.', '这个果园里的苹果非常新鲜且富含维生素。'],
    ['She made an apple pie with these fresh apples for dessert.', '她用这些新鲜的苹果做了一个苹果派当甜点。'],
    ['I do not like apples that are a bit sour.', '我不喜欢有点酸的苹果。'],
    ['Would you like to have a sweet apple from this basket?', '你想吃这个篮子里的一个甜苹果吗？'],
    ['I picked some apples in the orchard.', '我在果园里摘了一些苹果。'],
    ['Do you like apple pie?', '你喜欢苹果派吗？'],
    ['I like this apple pie.', '我喜欢这个苹果派。'],
    ['I dislike this apple.', '我不喜欢这个苹果。'],
    ['I dislike apples.', '我不喜欢苹果。'],
    ['I dislike apple pie.', '我不喜欢苹果派。'],
    ['I enjoy this apple.', '我享受这个苹果。'],
    ['I enjoy apples.', '我享受苹果。'],
    ['I enjoy apple pie.', '我享受苹果派。'],
    ['The apples in this orchard are very sweet.', '这个果园里的苹果非常甜。'],
    ['The apples in this orchard are very fresh.', '这个果园里的苹果非常新鲜。'],
    ['This apples rich in vitamins.', '这个苹果富含维生素。'],
    ['This apple is very fresh.', '这个苹果非常新鲜。'],
]

foods_list = [
    ['banana', 'bananas', '香蕉'],
    ['pear', 'pears', '桃子'],
    ['peach', 'peaches', '猕猴桃'],
    ['watermelon', 'watermelons', '西瓜'],
    ['pineapple', 'pineapples', '菠萝'],
    ['coconut', 'coconuts', '椰子'],
    ['cherry', 'cherries', '樱桃'],
    ['grape', 'grapes', '葡萄'],
    ['orange', 'oranges', '橙子'],
    ['lemon', 'lemons', '柠檬'],
    ['mango', 'mangoes', '芒果'],
    ['kiwi', 'kiwis', '奇异果'],
    ['strawberry', 'strawberries', '草莓'],
    ['blueberry', 'blueberries', '蓝莓'],
    ['raspberry', 'raspberries', '覆盆子'],
    ['blackberry', 'blackberries', '黑莓'],
    ['pomegranate', 'pomegranates', '石榴'],
    ['fig', 'figs', '无花果'],
    ['apricot', 'apricots', '杏子'],
    ['plum', 'plums', '李子'],
    ['date', 'dates', '枣'],
    ['persimmon', 'persimmons', '柿子'],
    ['papaya', 'papayas', '木瓜'],
    ['guava', 'guavas', '番石榴'],
    ['lychee', 'lychees', '荔枝'],
    ['dragon fruit', 'dragon fruits', '火龙果'],
    ['passion fruit', 'passion fruits', '百香果']
]

a = data.copy()
progress = 0
for food, foods, food_c in foods_list:
    progress += 1
    for E, C in a:
        E = E.replace('apples', foods).replace('apple', food)
        C = C.replace('苹果', food_c)
        data.append([E, C])

space_mark = '</>'
strat_mark = '<start>'
end_mark = '<end>'
input_index_data = {space_mark: 0}
output_index_data = {strat_mark: 0, end_mark: 1, space_mark: 2}
output_list = []

test_src_seqs = []
test_tgt_seqs = []
for _ in range(test_size):
    rand_index = random.randint(0, len(data)-1)
    test_src_seqs.append(data[rand_index][0])
    test_tgt_seqs.append(data[rand_index][1])
    data.pop(rand_index)

# 示例数据
src_seqs = []
tgt_seqs = []
train_data = []
for E, C in data:
    e = [input_index_data[i] for i in split_English(E)]
    c = [output_index_data[i] for i in split_Chinese(C, strat_mark, end_mark)]

    src_seqs.append(e)
    tgt_seqs.append(c)

print('数据集大小：', len(src_seqs))
print('英语词汇表大小：', len(input_index_data))
print('中文词汇表大小：', len(output_index_data))
tgt_vocab = {}
for i in output_index_data.keys():
    tgt_vocab[output_index_data[i]] = i

# 填充序列到相同长度
src_max_len = max(len(seq) for seq in src_seqs)
tgt_max_len = max(len(seq) for seq in tgt_seqs)
print('英文序列最大长度：', src_max_len)
print('中文序列最大长度：', tgt_max_len)
src_padded = [seq + [0] * (src_max_len - len(seq)) for seq in src_seqs]
tgt_padded = [seq + [0] * (tgt_max_len - len(seq)) for seq in tgt_seqs]

# 转换为张量
src_tensor = torch.tensor(src_padded)
tgt_tensor = torch.tensor(tgt_padded)

batch_size = 256
# 创建数据集和数据加载器
dataset = TensorDataset(src_tensor, tgt_tensor)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

print('输入张量：', src_tensor.shape)
print('tgt张量：', tgt_tensor.shape)
print('批次大小：', len(dataloader))
for src, tgt in dataloader:
    print('\t-每批输入张量：', src.shape)
    print('\t-每批tgt张量：', tgt.shape)
    break

print('测试数据：')
for i in range(test_size):
    print(test_src_seqs[i])
    print(test_tgt_seqs[i])

# 位置编码类
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

        # 6. 扩展batch维度 [1, seq_num, embedding_dim] → [batch, seq_num, embedding_dim]
        pos_encoding = pos_encoding.unsqueeze(0).repeat(batch, 1, 1)
        # 7. 叠加位置编码（保持输入输出形状一致）
        x += pos_encoding
        return x

class TransformerModel(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout):
        super(TransformerModel, self).__init__()
        # 源语言嵌入层
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        # 目标语言嵌入层
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)

        # 位置编码
        self.positional_encoding = SinCosPosEncoding(d_model)

        # 编码器层
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # 解码器层
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        # 线性输出层，将解码器输出映射到目标语言词汇表大小
        self.output_layer = nn.Linear(d_model, tgt_vocab_size)

    def generate_square_subsequent_mask(self, sz):
        """生成自注意力掩码，防止模型看到未来信息"""
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def create_padding_mask(self, seq):
        """创建填充掩码，用于忽略填充的0元素"""
        return (seq == 0).to(seq.device)

    def forward(self, src, tgt):
        # 创建源序列和目标序列的填充掩码
        src_key_padding_mask = self.create_padding_mask(src)
        tgt_key_padding_mask = self.create_padding_mask(tgt)

        # 检查并调整掩码形状
        if src_key_padding_mask.dim() == 2 and src_key_padding_mask.size(0) != src.size(1):
            src_key_padding_mask = src_key_padding_mask.transpose(0, 1)
        if tgt_key_padding_mask.dim() == 2 and tgt_key_padding_mask.size(0) != tgt.size(1):
            tgt_key_padding_mask = tgt_key_padding_mask.transpose(0, 1)

        # 源序列嵌入和位置编码
        src_embedded = self.src_embedding(src)
        src_embedded = self.positional_encoding(src_embedded)

        # 目标序列嵌入和位置编码
        tgt_embedded = self.tgt_embedding(tgt)
        tgt_embedded = self.positional_encoding(tgt_embedded)

        # 编码器处理
        memory = self.transformer_encoder(src_embedded, src_key_padding_mask=src_key_padding_mask)

        # 解码器掩码
        tgt_mask = self.generate_square_subsequent_mask(tgt.size(0)).to(tgt.device)

        # 解码器处理
        output = self.transformer_decoder(tgt_embedded, memory, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=src_key_padding_mask)

        # 线性输出层
        output = self.output_layer(output)
        return output

# 定义超参数
d_model = 24  # 模型的特征维度
nhead = 4  # 多头注意力机制中的头数
num_encoder_layers = 2  # 编码器层数
num_decoder_layers = 2  # 解码器层数
dim_feedforward = 1800  # 前馈神经网络的中间层维度
dropout = 0.1  # Dropout 概率
src_vocab_size = len(input_index_data)  # 源语言词汇表大小
tgt_vocab_size = len(output_index_data)  # 目标语言词汇表大小
lr = 0.001
num_epochs = 150

# 检查 CUDA 是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 创建模型实例
model = TransformerModel(src_vocab_size, tgt_vocab_size, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout)
model = model.to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
# 定义学习率调度器
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.99)

for epoch in range(num_epochs):
    total_loss = 0
    for src_batch, tgt_batch in dataloader:
        progress += 1
        src_batch = src_batch.transpose(0, 1).to(device)
        tgt_batch = tgt_batch.transpose(0, 1).to(device)
        optimizer.zero_grad()
        output = model(src_batch, tgt_batch[:-1])  # 去掉最后一个时间步
        target = tgt_batch[1:]  # 去掉第一个时间步
        loss = criterion(output.reshape(-1, tgt_vocab_size), target.reshape(-1))
        loss.backward()

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        total_loss += loss.item()

    scheduler.step()
    if (epoch + 1) % 30 == 0:
        progress = 0
        print(f'Epoch[{epoch + 1}] Loss: {total_loss / len(dataloader)}')

# 测试数据
test = test_src_seqs.copy()

for i0, s in enumerate(test):
    words = split_English(s)
    test[i0] = [input_index_data[i] for i in words]

# 填充测试源序列到相同长度
test_src_max_len = max(len(seq) for seq in test)
print(test_src_max_len)
test_src_padded = [seq + [0] * (test_src_max_len - len(seq)) for seq in test]
# 转换为张量
test_src_tensor = torch.tensor(test_src_padded).transpose(0, 1).to(device)
# 测试阶段
model.eval()
with torch.no_grad():
    for i in range(len(test)):
        src = test_src_tensor[:, i].unsqueeze(1)
        # 初始化目标序列，以 <start> 开始
        tgt = torch.tensor([[0]]).to(device)
        output_sequence = []
        # 循环生成，最长20个词。
        for x in range(20):
            output = model(src, tgt)
            next_token = torch.argmax(output[-1, :, :], dim=1, keepdim=True)
            if next_token.item() == 1:  # 遇到 <end> 停止
                break
            output_sequence.append(next_token.item())
            tgt = torch.cat([tgt, next_token], dim=0)
            # print(tgt.shape)
        # 将索引序列转换为文本
        output_text = "".join([tgt_vocab.get(idx, space_mark) for idx in output_sequence])
        print(f"原语句：{test_src_seqs[i]}")
        print(f'实际翻译：{test_tgt_seqs[i]}')
        print(f"翻译结果: {output_text}\t生成序列：{output_sequence}")