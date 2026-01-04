import torch
import torch.nn as nn
import thulac
import random

class Model(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers, num_tags):
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers * 2  # 双向LSTM，层数翻倍
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.RNN(embedding_dim, hidden_size, num_layers, batch_first=True, bidirectional=True)

        self.fc = nn.Linear(hidden_size * 2, num_tags)

    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.lstm(x)

        out = self.fc(out)
        return out


thu1 = thulac.thulac()

max_len = 30
space_mark = '<S>'
NoName_mark = '/'
MidName_mark = '-'

name_list = ['杭电信工', '杭州电子科技大学信息工程学院', '杭州电子科技大学', '杭州电子科技大学（青山湖校区）', '电子科技大学', '北京大学','北大', '清华', '清华大学', '浙江中医药大学', '国防科技大学', '合肥工业大学', '杭州师范大学', '浙江工商大学', '浙江农林大学','安徽医科大学', '安徽农业大学', '合肥理工大学', '合工大', '杭电']

sentence_list = [
    '我是Name的一名学生',
    '我目前就读于Name',
    'Name是我所在的学府',
    '我正在Name接受教育',
    'Name是我选择的大学',
    '我是Name的学子',
    '我在Name大学深造',
    'Name是我求学的地方',
    '我于Name大学就读',
    '我正在Name攻读学位',
    'Name是我梦想中的大学',
    '我很高兴能在Name学习',
    'Name是我学术旅程的起点',
    '我选择了Name作为我的大学',
    '在Name，我追求知识的真谛',
    'Name为我提供了广阔的学习平台',
    '我是Name大学的一份子',
    'Name见证了我的成长与学习',
    '我将在Name完成我的学业',
    'Name是我人生中的重要一站',
    '我在Name大学就读',
    'Name是我目前就读的高等学府',
    '我正在Name接受高等教育',
    'Name是我选择深造的大学',
    '我于Name大学开始学习之旅',
    '在Name，我开始了我的大学生活',
    'Name是我梦想中的学府，我现在正在那里学习',
    '我目前的学习地点是Name大学',
    '我正在Name攻读我的学位',
    'Name是我求学之路的下一站',
    '我很高兴能在Name这样的名校学习',
    '作为Name的学生，我感到非常自豪',
    '在Name的学习经历对我来说非常宝贵',
    '我正在Name努力提升自己的学识和能力',
    'Name拥有一支高水平的师资队伍',
    '师资力量是Name发展的坚实后盾',
    'Name积极与国内外高校开展合作交流',
]

X_dict = {}
for sentence in sentence_list:
    for name in name_list:
        sentence0 = sentence.replace('Name', name)
        result = [i[0] for i in thu1.cut(sentence0)]
        X_dict[tuple(result)] = name

X_data = []
Y_data = []
test = False
for words, name in X_dict.items():
    sentence_list = []
    for word in words:
        if word in name:
            sentence_list.append(MidName_mark)
        else:
            sentence_list.append(NoName_mark)

    words = list(words)

    while len(words) < max_len:
        words.append(space_mark)
        sentence_list.append(NoName_mark)

    X_data.append(list(words))
    Y_data.append(sentence_list)

# print(X_data)
# print(Y_data)

# 构建词汇表，为每个词分配数字索引
all_words = []
for x in X_data:
    all_words.extend(x)
for y in Y_data:
    all_words.extend(y)
vocab = sorted(set(all_words))

word_to_idx = {word: idx for idx, word in enumerate(vocab)}
idx_to_word = {idx: word for word, idx in word_to_idx.items()}  # 方便后续将索引转换回词

# 将数据中的词转换为数字索引表示
X_data_idx = [[word_to_idx[word] for word in x] for x in X_data]
Y_data_idx = [[word_to_idx[word] for word in y] for y in Y_data]

train_x = torch.tensor(X_data_idx)
train_y = torch.tensor(Y_data_idx)

batch_size = 128
# 划分批次
def split_batch(data, batch_size):
    # 核心操作：沿第一个维度（dim=0）分割，保留后续所有维度
    split_tensors = torch.split(data, batch_size, dim=0)
    # 转为列表返回（torch.split返回tuple，列表更易操作）
    return list(split_tensors)

train_x_batch = split_batch(train_x, batch_size)
train_y_batch = split_batch(train_y, batch_size)

print('输入数据形状:', train_x.shape)
print('输入批次数量:', len(train_x_batch), '\t批次形状:', train_x_batch[0].shape)
print('标签数据形状:', train_y.shape)
print('输入批次数量:', len(train_y_batch), '\t批次形状:', train_y_batch[0].shape)

vocab_size = len(vocab)
embedding_dim = 100
hidden_size = 256
num_layers = 2
num_tags = vocab_size  # 根据标签种类确定类别数量
learning_rate = 0.005

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Model(vocab_size, embedding_dim, hidden_size, num_layers, num_tags).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 20
# 训练模型
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for i in range(len(train_x_batch)):
        batch_x = train_x_batch[i].to(device)
        batch_y = train_y_batch[i].to(device)
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs.view(-1, num_tags), batch_y.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    if (epoch + 1) % 4 == 0:
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(train_x_batch)}')


# 简单的预测函数（这里只是示例，实际应用中可进一步完善）
def predict(model, sentence):
    result = [i[0] for i in thu1.cut(sentence)]
    words_idx = [word_to_idx.get(word, word_to_idx[space_mark]) if word in word_to_idx else word_to_idx.get(space_mark, word_to_idx) for word in result]
    words_idx = torch.tensor([words_idx]).long().to(device)  # 转换为合适的张量形式
    with torch.no_grad():
        output = model(words_idx)
        predicted_tags = torch.argmax(output, dim=2)[0].tolist()
        predicted_tags = [idx_to_word[tag] for tag in predicted_tags]
    return result, predicted_tags


# 简单的测试示例（重新加载模型后进行预测）
model.eval()
with torch.no_grad():
    # 示例预测
    for x in random.choices(list(X_dict.keys()), k=5):
        # 我在杭州电子科技大学学习
        test_sentence = ''.join(x)
        print('>>> 测试语句：', test_sentence)
        words, predicted_result = predict(model, test_sentence)
        sentence = ''
        for i in range(len(predicted_result)):
            mark = predicted_result[i]
            if mark == '-':
                sentence += f'\033[91m{words[i]}\033[0m'
            else:
                sentence += words[i]

        print('命名实体识别结果：', sentence)