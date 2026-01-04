import torch
import torch.nn as nn
import random
import thulac
from sklearn.model_selection import train_test_split

class Model(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers, output_size):
        super(Model, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])  # 取最后一个时间步的输出

        return out


thu1 = thulac.thulac()

food_list = ['苹果', '饼干', '汉堡', '披萨', '饺子', '面条', '香菇', '香菜', '大蒜', '馄饨', '面包', '煎饼', '面皮',
             '炒饭', '炒面', '蛋糕', '榴莲', '菠萝', '香蕉']
play_list = ['足球', '篮球', '羽毛球', '网球', '棒球', '游泳', '橄榄球', '跳绳', '跑步', '滑冰', '滑雪', '曲棍球',
             '冰球', '射箭', '台球', '马拉松']

food_like = [
    "我喜欢吃……，它很甜很好吃。",
    "……很香，我喜欢吃。",
    '我觉得……很好吃。',
    '我喜欢吃……，每次吃都会觉得美味。',
    '我非常喜欢……，它非常美味。',
    ' 是我最喜欢的食物。',
    '我特别喜欢吃……。',
    '我抵抗不了……的诱惑。',
    '……是我最喜欢的食物。',
    '我真的特别喜欢吃……。',
    '你不觉得……非常好吃吗？',
    '我觉得……是世界上最好吃的。',
    '我超爱……，吃一口就幸福感爆棚。',
    '我对……情有独钟，那滋味太赞了。',
    '每次品尝……，都感觉味蕾在欢呼。',
    '我痴迷于……，它的美味无可抵挡。',
    '一提到……我就流口水，真的太喜欢了。',
    '……是我的本命食物，怎么吃都不腻。',
    '我热衷于……，它是我生活中的小确幸美食。',
    '……能瞬间点亮我的心情，太爱这一口了。',
    '我简直为……疯狂，美味到犯规。',
    '……给我带来超多快乐，必须喜欢啊。',
    '我被……的美味折服，时不时就想来点儿。',
    '只要有……，这顿饭就超满足。',
]

food_unlike = [
    '我讨厌……，我受不了这个味道。',
    '我不喜欢……，它的味道让我恶心。',
    '……那么难吃。',
    '我吃不下……。',
    '……太难吃了，我吃不下去。',
    '我受不了……的味道。',
    '不要……，它太难吃了。',
    '……太难闻了，我不要吃。',
    '我太讨厌……的味道了。',
    '我极其厌恶……，闻到味儿就难受。',
    '……的味道让我作呕，实在难以下咽。',
    '我一看到……就没胃口，真心不喜欢。',
    '……真的很倒胃口，我绝对不会碰。',
    '我嫌弃……，它的口感太差劲了。',
    '……对我来说就是噩梦，味道太恐怖。',
    '别给我……，那股味儿我受不了。',
    '我避开……还来不及呢，太难吃了。',
    '……让我敬而远之，味道实在不敢恭维。',
    '我和……绝缘，那味道我无法忍受。',
    '一想到……的味道，我就头皮发麻。',
    '……是我的“黑名单”食物，绝不吃它。',
    '我对……毫无好感，味道太糟糕。',
    '……真不是我的菜，难吃程度五颗星。',
    '我见到……就想绕道走，太难闻了。',
    '我觉得……难闻。'
]

play_like = [
    '我喜欢……，他让我觉得放松。',
    '我很喜欢……，尽管我不擅长它。',
    '我擅长……，所以……是我最喜欢的运动。',
    '……非常有趣，所以我每天都去。',
    '我每天都去……来放松自己。',
    '……让我觉得很有动力去运动。',
    '……是我最喜欢的运动。',
    '我超享受……，玩起来特别解压。',
    '……总能让我忘却烦恼，超爱这项运动。',
    '我热衷于……，每次参与都活力满满。',
    '……是我的快乐源泉，一玩就停不下来。',
    '我超迷……，它让我变得更有活力。',
    '只要有空，我就去……，太好玩了。',
    '……给我带来无限乐趣，必须列为最爱。',
    '我对……上瘾，感觉越玩越带劲。',
    '我超爱投身于……，那种畅快难以言表。',
    '……让我找到了激情，玩不够啊。',
    '每次……都让我热血沸腾，超喜欢。',
    '我钟情于……，它让生活变得更精彩。',
    '……能让我尽情释放能量，超赞的运动。',
    '我一玩……就兴奋，它是我的心头好运动。'
]

play_unlike = [
    '我不擅长……，它让我很累。',
    '我不喜欢……，我觉得它很无聊。',
    '……真的很没意思。',
    '……太难了，不适合我。',
    '我认为……很无聊，我不喜欢。',
    '我不喜欢……。',
    '……实在是有些无聊了。',
    '我一玩……就犯困，实在提不起兴趣。',
    '我觉得……枯燥乏味，完全不想尝试。',
    '……对我来说就是折磨，毫无乐趣可言。',
    '我尝试过……，但真心觉得无聊透顶。',
    '我受不了……的单调，玩几次就放弃了。',
    '……让我感到无趣至极，不会再碰。',
    '我对……无感，找不到一点好玩的地方。',
    '……太沉闷了，我宁愿闲着也不玩。',
    '我不理解……的乐趣所在，就是不喜欢。',
    '我避开……，它无法给我带来任何愉悦。',
    '一想到……，我就觉得没意思，不想动。',
    '……是我最不想参与的，太没劲了。',
    '……实在引不起我的兴致，太乏味了。',
    '我和……气场不合，玩起来别扭'
    '我对……无感。',
    "我对……无感，它很枯燥。",
]

X_data = []
# Y表示喜欢和不喜欢的二分类
Y_data = []
for like_food in food_like:
    for food in food_list:
        food = like_food.replace('……', food)
        result = [i[0] for i in thu1.cut(food)]
        X_data.append(result)
        Y_data.append(0)
for like_play in play_like:
    for play in play_list:
        play = like_play.replace('……', play)
        result = [i[0] for i in thu1.cut(play)]
        X_data.append(result)
        Y_data.append(0)

for unlike_food in food_unlike:
    for food in food_list:
        food = unlike_food.replace('……', food)
        result = [i[0] for i in thu1.cut(food)]
        X_data.append(result)
        Y_data.append(1)

for unlike_play in play_unlike:
    for play in play_list:
        play = unlike_play.replace('……', play)
        result = [i[0] for i in thu1.cut(play)]
        X_data.append(result)
        Y_data.append(1)

# 创建词汇表
vocab = set()
for sentence in X_data:
    vocab.update(sentence)
vocab = sorted(list(vocab))
word_to_idx = {word: idx for idx, word in enumerate(vocab)}

max_len = max(len(sentence) for sentence in X_data)
print("最大长度", max_len)
# 将文本数据转换为索引表示的张量
X_data_idx = []
for sentence in X_data:
    sentence_idx = [word_to_idx[word] if word in word_to_idx else len(word_to_idx) for word in sentence]
    if len(sentence_idx) < max_len:
        sentence_idx += [len(word_to_idx)] * (max_len - len(sentence_idx))
    X_data_idx.append(sentence_idx)

X_data = torch.tensor(X_data_idx, dtype=torch.long)
Y_data = torch.tensor(Y_data, dtype=torch.long)

# 训练数据
img_size = 64
batch_size = 128
train_x, val_x, train_y, val_y = train_test_split(X_data, Y_data, test_size=0.1)


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

vocab_size = len(word_to_idx) + 1  # 词汇表大小，加1是为了处理未登录词的索引
embedding_dim = 64
hidden_size = 64
num_layers = 3
output_size = 2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Model(vocab_size, embedding_dim, hidden_size, num_layers, output_size).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

epochs = 80
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
    val_loss = criterion(output, val_y)

    # 更改验证逻辑为适合分类任务的准确率和召回率
    if (epoch + 1) % 20 == 0:
        print(f'[epoch {epoch+1}]loss:', loss.item())
        print(f'\t val loss:', val_loss.item())

test_text_list = [
    "我喜欢吃苹果，它很甜很好吃。",
    "香菜太难闻了，我不喜欢吃。",
    "香蕉很香，我喜欢吃。",
    "我觉得榴莲太难闻了，我不喜欢吃。",
    "我特别喜欢棒球，它太好玩了。",
    "我对羽毛球无感，它很枯燥。",
]
model.eval()
for test_text in test_text_list:
    test_result = [i[0] for i in thu1.cut(test_text)]
    test_data_idx = [word_to_idx[word] if word in word_to_idx else len(word_to_idx) for word in test_result]
    if len(test_data_idx) < max_len:
        test_data_idx += [len(word_to_idx)] * (max_len - len(test_data_idx))
    test_data = torch.tensor(test_data_idx, dtype=torch.long)

    input_tensor = torch.stack([test_data]).to(device)
    output = model(input_tensor)
    _, predicted = torch.max(output.data, 1)
    print(f"{test_text}: {'喜欢' if predicted.item() == 0 else '不喜欢'}")