import torch
import torch.nn as nn

batch_size = 8
seq_num = 4
emb_dim = 16
nhead = 8
num_layers = 2

decoder_layer = nn.TransformerDecoderLayer(d_model=emb_dim, nhead=nhead, batch_first=True)
decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

memory = torch.ones(batch_size, 10, emb_dim)
tgt = torch.ones(batch_size, 1, emb_dim)

print('输入张量：', memory.shape)
# 续写5个内容
for i in range(5):
    output_tensor = decoder(tgt, memory)
    # 只取最后一个时间步的输出 [8, n, 16] → [8, 16]
    last_output = output_tensor[:, -1, :]
    # 将序列维度补充回去，不然cat方法拼接不了
    last_output = last_output.unsqueeze(dim=1)
    print(f'第{i}次的输出：', last_output.shape)
    # 拼接回原有的数据作为新的 tgt
    tgt = torch.cat([tgt, last_output], dim=1)

    print(f'第{i}次的tgt张量：', tgt.shape)