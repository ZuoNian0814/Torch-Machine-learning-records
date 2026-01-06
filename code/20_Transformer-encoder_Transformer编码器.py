import torch
import torch.nn as nn

batch_size = 8
seq_num = 4
# 特征值和嵌入维度都是与d_model匹配的
feature_dim = emb_dim = 16
nhead = 8

num_layers = 2

input_tensor = torch.ones(batch_size, emb_dim)  # [seq_len, batch, features]
input_seq_tensor = torch.ones(batch_size, seq_num, emb_dim)  # [seq_len, batch, features]

encoder_layer = nn.TransformerEncoderLayer(dropout=True, d_model=emb_dim, nhead=nhead, batch_first=True)
encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

print('多层Transformer：')
output_tensor = encoder(input_tensor)
output_seq_tensor = encoder(input_seq_tensor)
print('特征值输入张量：', input_tensor.shape)
print('特征值输出张量：', output_tensor.shape)
print('序列数据输入张量：', input_seq_tensor.shape)
print('序列数据输出张量：', output_seq_tensor.shape)

print('\n直接传入Transformer层也可使用：')
output_tensor = encoder_layer(input_tensor)
output_seq_tensor = encoder_layer(input_seq_tensor)
print('特征值输入张量：', input_tensor.shape)
print('特征值输出张量：', output_tensor.shape)
print('序列数据输入张量：', input_seq_tensor.shape)
print('序列数据输出张量：', output_seq_tensor.shape)