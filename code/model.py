import torch
import numpy as np
import torch.nn as nn

# 超参数定义 / Hyperparameters
d_model = 512   # 词向量维度 / Dimension of word embeddings
d_ff = 2048    # FFN 隐藏层大小 / Feed-forward network hidden size
d_k = d_v = 64  # Query/Key/Value 维度 / Q, K, V dimension
n_layers = 6    # Decoder 层数 / Number of decoder layers
n_heads = 8     # 多头注意力的头数 / Number of attention heads
tgt_vocab_size = 21128  # 目标词汇表大小 / Target vocabulary size


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # 生成位置编码矩阵 / Generate positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # 位置索引 / Position indices
        div_term = pos / torch.pow(10000.0, torch.arange(0, d_model, 2).float() / d_model)
        pe[:, 0::2] = torch.sin(div_term)  # 偶数索引位置使用 sin / Apply sin to even indices
        pe[:, 1::2] = torch.cos(div_term)  # 奇数索引位置使用 cos / Apply cos to odd indices
        
        pe = pe.unsqueeze(0)  # [1, max_len, d_model] 以适配 batch 维度 / Add batch dimension
        self.register_buffer('pe', pe)  # 注册为缓冲区变量，不被训练 / Register as buffer (non-trainable)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)  # 应用 dropout / Apply dropout


# 获取自回归掩码 / Generate auto-regressive mask
def get_attn_subsequence_mask(seq):
    """ 生成自回归掩码 / Generate auto-regressive mask """
    batch_size, seq_len = seq.size()
    
    # 生成上三角矩阵（防止关注未来位置） / Generate upper triangular mask
    subsequence_mask = torch.triu(torch.ones((seq_len, seq_len), dtype=torch.bool), diagonal=1)
    
    # 扩展 mask 以适配多头注意力 / Expand mask for multi-head attention
    return subsequence_mask.unsqueeze(0).unsqueeze(1).expand(batch_size, n_heads, seq_len, seq_len)


class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_model, d_model)  # 查询矩阵 / Query matrix
        self.W_K = nn.Linear(d_model, d_model)  # 键矩阵 / Key matrix
        self.W_V = nn.Linear(d_model, d_model)  # 值矩阵 / Value matrix
        self.concat = nn.Linear(d_model, d_model)  # 线性层拼接结果 / Linear transformation
    
    def forward(self, Q, K, V, attn_mask):
        batch_size = Q.size(0)
        
        # 线性变换并拆分成多头 / Linear projection and split into multiple heads
        Q = self.W_Q(Q).view(batch_size, -1, n_heads, d_k).transpose(1, 2)
        K = self.W_K(K).view(batch_size, -1, n_heads, d_k).transpose(1, 2)
        V = self.W_V(V).view(batch_size, -1, n_heads, d_v).transpose(1, 2)
        
        # 计算缩放点积注意力 / Compute scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)
        scores.masked_fill_(attn_mask, -1e9)  # 应用掩码 / Apply mask
        attn = nn.Softmax(dim=-1)(scores)  # 归一化权重 / Normalize weights
        
        context = torch.matmul(attn, V)  # 加权求和 / Weighted sum
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, d_model)  # 拼接多头结果 / Concatenate heads
        return self.concat(context)  # 通过线性变换 / Apply linear transformation


class PositionwiseFeedForward(nn.Module):
    def __init__(self):
        super(PositionwiseFeedForward, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),  # 激活函数 / Activation function
            nn.Linear(d_ff, d_model)
        )
    
    def forward(self, inputs):
        return self.fc(inputs)  # 应用前馈网络 / Apply feed-forward network


class DecoderLayer(nn.Module):
    def __init__(self):
        super(DecoderLayer, self).__init__()
        self.dec_self_attn = MultiHeadAttention()  # 自注意力 / Self-attention
        self.pos_ffn = PositionwiseFeedForward()  # 前馈神经网络 / Feed-forward network
    
    def forward(self, dec_inputs, dec_self_attn_mask):
        dec_outputs = self.dec_self_attn(dec_inputs, dec_inputs, dec_inputs, dec_self_attn_mask)
        dec_outputs = self.pos_ffn(dec_outputs)
        return dec_outputs  # 返回解码结果 / Return decoded outputs


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.tgt_emb = nn.Embedding(tgt_vocab_size, d_model)  # 目标词嵌入 / Target word embedding
        self.pos_emb = PositionalEncoding(d_model)  # 位置编码 / Positional encoding
        self.layers = nn.ModuleList([DecoderLayer() for _ in range(n_layers)])  # 多层解码器 / Multi-layer decoder
    
    def forward(self, dec_inputs):
        dec_outputs = self.tgt_emb(dec_inputs)
        dec_outputs = self.pos_emb(dec_outputs)
        
        dec_self_attn_mask = get_attn_subsequence_mask(dec_inputs).cuda()  # 计算掩码 / Compute mask
        
        for layer in self.layers:
            dec_outputs = layer(dec_outputs, dec_self_attn_mask)
        
        return dec_outputs  # 返回解码结果 / Return decoded outputs


class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        self.decoder = Decoder().cuda()  # 初始化解码器 / Initialize decoder
        self.projection = nn.Linear(d_model, tgt_vocab_size).cuda()  # 词汇投影层 / Vocabulary projection layer
    
    def forward(self, dec_inputs):
        dec_outputs = self.decoder(dec_inputs)
        dec_logits = self.projection(dec_outputs)  # [batch_size, tgt_len, tgt_vocab_size] 计算输出概率 / Compute output probabilities
        return dec_logits.view(-1, dec_logits.size(-1))  # 调整形状以适配交叉熵损失 / Reshape for cross-entropy loss
