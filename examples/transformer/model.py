from dataclasses import dataclass
import torch
import torch.nn as nn
import math

@dataclass
class Config:
    num_heads: int = 8
    hidden_size: int = 512
    num_layers: int = 6
    dropout: float = 0.1
    max_len: int = 100

class PositionalEncoding(nn.Module):
    def __init__(self, hidden_size, dropout, max_len):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, hidden_size)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_size, 2).float() * (-math.log(10000.0) / hidden_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.query_linear = nn.Linear(hidden_size, hidden_size)
        self.key_linear = nn.Linear(hidden_size, hidden_size)
        self.value_linear = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, query, key, value):
        batch_size = query.size(0)
        seq_len = query.size(1)
        query = self.query_linear(query).view(batch_size, -1, self.num_heads, -1).transpose(1, 2)
        key = self.key_linear(key).view(batch_size, -1, self.num_heads, -1).transpose(1, 2)
        value = self.value_linear(value).view(batch_size, -1, self.num_heads, -1).transpose(1, 2)

        attention_scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(query.size(-1))
        attention_weights = torch.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        output = torch.matmul(attention_weights, value).transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return output

class TransformerLayer(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super(TransformerLayer, self).__init__()
        self.self_attention = MultiHeadAttention(hidden_size, num_heads)
        self.feed_forward = nn.Linear(hidden_size, hidden_size)
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.layer_norm2 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        attention_output = self.self_attention(x, x, x)
        attention_output = self.layer_norm1(x + attention_output)
        feed_forward_output = self.feed_forward(attention_output)
        feed_forward_output = self.dropout(feed_forward_output)
        output = self.layer_norm2(attention_output + feed_forward_output)
        return output

class Transformer(nn.Module):
    def __init__(self, config):
        super(Transformer, self).__init__()
        self.positional_encoding = PositionalEncoding(config.hidden_size, config.dropout, config.max_len)
        self.embedding = nn.Embedding(config.hidden_size, config.hidden_size)
        self.layers = nn.ModuleList([TransformerLayer(config.hidden_size, config.num_heads) for _ in range(config.num_layers)])
        self.final_layer = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, x):
        x = self.embedding(x)
        x = self.positional_encoding(x)
        for layer in self.layers:
            x = layer(x)
        x = self.final_layer(x)
        return x

if __name__ == "__main__":
    config = Config()
    model = Transformer(config)
    input_ids = torch.randint(0, config.hidden_size, (1, 10))
    output = model(input_ids)
    print(output.shape)