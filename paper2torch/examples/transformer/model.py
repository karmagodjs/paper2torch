import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass

@dataclass
class TransformerConfig:
    num_heads: int = 8
    num_layers: int = 6
    embedding_dim: int = 512
    input_dim: int = 37000
    max_position: int = 5000
    dropout: float = 0.1
    max_len: int = 100

class SelfAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads):
        super(SelfAttention, self).__init__()
        self.num_heads = num_heads
        self.embedding_dim = embedding_dim
        # Query, Key, and Value projection layers
        self.query_linear = nn.Linear(embedding_dim, embedding_dim)
        self.key_linear = nn.Linear(embedding_dim, embedding_dim)
        self.value_linear = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, x):
        # Split the input into query, key, and value
        query = self.query_linear(x)
        key = self.key_linear(x)
        value = self.value_linear(x)
        
        # Calculate attention weights
        attention_weights = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(self.embedding_dim)
        
        # Calculate attention output
        attention_output = torch.matmul(attention_weights, value)
        
        return attention_output

class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.self_attention = SelfAttention(embedding_dim, num_heads)
        
    def forward(self, x):
        # Split the input into multiple attention heads
        x = x.view(-1, x.size(1), self.num_heads, self.embedding_dim // self.num_heads)
        
        # Apply self-attention for each head
        attention_output = self.self_attention(x)
        
        # Combine attention outputs from all heads
        attention_output = attention_output.view(-1, x.size(1), self.embedding_dim)
        
        return attention_output

class TransformerLayer(nn.Module):
    def __init__(self, embedding_dim, num_heads):
        super(TransformerLayer, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.multi_head_attention = MultiHeadAttention(embedding_dim, num_heads)
        self.feed_forward = nn.Linear(embedding_dim, embedding_dim)
        
    def forward(self, x):
        # Apply multi-head attention
        attention_output = self.multi_head_attention(x)
        
        # Apply feed-forward network
        feed_forward_output = F.relu(self.feed_forward(attention_output))
        
        return feed_forward_output

class Transformer(nn.Module):
    def __init__(self, config):
        super(Transformer, self).__init__()
        self.config = config
        self.embedding = nn.Embedding(config.input_dim, config.embedding_dim)
        self.position_encoding = nn.Embedding(config.max_position, config.embedding_dim)
        self.transformer_layers = nn.ModuleList([TransformerLayer(config.embedding_dim, config.num_heads) for _ in range(config.num_layers)])
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, x):
        # Apply embedding and position encoding
        embedded_x = self.embedding(x)
        position_encoding = self.position_encoding(torch.arange(x.size(1), device=x.device)).unsqueeze(0)
        embedded_x += position_encoding
        
        # Apply dropout to the embedded input
        embedded_x = self.dropout(embedded_x)
        
        # Apply transformer layers
        for layer in self.transformer_layers:
            embedded_x = layer(embedded_x)
            
        return embedded_x

if __name__ == "__main__":
    import math
    config = TransformerConfig()
    model = Transformer(config)
    input_ids = torch.randint(0, config.input_dim, size=(1, config.max_len))
    output = model(input_ids)
    print(output.shape)