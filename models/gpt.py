"""
General Pre-trained Transformer (GPT)

https://arxiv.org/abs/1810.04805
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class GPTEmbeddings(nn.Module):
    def __init__(self, vocab_size: int, max_seq_length: int, d_model: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.d_model = d_model
        self.token_embeddings = nn.Embedding(vocab_size, d_model)
        self.position_embeddings = nn.Embedding(max_seq_length, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, token_ids):
        batch_size, seq_length = token_ids.shape
        device = token_ids.device

        # create position ids
        position_ids = torch.arange(seq_length, device=device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

        # look up embeddings
        token_embed = self.token_embeddings(token_ids)
        position_embed = self.position_embeddings(position_ids)

        embeddings = self.dropout(token_embed + position_embed)

        return embeddings
    

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int) -> None:
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.num_heads = num_heads
        self.d_model = d_model
        self.head_dim = d_model // num_heads

        self.query_linear = nn.Linear(d_model, d_model, bias=False)
        self.key_linear = nn.Linear(d_model, d_model, bias=False)
        self.value_linear = nn.Linear(d_model, d_model, bias=False)

        self.output_linear = nn.Linear(d_model, d_model)

    def split_heads(self, x, batch_size):
        seq_length = x.size(1)
        x = x.reshape(batch_size, seq_length, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)
    
    def compute_attention(self, query, key, value, mask=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) / (self.head_dim ** 0.5)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        attention_weights = F.softmax(scores, dim=-1)
        return torch.matmul(attention_weights, value)
    
    def combine_heads(self, x, batch_size):
        x = x.permute(0, 2, 1, 3).contiguous()
        return x.view(batch_size, -1, self.d_model)
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # projections
        query = self.query_linear(query)
        key = self.key_linear(key)
        value = self.value_linear(value)

        # split
        query = self.split_heads(query, batch_size)
        key = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)

        # attention
        attention_output = self.compute_attention(query, key, value, mask)

        # combine
        attention_output = self.combine_heads(attention_output, batch_size)

        # output linear
        x = self.output_linear(attention_output)

        return x
    

class FeedForwardSubLayer(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def foward(self, x):
        x = self.gelu(self.fc1(x))
        x = self.fc2(self.dropout(x))
        return x


class GPTDecoderLayer(nn.Module):
    pass
