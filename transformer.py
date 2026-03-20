import torch
import math
import torch.nn as nn

class InputEmbeddings(nn.Module):
    """
    Weight Matrix (lookup table): shape (vocab_size, d_model), e.g. (10000, 512)

        d_model = 512
        ←────────────────→
    ┌──────────────────────┐
    │ 0.02  -0.01  0.03 ...│ ← row 0 (token ID 0)
    │ 0.01   0.04  -0.02...│ ← row 1 (token ID 1)
    │ -0.03  0.02  0.01 ...│ ← row 2 (token ID 2)
    │         ...          │
    │ 0.00  -0.02  0.04 ...│ ← row 9999
    └──────────────────────┘

    Each row is a vector representation of a token in the vocab
    
    e.g. Token ID 42 → Look up row 42 → Get a 512-dim vector
    
    Input tensor shape: (batch_size, seq_len) e.g. (2, 4)
    ┌─────────────────────────────┐
    │  [1,  2,  3,  4]            │  ← sequence 1 (4 tokens)
    │  [5,  6,  7,  8]            │  ← sequence 2 (4 tokens)
    └─────────────────────────────┘
            ↓
        Embedding Lookup
        (each token ID → 512-dim vector)
            ↓
    Output shape: (2, 4, 512)
    ┌─────────────────────────────────────────────────────┐
    │  [vec₁, vec₂, vec₃, vec₄]  each vec is 512 floats  │
    │  [vec₅, vec₆, vec₇, vec₈]                          │
    └─────────────────────────────────────────────────────┘

    Each token is now represented by a 512 vector.
    """
    def __init__(self, vocab_size: int, d_model: int) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model) # scaled by sqrt(dimension of model)
    
class PositionalEncoding(nn.Module):
    """
    Based on:
    PE_(pos, 2i)    = sin( pos / 10_000^(2i/d_model) )
    PE_(pos, 2i+1)  = cos( pos / 10_000^(2i/d_model) )

    (1) Initilisation

    Shape: (max_seq_length, d_model)
    Example: (5000, 512)

        d_model = 512
        ←──────────────→
    ┌──────────────────┐
    │  0  0  0  0  0   │ ← position 0
    │  0  0  0  0  0   │ ← position 1
    │       ...        │
    │  0  0  0  0  0   │ ← position 4999
    └──────────────────┘

    (2) Create positions

    position shape: (max_seq_length, 1)

    [[0],
     [1],
     [2],
     ...
     [max_seq_length-1]]

    (3) Create division term

    Convert x^y via power functions to native exp and log functions for efficiency
    and stability

    We use 2i or arange(0, d_model, 2) because each div_term is used by both sin and cos

    Equation: 1 / 10000^(2i/d_model)

    Using x = exp(log(x)) : 10000^(2i/d_model)  = exp(log( 10000^(2i/d_model) ))
                                                = exp(log(10000) * 2i/d_model)
                                                = exp(2i * log(10000) / d_model)

    So: 1 / 10000^(2i/d_model)  = exp(-2i * log(10000) / d_model)
                                = exp(2i * (-log(10000) / d_model))

    div_term shape: (d_model/2,) = (256,)

    Values: [1.0, 0.96, 0.93, 0.90, ..., very small] -> decreasing

    These are the denominators: 1/10000^(2i/d_model)
    for i = 0, 1, 2, ..., d_model/2 - 1

    (4) Broadcast to fill pe matrix

    position * div_term:

        div_term values (decreasing)
        ←───────────────────────────→
    ┌─────────────────────────────────┐
    │ 0x1.0   0x0.96   0x0.93   ...   │ ← pos=0: all zeros
    │ 1x1.0   1x0.96   1x0.93   ...   │ ← pos=1: [1.0, 0.96, 0.93, ...]
    │ 2x1.0   2x0.96   2x0.93   ...   │ ← pos=2: [2.0, 1.92, 1.86, ...]
    │             ...                 │
    └─────────────────────────────────┘

    Final pe matrix:
    column: 0    1    2    3    4    5   ...
                ←────────────────────────────→
        ┌──────────────────────────────────────┐
    pos│ sin  cos  sin  cos  sin  cos  ...     │
     0 │ 0.0  1.0  0.0  1.0  0.0  1.0  ...     │
     1 │ 0.84 0.67 0.82 0.69 0.80 0.71  ...    │
     2 │ 0.91 0.42 0.89 0.46 0.86 0.51  ...    │
     . │            ...                        │
        └──────────────────────────────────────┘

    Word Embedding (learned)     Positional Encoding (fixed)
            ↓                            ↓
    [0.1, 0.3, 0.2, ...]      +  [0.0, 1.0, 0.0, ...]
            ↓                            ↓
            └──────────→ ADD ←──────────┘
                        ↓
                Combined Representation
                        ↓
                    Self-Attention
    """
    def __init__(self, d_model: int, max_seq_length: int) -> None:
        super().__init__()
        
        # (1) init to empty first
        pe = torch.zeros(max_seq_length, d_model)

        # (2) create position index

        # [0, 1, 2, ...] -> unsqueeze -> [[0], [1], [2], ...]
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)

        # (3) denominator
        # torch.arange(0, d_model, 2)    # [0, 2, 4, 6, ..., 510] -> 2i values
        #                                # shape: (d_model/2,) = (256,)

        # math.log(10_000.0) / d_model   # log(10000) / 512 ≈ 0.0176

        # -(math.log(10_000.0) / d_model)  # negated: -0.0176

        # # Multiply and exponentiate:
        # torch.arange(0, d_model, 2) * -(math.log(10_000.0) / d_model)
        # # Results in: [0, -0.035, -0.07, -0.105, ...]

        # torch.exp(...)  # [1.0, 0.965, 0.932, 0.90, ...]
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * -(math.log(10_000.0) / d_model))

        # (4) broadcast to fill

        # position:  (max_seq_len, 1)     e.g., (5000, 1)
        # div_term:  (d_model/2,)         e.g., (256,)

        # position * div_term broadcasts to:
        #            (max_seq_len, d_model/2)  e.g., (5000, 256)
        #         pe[:, 0::2] = torch.sin(position * div_term)
        #         pe[:, 1::2] = torch.cos(position * div_term)

        pe[:, 0::2] = torch.sin(position * div_term) # even columns
        pe[:, 1::2] = torch.sin(position * div_term) # odd columns

        # fixed params, no grad
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        # pe shape: (1, max_seq_len, d_model)

        # slice to match seq length -> (1, seq_len, d_model)
        # addition broadcast over batch dim - each seq gets same pos enc
        return x + self.pe[:, :x.size(1)]