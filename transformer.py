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
    
