import torch
import math
import torch.nn as nn
import torch.nn.functional as F

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

    =================================================================================
    (1) Initilisation
    =================================================================================
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

    =================================================================================
    (2) Create positions
    =================================================================================
    position shape: (max_seq_length, 1)

    [[0],
     [1],
     [2],
     ...
     [max_seq_length-1]]

    =================================================================================
    (3) Create division term
    =================================================================================
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

    =================================================================================
    (4) Broadcast to fill pe matrix
    =================================================================================
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


class MultiHeadAttention(nn.Module):
    """
    Input x: Shape (2, 4, 8)

    Batch 0:
    ┌─────────────────────────────────────────────────────────┐
    │ Token 0: [a0, a1, a2, a3, a4, a5, a6, a7]               │
    │ Token 1: [b0, b1, b2, b3, b4, b5, b6, b7]               │
    │ Token 2: [c0, c1, c2, c3, c4, c5, c6, c7]               │
    │ Token 3: [d0, d1, d2, d3, d4, d5, d6, d7]               │
    └─────────────────────────────────────────────────────────┘

    Batch 1:
    ┌─────────────────────────────────────────────────────────┐
    │ Token 0: [e0, e1, e2, e3, e4, e5, e6, e7]               │
    │ Token 1: [f0, f1, f2, f3, f4, f5, f6, f7]               │
    │ Token 2: [g0, g1, g2, g3, g4, g5, g6, g7]               │
    │ Token 3: [h0, h1, h2, h3, h4, h5, h6, h7]               │
    └─────────────────────────────────────────────────────────┘

    Each token = word embedding + positional encoding

    =================================================================================
    (1) Apply Q, K, V linear projections
    =================================================================================
                    Input x
                       │
         ┌─────────────┼─────────────┐
         ↓             ↓             ↓
    query_linear   key_linear   value_linear
         ↓             ↓             ↓
       Query         Key          Value
      (2,4,8)       (2,4,8)      (2,4,8)

    Batch 0: x @ W_Q
    ┌─────────────────────────────────────────────────────────┐
    │ Token 0: [q00, q01, q02, q03, q04, q05, q06, q07]      │
    │ Token 1: [q10, q11, q12, q13, q14, q15, q16, q17]      │
    │ Token 2: [q20, q21, q22, q23, q24, q25, q26, q27]      │
    │ Token 3: [q30, q31, q32, q33, q34, q35, q36, q37]      │
    └─────────────────────────────────────────────────────────┘

    Batch 0: x @ W_K
    ┌─────────────────────────────────────────────────────────┐
    │ Token 0: [k00, k01, k02, k03, k04, k05, k06, k07]      │
    │ Token 1: [k10, k11, k12, k13, k14, k15, k16, k17]      │
    │ Token 2: [k20, k21, k22, k23, k24, k25, k26, k27]      │
    │ Token 3: [k30, k31, k32, k33, k34, k35, k36, k37]      │
    └─────────────────────────────────────────────────────────┘
    
    Batch 0: x @ W_V
    ┌─────────────────────────────────────────────────────────┐
    │ Token 0: [v00, v01, v02, v03, v04, v05, v06, v07]      │
    │ Token 1: [v10, v11, v12, v13, v14, v15, v16, v17]      │
    │ Token 2: [v20, v21, v22, v23, v24, v25, v26, v27]      │
    │ Token 3: [v30, v31, v32, v33, v34, v35, v36, v37]      │
    └─────────────────────────────────────────────────────────┘

    =================================================================================
    (2) Split heads
    =================================================================================
    Before split: (2, 4, 8)
                
    Batch 0:
    ┌─────────────────────────────────────────────────────────┐
    │ T0: [q00, q01, q02, q03 │ q04, q05, q06, q07]          │
    │ T1: [q10, q11, q12, q13 │ q14, q15, q16, q17]          │
    │ T2: [q20, q21, q22, q23 │ q24, q25, q26, q27]          │
    │ T3: [q30, q31, q32, q33 │ q34, q35, q36, q37]          │
    └─────────────────────────────────────────────────────────┘
            ↑                   ↑
        Head 0 dims         Head 1 dims
        (0-3)               (4-7)

    After split_heads: (2, 2, 4, 4) = (batch, heads, seq, head_dim)
                
    Batch 0, Head 0:              Batch 0, Head 1:

    Q projections
    ┌─────────────────────┐       ┌─────────────────────┐
    │ T0: [q00, q01, q02, q03]   │ T0: [q04, q05, q06, q07]
    │ T1: [q10, q11, q12, q13]   │ T1: [q14, q15, q16, q17]
    │ T2: [q20, q21, q22, q23]   │ T2: [q24, q25, q26, q27]
    │ T3: [q30, q31, q32, q33]   │ T3: [q34, q35, q36, q37]
    └─────────────────────┘       └─────────────────────┘

    K projections
    ┌─────────────────────┐       ┌─────────────────────┐
    │ T0: [k00, k01, k02, k03]   │ T0: [k04, k05, k06, k07]
    │ T1: [k10, k11, k12, k13]   │ T1: [k14, k15, k16, k17]
    │ T2: [k20, k21, k22, k23]   │ T2: [k24, k25, k26, k27]
    │ T3: [k30, k31, k32, k33]   │ T3: [k34, k35, k36, k37]
    └─────────────────────┘       └─────────────────────┘
    
    V projections
    ┌─────────────────────┐       ┌─────────────────────┐
    │ T0: [v00, v01, v02, v03]   │ T0: [v04, v05, v06, v07]
    │ T1: [v10, v11, v12, v13]   │ T1: [v14, v15, v16, v17]
    │ T2: [v20, v21, v22, v23]   │ T2: [v24, v25, v26, v27]
    │ T3: [v30, v31, v32, v33]   │ T3: [v34, v35, v36, v37]
    └─────────────────────┘       └─────────────────────┘
    =================================================================================
    (3) Compute attention
    =================================================================================
    For head 0, batch 0:
    
    Query @ Key^T:

            Key positions
                T0        T1        T2        T3
             ┌────────────────────────────────────────┐
          T0 │ Q0·K0    Q0·K1    Q0·K2    Q0·K3       │
    Query T1 │ Q1·K0    Q1·K1    Q1·K2    Q1·K3       │
          T2 │ Q2·K0    Q2·K1    Q2·K2    Q2·K3       │
          T3 │ Q3·K0    Q3·K1    Q3·K2    Q3·K3       │
             └────────────────────────────────────────┘

    Where:
    Q0·K0 = q00*k00 + q01*k01 + q02*k02 + q03*k03  (dot product)
    Q0·K1 = q00*k10 + q01*k11 + q02*k12 + q03*k13
    ...

    After scaling (÷ √4 = ÷ 2):

    Scores: shape (2, 2, 4, 4)

    Batch 0, Head 0:
    ┌────────────────────────────────────────────────────────┐
    │         T0       T1       T2       T3                  │
    │  T0  [ 0.92,    0.15,   -0.30,    0.08 ]              │
    │  T1  [ 0.21,    0.88,    0.12,   -0.05 ]              │
    │  T2  [-0.15,    0.25,    0.95,    0.18 ]              │
    │  T3  [ 0.05,   -0.10,    0.20,    0.91 ]              │
    └────────────────────────────────────────────────────────┘

    Batch 0, Head 1:
    ┌────────────────────────────────────────────────────────┐
    │         T0       T1       T2       T3                  │
    │  T0  [ 0.45,    0.55,    0.10,   -0.20 ]              │
    │  T1  [ 0.30,    0.40,    0.60,    0.15 ]              │
    │  T2  [ 0.05,    0.25,    0.50,    0.70 ]              │
    │  T3  [ 0.10,    0.20,    0.30,    0.80 ]              │
    └────────────────────────────────────────────────────────┘

    Example:
    Head 0: Diagonal dominance → tokens attend mostly to themselves
    Head 1: More distributed → tokens attend to multiple positions

    =================================================================================
    Masking
    =================================================================================
    
    Padding Mask
    ------------
    If Token 3 is padding in Batch 0:

    Mask for Batch 0:
    ┌─────────────────────┐
    │ [1, 1, 1, 0]        │  ← 0 means "don't attend"
    └─────────────────────┘

    Scores after masking (Head 0):
    ┌────────────────────────────────────────────────────────┐
    │         T0       T1       T2       T3                  │
    │  T0  [ 0.92,    0.15,   -0.30,   -inf   ]              │
    │  T1  [ 0.21,    0.88,    0.12,   -inf   ]              │
    │  T2  [-0.15,    0.25,    0.95,   -inf   ]              │
    │  T3  [ 0.05,   -0.10,    0.20,   -inf   ]              │
    └────────────────────────────────────────────────────────┘

    Causal mask (decoder)
    ---------------------
    ┌─────────────────────┐
    │ [1, 0, 0, 0]        │  T0 can see T0 only
    │ [1, 1, 0, 0]        │  T1 can see T0, T1
    │ [1, 1, 1, 0]        │  T2 can see T0, T1, T2
    │ [1, 1, 1, 1]        │  T3 can see all
    └─────────────────────┘

    Scores after masking:
    ┌────────────────────────────────────────────────────────┐
    │         T0       T1       T2       T3                  │
    │  T0  [ 0.92,   -inf,    -inf,    -inf   ]              │
    │  T1  [ 0.21,    0.88,   -inf,    -inf   ]              │
    │  T2  [-0.15,    0.25,    0.95,   -inf   ]              │
    │  T3  [ 0.05,   -0.10,    0.20,    0.91  ]              │
    └────────────────────────────────────────────────────────┘

    =================================================================================
    Softmax
    =================================================================================
    Before softmax (Head 0, Batch 0):
    ┌────────────────────────────────────────────────────────┐
    │         T0       T1       T2       T3                  │
    │  T0  [ 0.92,    0.15,   -0.30,    0.08 ]               │
    │  T1  [ 0.21,    0.88,    0.12,   -0.05 ]               │
    │  T2  [-0.15,    0.25,    0.95,    0.18 ]               │
    │  T3  [ 0.05,   -0.10,    0.20,    0.91 ]               │
    └────────────────────────────────────────────────────────┘

    After softmax (each row sums to 1.0):
    ┌────────────────────────────────────────────────────────┐
    │         T0       T1       T2       T3          Sum     │
    │  T0  [ 0.55,    0.26,    0.11,    0.08 ]  =  1.00      │
    │  T1  [ 0.20,    0.46,    0.19,    0.15 ]  =  1.00      │
    │  T2  [ 0.13,    0.20,    0.46,    0.21 ]  =  1.00      │
    │  T3  [ 0.15,    0.13,    0.21,    0.51 ]  =  1.00      │
    └────────────────────────────────────────────────────────┘

    Head 1 (different pattern):
    ┌────────────────────────────────────────────────────────┐
    │         T0       T1       T2       T3          Sum     │
    │  T0  [ 0.32,    0.35,    0.22,    0.11 ]  =  1.00      │
    │  T1  [ 0.21,    0.24,    0.31,    0.24 ]  =  1.00      │
    │  T2  [ 0.16,    0.19,    0.25,    0.40 ]  =  1.00      │
    │  T3  [ 0.14,    0.16,    0.20,    0.50 ]  =  1.00      │
    └────────────────────────────────────────────────────────┘

    Example:
    Row T0 in Head 0: "I (token 0) pay 55% attention to myself,
                    26% to token 1, 11% to token 2, 8% to token 3"

    Row T2 in Head 1: "I (token 2) pay 40% attention to token 3,
                    25% to myself, 19% to token 1, 16% to token 0"

    =================================================================================
    Weighted sum with values
    =================================================================================
    For Head 0, Batch 0:

    Output_T0 = 0.55 x Value_T0 + 0.26 x Value_T1 + 0.11 x Value_T2 + 0.08 x Value_T3
            = 0.55 x [v00,v01,v02,v03] 
            + 0.26 x [v10,v11,v12,v13]
            + 0.11 x [v20,v21,v22,v23]
            + 0.08 x [v30,v31,v32,v33]

    Result for Head 0:
    ┌─────────────────────────────────────┐
    │ T0: [out00, out01, out02, out03]    │  ← weighted combination of all values
    │ T1: [out10, out11, out12, out13]    │
    │ T2: [out20, out21, out22, out23]    │
    │ T3: [out30, out31, out32, out33]    │
    └─────────────────────────────────────┘
    Shape: (2, 2, 4, 4) = (batch, heads, seq, head_dim)

    Result for Head 1:
    ┌─────────────────────────────────────┐
    │ T0: [out'00, out'01, out'02, out'03]│
    │ T1: [out'10, out'11, out'12, out'13]│
    │ T2: [out'20, out'21, out'22, out'23]│
    │ T3: [out'30, out'31, out'32, out'33]│
    └─────────────────────────────────────┘
    =================================================================================
    (4) Combine heads
    =================================================================================
    Before combine (Batch 0):
    Head 0:                       Head 1:
    ┌───────────────────────┐     ┌───────────────────────┐
    │ T0: [o00, o01, o02, o03]    │ T0: [o04, o05, o06, o07]
    │ T1: [o10, o11, o12, o13]    │ T1: [o14, o15, o16, o17]
    │ T2: [o20, o21, o22, o23]    │ T2: [o24, o25, o26, o27]
    │ T3: [o30, o31, o32, o33]    │ T3: [o34, o35, o36, o37]
    └───────────────────────┘     └───────────────────────┘

    After permute + view (concatenate):

    Batch 0:
    ┌─────────────────────────────────────────────────────────────────┐
    │ T0: [o00, o01, o02, o03, o04, o05, o06, o07]                    │
    │ T1: [o10, o11, o12, o13, o14, o15, o16, o17]                    │
    │ T2: [o20, o21, o22, o23, o24, o25, o26, o27]                    │
    │ T3: [o30, o31, o32, o33, o34, o35, o36, o37]                    │
    └─────────────────────────────────────────────────────────────────┘
             └── Head 0 ──┘       └── Head 1 ──┘

    Shape: (2, 4, 8) = (batch, seq_len, d_model)
    
    """
    def __init__(self, d_model: int, num_heads: int) -> None:
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.num_heads = num_heads
        self.d_model = d_model
        self.head_dim = d_model // num_heads # e.g. 512 // 8 = 64 -> each head works with 64-d vectors

        self.query_linear = nn.Linear(d_model, d_model, bias=False) # no biases as per original paper
        self.key_linear = nn.Linear(d_model, d_model, bias=False)   # not necessary and slightly reduces param count
        self.value_linear = nn.Linear(d_model, d_model, bias=False)
        self.output_linear = nn.Linear(d_model, d_model)

    def split_heads(self, x, batch_size):
        """
        Input: (batch_size, seq_len, d_model)
        Example: (32, 10, 512)

        reshape -> (32, 10, 8, 64)
        Split d_model=512 into num_heads=8 x head_dim=64

        permute -> (32, 8, 10, 64)
        Rearrange: (batch, num_heads, seq_len, head_dim)
        We want heads to be a separate dim so each head can compute attention independently
        """
        seq_length = x.size(1)
        x = x.reshape(batch_size, seq_length, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)
    
    def compute_attention(self, query, key, value, mask=None):
        """
        query:       (batch_size, n_heads, seq, head_dim)
        key:         (batch_size, n_heads, seq, head_dim)

        scores = softmax( Q @ K.T / sqrt(head_dim) )

        scores:      (batch_size, n_heads, seq, seq)  ← attention scores
        softmax:     (batch_size, n_heads, seq, seq)  ← normalised to probabilities

        value:       (batch_size, n_heads, seq, head_dim)

        output = attention_weights @ value

        output:      (batch_size, n_heads, seq, head_dim)  ← weighted sum of values
        """
        scores = torch.matmul(query, key.transpose(-2, -1)) / (self.head_dim ** 0.5)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        attention_weights = F.softmax(scores, dim=-1)
        return torch.matmul(attention_weights, value)
    
    def combine_heads(self, x, batch_size):
        """
        Input:      (batch_size, n_heads, seq, head_dim)

        permute ->  (batch_size, seq, n_heads, head_dim)

        view ->     (batch_size, seq, n_heads x head_dim)
        """
        # after permute, tensor memory layout doesnt match logical shape
        # call contiguous to rearrange so that view works after
        x = x.permute(0, 2, 1, 3).contiguous()
        return x.view(batch_size, -1, self.d_model)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # linear projections
        query = self.query_linear(query)
        key = self.key_linear(key)
        value = self.value_linear(value)

        # split into heads
        query = self.split_heads(query, batch_size)
        key = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)

        # compute attention
        attention_output = self.compute_attention(query, key, value, mask)

        # combine heads
        attention_output = self.combine_heads(attention_output, batch_size)

        # final linear projection
        return self.output_linear(attention_output)
    

class FeedForwardSubLayer(nn.Module):
    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))
    

class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: int) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.ff_sublayer = FeedForwardSubLayer(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, src_mask):
        attn_output = self.self_attn(x, x, x, src_mask) # q, k, v all x -> self attn, padding mask
        
        # dropout + residual + layer norm
        x = self.norm1(x + self.dropout(attn_output))

        ff_output = self.ff_sublayer(x)

        x = self.norm2(x + self.dropout(ff_output))

        return x


class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, num_layers: int, num_heads: int, 
                 d_ff: int, dropout: int, max_seq_length: int) -> None:
        super().__init__()
        self.embedding = InputEmbeddings(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )

    def forward(self, x, src_mask):
        x = self.embedding(x)
        x = self.positional_encoding(x)
        
        for layer in self.layers:
            x = layer(x, src_mask)
        return x