"""
Bidirectional Encoder Representations from Transformers (BERT)

https://arxiv.org/abs/1810.04805

===================================================================================
Applications
===================================================================================

-----------------------------------------------------------------------------------
1. Pre-training: Next Sentence Prediction (NSP)
-----------------------------------------------------------------------------------
Input:  "[CLS] The cat sat [SEP] It was happy [SEP]"
        └─ Sentence A ─┘  └─ Sentence B ─┘

Task: Is "It was happy" the actual next sentence after "The cat sat"?

Training examples:
  50% true pairs (IsNext):
    "[CLS] The cat sat [SEP] It was happy [SEP]" → Label: IsNext
  
  50% random pairs (NotNext):
    "[CLS] The cat sat [SEP] The stock market crashed [SEP]" → Label: NotNext

Purpose: Teach BERT to understand relationships between sentences.

-----------------------------------------------------------------------------------
2. Question answering (SQuAD-style)
-----------------------------------------------------------------------------------
Input:  "[CLS] Where is the cat? [SEP] The cat sat on the mat [SEP]"
        └── Question ────┘  └──── Context ────┘

Task: Find the answer span in the context.

Output: Start position = 1 ("The"), End position = 3 ("mat")
        Answer: "The cat" or "on the mat" depending on the question

-----------------------------------------------------------------------------------
3. Natural Language Inference (NLI)
-----------------------------------------------------------------------------------
Input:  "[CLS] A man is playing guitar [SEP] A person is making music [SEP]"
        └─── Premise ──────────┘  └─── Hypothesis ────────┘

Task: Does the premise entail the hypothesis?

Labels:
  - Entailment (must be true)
  - Contradiction (cannot be true)
  - Neutral (might be true)

Example above: Entailment (playing guitar IS making music)

-----------------------------------------------------------------------------------
4. Semantic similarity/paraphrase
-----------------------------------------------------------------------------------
Input:  "[CLS] The cat sat on the mat [SEP] A feline rested on the rug [SEP]"
        └──── Sentence A ─────────┘  └──── Sentence B ───────────┘

Task: Are these sentences semantically equivalent?

Output: Similarity score (0 to 1, or binary classification)

-----------------------------------------------------------------------------------
5. Search/Relevance ranking
-----------------------------------------------------------------------------------
Input:  "[CLS] how to bake a cake [SEP] Preheat oven to 350F and mix ingredients [SEP]"
        └─── Query ─────────────┘  └───────── Document ─────────────────┘

Task: Is this document relevant to the query?

Output: Relevance score (used in search engines)

-----------------------------------------------------------------------------------
6. Text classification
-----------------------------------------------------------------------------------
Input:  "[CLS] This movie was absolutely terrible [SEP]"
        └────────── Single sentence ───────────┘

segment_ids: [0, 0, 0, 0, 0, 0, 0, 0, 0]  (all zeros - no sentence B)

Task: Classify the sentiment of the text.

Output: Single label (Positive / Negative / Neutral)

[CLS] token's output representation → Linear layer → Softmax → [0.15, 0.82, 0.03]
                                                         Negative ↑
"""

import torch
import math
import torch.nn as nn
import torch.nn.functional as F

class BertEmbeddings(nn.Module):
    """
    Token + Position + Segment

    =================================================================================
    (1) Create embedding tables
    =================================================================================
    token_embeddings: Shape (vocab_size, d_model) = (30522, 768)
    ┌─────────────────────────────────────────────────────────┐
    │ Row 0:    [0.02, -0.15, 0.08, ..., 0.11]                │ ← token ID 0
    │ Row 101:  [0.33, 0.07, -0.21, ..., 0.05]                │ ← [CLS]
    │ Row 1996: [-0.14, 0.22, 0.03, ..., -0.08]               │ ← "The"
    │ ...                                                     │
    │ Row 30521: [0.01, -0.03, 0.15, ..., 0.09]               │ ← last token
    └─────────────────────────────────────────────────────────┘
    Each row is a 768-dim vector learned during pre-training

    position_embeddings: Shape (max_seq_length, d_model) = (512, 768)
    ┌─────────────────────────────────────────────────────────┐
    │ Row 0:   [0.12, -0.05, 0.18, ..., 0.02]                 │ ← position 0
    │ Row 1:   [-0.07, 0.11, 0.04, ..., -0.15]                │ ← position 1
    │ Row 2:   [0.09, 0.03, -0.12, ..., 0.21]                 │ ← position 2
    │ ...                                                     │
    │ Row 511: [0.05, -0.18, 0.07, ..., 0.13]                 │ ← position 511
    └─────────────────────────────────────────────────────────┘
    LEARNED during training (unlike fixed sinusoidal)

    segment_embeddings: Shape (2, d_model) = (2, 768)
    ┌─────────────────────────────────────────────────────────┐
    │ Row 0: [0.08, 0.14, -0.03, ..., 0.11]                   │ ← sentence A
    │ Row 1: [-0.05, 0.09, 0.17, ..., -0.07]                  │ ← sentence B
    └─────────────────────────────────────────────────────────┘
    Only 2 rows because there are only 2 segment types

    =================================================================================
    (2) Input tensors
    =================================================================================
    token_ids: Shape (2, 9)
    ┌──────────────────────────────────────────────────────┐
    │ [101, 1996, 4937, 2363, 102, 2009, 2001, 5719, 102]  │ ← batch 0
    │ [101, 1996, 4937, 102, 2009, 2001, 5719, 102, 0]     │ ← batch 1 (shorter)
    └──────────────────────────────────────────────────────┘

    segment_ids: Shape (2, 9)
    ┌───────────────────────────────────────────────┐
    │ [0, 0, 0, 0, 0, 1, 1, 1, 1]                   │ ← batch 0: sentence A (pos 0-4), B (pos 5-8)
    │ [0, 0, 0, 0, 1, 1, 1, 1, 0]                   │ ← batch 1: shorter sentence A
    └───────────────────────────────────────────────┘

    Extracted values:
    batch_size = 2
    seq_len = 9

    =================================================================================
    (3) Create position ids 
    =================================================================================
    torch.arange(9) → [0, 1, 2, 3, 4, 5, 6, 7, 8]
    Shape: (9,)

    .unsqueeze(0) → [[0, 1, 2, 3, 4, 5, 6, 7, 8]]
    Shape: (1, 9)

    .expand(2, -1) → [[0, 1, 2, 3, 4, 5, 6, 7, 8],
                    [0, 1, 2, 3, 4, 5, 6, 7, 8]]
    Shape: (2, 9)

    Same position IDs for every sample in the batch

    =================================================================================
    (4) Initialise segment ids 
    =================================================================================
    if no segment_ids provided, assume single sentence, then segment_ids would become:
    ┌───────────────────────────────────────────────┐
    │ [0, 0, 0, 0, 0, 0, 0, 0, 0]                   │
    │ [0, 0, 0, 0, 0, 0, 0, 0, 0]                   │
    └───────────────────────────────────────────────┘
    All tokens belong to sentence A

    =================================================================================
    (4) Look up embeddings 
    =================================================================================
    Token
    -----
    token_ids: [101, 1996, 4937, 2363, 102, 2009, 2001, 5719, 102]

    For each token ID, look up the corresponding row:

    token_embed for batch 0:
    ┌─────────────────────────────────────────────────────────────────┐
    │ pos 0: token_embeddings[101]  = [0.33, 0.07, -0.21, ..., 0.05] │ [CLS]
    │ pos 1: token_embeddings[1996] = [-0.14, 0.22, 0.03, ..., -0.08]│ "The"
    │ pos 2: token_embeddings[4937] = [0.08, -0.11, 0.19, ..., 0.14] │ "cat"
    │ pos 3: token_embeddings[2363] = [0.21, 0.04, -0.07, ..., 0.11] │ "sat"
    │ pos 4: token_embeddings[102]  = [-0.02, 0.15, 0.09, ..., -0.03]│ [SEP]
    │ pos 5: token_embeddings[2009] = [0.17, -0.08, 0.12, ..., 0.06] │ "It"
    │ pos 6: token_embeddings[2001] = [-0.05, 0.13, 0.02, ..., 0.18] │ "was"
    │ pos 7: token_embeddings[5719] = [0.11, 0.06, -0.14, ..., -0.09]│ "happy"
    │ pos 8: token_embeddings[102]  = [-0.02, 0.15, 0.09, ..., -0.03]│ [SEP]
    └─────────────────────────────────────────────────────────────────┘
    Shape: (2, 9, 768)

    Position
    --------
    position_ids: [0, 1, 2, 3, 4, 5, 6, 7, 8]

    position_embed:
    ┌─────────────────────────────────────────────────────────────────┐
    │ pos 0: position_embeddings[0] = [0.12, -0.05, 0.18, ..., 0.02] │
    │ pos 1: position_embeddings[1] = [-0.07, 0.11, 0.04, ..., -0.15]│
    │ pos 2: position_embeddings[2] = [0.09, 0.03, -0.12, ..., 0.21] │
    │ pos 3: position_embeddings[3] = [0.15, -0.08, 0.06, ..., 0.09] │
    │ pos 4: position_embeddings[4] = [-0.03, 0.17, 0.11, ..., -0.04]│
    │ pos 5: position_embeddings[5] = [0.08, 0.02, -0.09, ..., 0.14] │
    │ pos 6: position_embeddings[6] = [-0.11, 0.14, 0.05, ..., 0.07] │
    │ pos 7: position_embeddings[7] = [0.04, -0.06, 0.16, ..., -0.11]│
    │ pos 8: position_embeddings[8] = [0.13, 0.09, -0.02, ..., 0.18] │
    └─────────────────────────────────────────────────────────────────┘
    Shape: (2, 9, 768)

    Note: Same for both batches (same sequence positions)

    Segment
    -------
    segment_ids: [0, 0, 0, 0, 0, 1, 1, 1, 1]

    segment_embed:
    ┌─────────────────────────────────────────────────────────────────┐
    │ pos 0: segment_embeddings[0] = [0.08, 0.14, -0.03, ..., 0.11]  │ sentence A
    │ pos 1: segment_embeddings[0] = [0.08, 0.14, -0.03, ..., 0.11]  │ sentence A
    │ pos 2: segment_embeddings[0] = [0.08, 0.14, -0.03, ..., 0.11]  │ sentence A
    │ pos 3: segment_embeddings[0] = [0.08, 0.14, -0.03, ..., 0.11]  │ sentence A
    │ pos 4: segment_embeddings[0] = [0.08, 0.14, -0.03, ..., 0.11]  │ sentence A
    │ pos 5: segment_embeddings[1] = [-0.05, 0.09, 0.17, ..., -0.07] │ sentence B
    │ pos 6: segment_embeddings[1] = [-0.05, 0.09, 0.17, ..., -0.07] │ sentence B
    │ pos 7: segment_embeddings[1] = [-0.05, 0.09, 0.17, ..., -0.07] │ sentence B
    │ pos 8: segment_embeddings[1] = [-0.05, 0.09, 0.17, ..., -0.07] │ sentence B
    └─────────────────────────────────────────────────────────────────┘
    Shape: (2, 9, 768)

    Positions 0-4 get segment A embedding
    Positions 5-8 get segment B embedding

    =================================================================================
    (4) Sum, layer norm, dropout 
    =================================================================================
    For position 0 ([CLS] token):

    token_embed[0]:    [0.33,  0.07, -0.21, ...,  0.05]
    position_embed[0]: [0.12, -0.05,  0.18, ...,  0.02]
    segment_embed[0]:  [0.08,  0.14, -0.03, ...,  0.11]
                    ─────────────────────────────────
    Sum:               [0.53,  0.16, -0.06, ...,  0.18]

    For position 5 ("It" token in sentence B):

    token_embed[5]:    [0.17, -0.08,  0.12, ...,  0.06]
    position_embed[5]: [0.08,  0.02, -0.09, ...,  0.14]
    segment_embed[5]:  [-0.05, 0.09,  0.17, ..., -0.07]
                    ─────────────────────────────────
    Sum:               [0.20,  0.03,  0.20, ...,  0.13]

    Final embeddings shape: (2, 9, 768)
    """
    def __init__(self, vocab_size: int, d_model: int, max_seq_length: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.d_model = d_model

        # token embeddings
        # same as inputembeddings from transformer but without sqrt scaling
        self.token_embeddings = nn.Embedding(vocab_size, d_model)

        # position embeddings
        # learned not sinusoidal
        # each pos gets own learned vector
        self.position_embeddings = nn.Embedding(max_seq_length, d_model)

        # segment embeddings
        # distinguish between sentence a and b, 0 for a, 1 for b.
        # used for q&a tasks where input is [question, context]
        self.segment_embeddings = nn.Embedding(2, d_model)

        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, token_ids, segment_ids):
        batch_size, seq_length = token_ids.shape
        device = segment_ids.device

        # create position ids
        # these are the same for every sample in the batch
        position_ids = torch.arange(seq_length, device=device)
        # expand to batch size: (seq_length,) -> (batch_size, seq_len)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

        # init segment_ids
        if segment_ids is None:
            segment_ids = torch.zeros_like(token_ids)

        # look up each embedding
        token_embed = self.token_embeddings(token_ids)
        position_embed = self.position_embeddings(position_ids)
        segment_embed = self.segment_embeddings(segment_ids)

        # sum all embeddings
        embeddings = token_embed + position_embed + segment_embed

        # norm and dropout
        embeddings = self.dropout(self.layer_norm(embeddings))

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
    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.gelu = nn.GELU()

    def forward(self, x):
        return self.fc2(self.gelu(self.fc1(x)))
    

class BertEncoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.ff_sublayer = FeedForwardSubLayer(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        attn_output = self.self_attn(x, x, x, mask)

        x = self.norm1(x + self.dropout(attn_output))

        ff_output = self.ff_sublayer(x)

        x = self.norm2(x + self.dropout(ff_output))

        return x


class Pooler(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()

        self.linear = nn.Linear(d_model, d_model)
        self.tanh = nn.Tanh()

    def forward(self, x):
        return self.tanh(self.linear(x))


class BERT(nn.Module):
    """
    ┌─────────────────────────────────────────────────────────────────────────────┐
    │                           BERT MODEL                                        │
    ├─────────────────────────────────────────────────────────────────────────────┤
    │                                                                             │
    │  Input: token_ids, segment_ids, attention_mask                              │
    │                                                                             │
    │         │                                                                   │
    │         ▼                                                                   │
    │  ┌─────────────────────────────────────┐                                    │
    │  │        BertEmbeddings               │                                    │
    │  │  token + position + segment         │                                    │
    │  │  LayerNorm, Dropout                 │                                    │
    │  └─────────────────────────────────────┘                                    │
    │         │                                                                   │
    │         ▼                                                                   │
    │  ┌─────────────────────────────────────┐                                    │
    │  │        BertEncoderLayer 1           │                                    │
    │  │  Self-Attn → Add&Norm → FFN → Add&Norm │                                 │
    │  └─────────────────────────────────────┘                                    │
    │         │                                                                   │
    │         ⋮ (repeated N times)                                                │
    │         │                                                                   │
    │         ▼                                                                   │
    │  ┌─────────────────────────────────────┐                                    │
    │  │        BertEncoderLayer N           │                                    │
    │  └─────────────────────────────────────┘                                    │
    │         │                                                                   │
    │         ├──────────────────────┐                                            │
    │         ▼                      ▼                                            │
    │  sequence_output         pooled_output                                      │
    │  (batch, seq, d_model)   (batch, d_model)                                   │
    │                                  │                                          │
    │                                  ▼                                          │
    │                         ┌─────────────────┐                                 │
    │                         │    Pooler       │                                 │
    │                         │ Linear + Tanh   │                                 │
    │                         │ [CLS] token     │                                 │
    │                         └─────────────────┘                                 │
    │                                                                             │
    └─────────────────────────────────────────────────────────────────────────────┘
    """
    def __init__(self, vocab_size: int, d_model: int, num_layers: int, num_heads: int, 
                 d_ff: int, max_seq_length: int, dropout: float = 0.1) -> None:
        super().__init__()

        self.embeddings = BertEmbeddings(vocab_size, d_model, max_seq_length, dropout)
        self.layers = nn.ModuleList(
            [BertEncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )

        # pooler
        self.pooler = Pooler(d_model)

    def forward(self, token_ids, segment_ids, attention_mask):
        x = self.embeddings(x, token_ids, segment_ids)
        
        # prep attention mask for mha
        # attention_mask: (batch, seq_len) -> (batch, 1, 1, seq_len)
        #
        # full example:
        # attention_mask: (batch, seq_len)
        # Example: [[1, 1, 1, 1, 0, 0],   # batch 0: 4 real tokens, 2 padding
        #           [1, 1, 1, 1, 1, 1]]   # batch 1: all real tokens

        # After unsqueeze(1).unsqueeze(2):
        # Shape: (batch, 1, 1, seq_len)
        # Example: [[[[1, 1, 1, 1, 0, 0]]],
        #           [[[1, 1, 1, 1, 1, 1]]]]

        # This shape broadcasts correctly with attention scores:
        # attention_scores: (batch, num_heads, seq_len, seq_len)
        # extended_mask:    (batch, 1,          1,        seq_len)
        # Broadcasting works across num_heads and query positions
        if attention_mask is not None:
            extended_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        else:
            extended_mask = None

        for layer in self.layers:
            x = layer(x, extended_mask)

        # (batch, seq_length, d_model)
        x_sequence = x

        # pool the [CLS] token (first position)
        # used for classification tasks
        x_pooled = x[:, 0, :]
        x_pooled = self.pooler(x_pooled)

        return x_sequence, x_pooled