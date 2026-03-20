# LLM Architectures from Scratch

A collection of notable Large Language Model architectures implemented from scratch in PyTorch for learning purposes.

## Objective

To deeply understand modern LLM architectures by building them component-by-component, without relying on high-level abstractions from libraries like Hugging Face or FairSeq.

## Implemented Architectures

| Architecture | Status | File |
|--------------|--------|------|
| **Transformer** (Vaswani et al., 2017) | ✅ Complete | `transformer.py` |

### Coming Soon

- GPT (Decoder-only)
- BERT (Encoder-only)
- Mixture of Experts (MoE)

## Current Implementation: Transformer

The original encoder-decoder Transformer from ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762).

### Architecture Overview

```
┌─────────────────────────────────────────┐
│              TRANSFORMER                │
│                                         │
│   Source --→ ENCODER ──┐                │
│                        │                │
│                        ▼                │
│   Target --→ DECODER --→ Output         │
│                                         │
└─────────────────────────────────────────┘
```

### Components

- **Input Embeddings** — Token ID → dense vector lookup
- **Positional Encoding** — Sinusoidal position signals
- **Multi-Head Attention** — Parallel attention heads with learned projections
- **Feed-Forward Network** — Position-wise expansion and contraction
- **Layer Normalization** — Stabilizes training
- **Residual Connections** — Gradient flow optimization

### Usage

```python
from transformer import Transformer

# Initialize model
model = Transformer(
    src_vocab_size=10000,
    tgt_vocab_size=8000,
    d_model=512,
    num_heads=8,
    num_layers=6,
    d_ff=2048,
    max_seq_length=512,
    dropout=0.1
)

# Forward pass
output = model(src, tgt, src_mask, tgt_mask)
# output: (batch, tgt_seq_len, tgt_vocab_size) log probabilities
```

### Key Features

- ✅ Multi-head self-attention & cross-attention
- ✅ Sinusoidal positional encoding
- ✅ Post-layer normalization (original paper style)
- ✅ Weight tying (embedding ↔ output projection)
- ✅ Masking support (causal + padding)

## Requirements

```
torch>=2.0
```

## References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Vaswani et al., 2017
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) — Jay Alammar
- [Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/) — Harvard NLP

## Structure

```
pytorch_llms/
├── transformer.py      # Original encoder-decoder Transformer
├── main.py             # Training/testing entry point
└── README.md
```

---

*Built for learning. Not optimized for production.*