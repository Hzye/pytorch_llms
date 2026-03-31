# LLM Architectures from Scratch

A collection of notable Large Language Model architectures implemented from scratch in PyTorch for learning purposes.

## Objective

To deeply understand modern LLM architectures by building them component-by-component, without relying on high-level abstractions from libraries like Hugging Face or FairSeq.

## Implemented Architectures

| Architecture | Status | File |
|--------------|--------|------|
| **Transformer** (Vaswani et al., 2017) | ✅ Complete | `transformer.py` |
| **BERT** (Devlin et al., 2018) | ✅ Complete | `bert.py` |
| **SSM** (Gu et al., 2021) | ✅ Complete | `ssm.py` |

### Coming Soon

- GPT (Decoder-only)
- Mixture of Experts (MoE)

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