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
| **S4D** (Gu et al., 2021) | ✅ Complete | `s4d.py` |
| **S6** (Gu et al., 2021) | ✅ Complete | `s6.py` |

### Coming Soon

- GPT (Decoder-only)
- Mixture of Experts (MoE)

## S6 Training on tiny-shakespeare Example
```text
>> python train_tiny_s6.py
device: cuda
dataset chars: 1,115,394
vocab size: 65
train chars: 1,003,854
val chars:   111,540
model params: 2,198,400

step     0 | train loss 125.2211 | val loss 125.0526
step    10 | train loss 120.6869 | val loss 120.5564
step    20 | train loss 110.9077 | val loss 111.0418
...
step  1970 | train loss 1.7398 | val loss 1.9449
step  1980 | train loss 1.7417 | val loss 1.9198
step  1990 | train loss 1.7445 | val loss 1.9262

--- sample ---

ROMEO:

COMIONA:
O, maist will my I come, of me sir the cre
Pavesty would you?

JOHN lord, my prises; a herses For,
And his is onfult's it im, fol follow'd
OWith ingot not
Whicent nown marst enanty fouth'd so
offend my him tops, parine the irle
Kinsal a likea feardled with dead
My Imind chils no shall I de
```

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