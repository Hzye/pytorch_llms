import torch
import torch.nn.functional as F
from pathlib import Path

# Change this import to match your file/module name
from models.s6 import TinyS6LM


# ------------------------------------------------------------
# Config
# ------------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

batch_size = 16
block_size = 128         # context length
max_iters = 2000
eval_interval = 10
eval_iters = 50
learning_rate = 3e-4
weight_decay = 1e-2
grad_clip = 1.0

# model size
d_model = 128
state_dim = 16
n_layers = 4

# ------------------------------------------------------------
# Load tiny_shakespeare.txt
# ------------------------------------------------------------
data_path = Path("data/tiny_shakespeare.txt")
text = data_path.read_text(encoding="utf-8")

# character-level vocab
chars = sorted(set(text))
vocab_size = len(chars)

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}

def encode(s: str) -> torch.Tensor:
    return torch.tensor([stoi[c] for c in s], dtype=torch.long)

def decode(ids) -> str:
    return "".join(itos[int(i)] for i in ids)

data = encode(text)

# train/val split
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

print(f"device: {device}")
print(f"dataset chars: {len(data):,}")
print(f"vocab size: {vocab_size}")
print(f"train chars: {len(train_data):,}")
print(f"val chars:   {len(val_data):,}")


# ------------------------------------------------------------
# Batch sampling
# ------------------------------------------------------------
def get_batch(split: str):
    source = train_data if split == "train" else val_data
    ix = torch.randint(0, len(source) - block_size - 1, (batch_size,))

    x = torch.stack([source[i:i + block_size] for i in ix])
    y = torch.stack([source[i + 1:i + block_size + 1] for i in ix])

    return x.to(device), y.to(device)


# ------------------------------------------------------------
# Model
# ------------------------------------------------------------
model = TinyS6LM(
    vocab_size=vocab_size,
    d_model=d_model,
    state_dim=state_dim,
    n_layers=n_layers,
).to(device)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=learning_rate,
    weight_decay=weight_decay,
)

num_params = sum(p.numel() for p in model.parameters())
print(f"model params: {num_params:,}")


# ------------------------------------------------------------
# Eval helper
# ------------------------------------------------------------
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()

    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters, device=device)
        for k in range(eval_iters):
            x, y = get_batch(split)
            logits = model(x)  # (B, T, vocab)
            B, T, V = logits.shape
            loss = F.cross_entropy(
                logits.reshape(B * T, V),
                y.reshape(B * T),
            )
            losses[k] = loss
        out[split] = losses.mean().item()

    model.train()
    return out


# ------------------------------------------------------------
# Simple generation helper
# ------------------------------------------------------------
@torch.no_grad()
def generate(model, idx, max_new_tokens, block_size, temperature=1.0):
    """
    idx: (batch, current_length) integer token ids
    """
    model.eval()
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -block_size:]   # crop context if needed
        logits = model(idx_cond)          # (B, T, vocab)
        logits = logits[:, -1, :] / temperature
        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        idx = torch.cat([idx, next_id], dim=1)
    model.train()
    return idx


# ------------------------------------------------------------
# Training loop
# ------------------------------------------------------------
for step in range(max_iters):
    if step % eval_interval == 0:
        losses = estimate_loss()
        print(
            f"step {step:5d} | "
            f"train loss {losses['train']:.4f} | "
            f"val loss {losses['val']:.4f}"
        )

    xb, yb = get_batch("train")
    logits = model(xb)  # (B, T, vocab)
    B, T, V = logits.shape

    loss = F.cross_entropy(
        logits.reshape(B * T, V),
        yb.reshape(B * T),
    )

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()


# ------------------------------------------------------------
# Generate sample text
# ------------------------------------------------------------
prompt = "ROMEO:\n"
start = encode(prompt).unsqueeze(0).to(device)
sample_ids = generate(
    model,
    idx=start,
    max_new_tokens=300,
    block_size=block_size,
    temperature=1.0,
)

print("\n--- sample ---\n")
print(decode(sample_ids[0].tolist()))