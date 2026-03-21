from models.transformer import (
    Transformer
)
import torch

def main():
    # hyperparameters (from original paper)
    src_vocab_size = 10000  # English vocabulary
    tgt_vocab_size = 8000   # French vocabulary
    d_model = 512
    num_heads = 8
    num_layers = 6
    d_ff = 2048
    max_seq_length = 512
    dropout = 0.1

    # create model
    model = Transformer(
        src_vocab_size, tgt_vocab_size, d_model,
        num_heads, num_layers, d_ff, 
        max_seq_length, dropout
    )

    # sample inputs
    batch_size = 2
    src_seq_len = 10  # english sequence length
    tgt_seq_len = 8   # french sequence length

    # source: english tokens
    src = torch.randint(0, src_vocab_size, (batch_size, src_seq_len))

    # target: french tokens (shifted right for training)
    tgt = torch.randint(0, tgt_vocab_size, (batch_size, tgt_seq_len))

    # masks
    src_mask = (src != 0).unsqueeze(1).unsqueeze(1)  # assume 0 is padding

    # causal mask for target
    causal_mask = torch.tril(torch.ones(tgt_seq_len, tgt_seq_len, dtype=torch.bool)).unsqueeze(0).unsqueeze(0)
    tgt_padding_mask = (tgt != 0).unsqueeze(1).unsqueeze(2)
    tgt_mask = causal_mask & tgt_padding_mask

    # forward pass
    output = model(src, tgt, src_mask, tgt_mask)

    print(f"Source shape:      {src.shape}")       # torch.Size([2, 10])
    print(f"Target shape:      {tgt.shape}")       # torch.Size([2, 8])
    print(f"Output shape:      {output.shape}")    # torch.Size([2, 8, 8000])
    print(f"Output represents: log probabilities over French vocabulary")

if __name__ == "__main__":
    main()