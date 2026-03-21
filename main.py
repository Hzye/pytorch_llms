from models.transformer import (
    Transformer
)
from models.bert import (
    BertEncoderLayer
)
import torch

def main():
    # BERT-base dimensions
    d_model = 768
    num_heads = 12
    d_ff = 3072  # 4 * d_model
    
    layer = BertEncoderLayer(d_model, num_heads, d_ff)
    
    # Simulated input from embeddings
    batch_size, seq_len = 2, 9
    x = torch.randn(batch_size, seq_len, d_model)
    
    # Optional: attention mask for padding
    mask = torch.ones(batch_size, 1, 1, seq_len)  # No padding in this example
    
    output = layer(x, mask)
    
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # Parameter count
    total_params = sum(p.numel() for p in layer.parameters())
    print(f"Parameters: {total_params:,}")

if __name__ == "__main__":
    main()