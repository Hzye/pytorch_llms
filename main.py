from models.ssm import (
    VanillaSSM
)
from models.bert import (
    BertEncoderLayer
)
import torch

def verify_modes_match(
    d_model:   int   = 8,
    state_dim: int   = 4,
    seq_len:   int   = 16,
    batch:     int   = 2,
) -> bool:
    """
    Verify that the recurrent and convolutional passes produce identical outputs.
 
    This is the most important sanity check when implementing any SSM.
    If the two modes diverge, there is a bug in the kernel construction
    or the FFT convolution setup.
 
    A tolerance of 1e-5 is appropriate for float32 FFT accumulation error.
    """
    model = VanillaSSM(d_model=d_model, state_dim=state_dim)
    model.eval()
 
    x = torch.randn(batch, seq_len, d_model)
 
    with torch.no_grad():
        y_rec  = model(x, mode='recurrent')
        y_conv = model(x, mode='convolutional')
 
    max_diff = (y_rec - y_conv).abs().max().item()
    match    = torch.allclose(y_rec, y_conv, atol=1e-5)
 
    print("=" * 58)
    print("VanillaSSM — mode equivalence check")
    print("=" * 58)
    print(f"  d_model   : {d_model}")
    print(f"  state_dim : {state_dim}")
    print(f"  seq_len   : {seq_len}")
    print(f"  batch     : {batch}")
    print()
    print(f"  Recurrent output shape    : {tuple(y_rec.shape)}")
    print(f"  Convolutional output shape: {tuple(y_conv.shape)}")
    print(f"  Max absolute difference   : {max_diff:.2e}")
    print(f"  Outputs match (atol=1e-5) : {'✓  YES' if match else '✗  NO — BUG!'}")
    print("=" * 58)
 
    return match

def main():
    verify_modes_match()

if __name__ == "__main__":
    main()