from models.s4d import S4D
import torch

def verify_modes_match(
    d_model:   int = 8,
    state_dim: int = 4,
    seq_len:   int = 16,
    batch:     int = 2,
) -> None:
    model = S4D(d_model=d_model, state_dim=state_dim)
    model.eval()
 
    x = torch.randn(batch, seq_len, d_model)
 
    with torch.no_grad():
        y_rec  = model(x, mode='recurrent')
        y_conv = model(x, mode='convolutional')
 
    max_diff = (y_rec - y_conv).abs().max().item()
    match    = torch.allclose(y_rec, y_conv, atol=1e-5)
 
    print("=" * 58)
    print("S4D — mode equivalence check")
    print("=" * 58)
    print(f"  d_model   : {d_model}")
    print(f"  state_dim : {state_dim}")
    print(f"  seq_len   : {seq_len}")
    print(f"  batch     : {batch}")
    print()
    print(f"  Recurrent  shape : {tuple(y_rec.shape)}")
    print(f"  Conv       shape : {tuple(y_conv.shape)}")
    print(f"  Max |diff|       : {max_diff:.2e}")
    print(f"  Match (atol=1e-5): {'✓  YES' if match else '✗  NO — BUG!'}")
    print("=" * 58)
 
 
def inspect_init(d_model: int = 4, state_dim: int = 8) -> None:
    """
    Print initial parameter values to confirm the HiPPO and dt spreads.
    """
    model = S4D(d_model=d_model, state_dim=state_dim)
 
    A  = -torch.exp(model.log_A[0]).detach()   # feature 0, all N dims
    dt =  torch.exp(model.log_dt).detach()
 
    print()
    print("=" * 58)
    print("S4D — initial parameter inspection")
    print("=" * 58)
 
    print("\nA values (feature 0) — should be -(n+1):")
    for n, a_val in enumerate(A):
        print(f"  n={n}: a = {a_val:.4f}  (expected {-(n+1):.1f})")
 
    print("\ndt values (all features) — should span [dt_min, dt_max]:")
    for d, dt_val in enumerate(dt):
        print(f"  feature {d}: dt = {dt_val:.5f}")
 
    print(f"\n  dt min : {dt.min().item():.5f}")
    print(f"  dt max : {dt.max().item():.5f}")
    print("=" * 58)

def main():
    verify_modes_match()

if __name__ == "__main__":
    main()