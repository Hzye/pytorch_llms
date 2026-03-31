from models.s6 import SelectiveSSMCore
import torch

if __name__ == "__main__":
    torch.manual_seed(0)

    model = SelectiveSSMCore(d_model=4, state_dim=8)
    x = torch.randn(2, 16, 4)   # (batch=2, seq=16, d_model=4)

    y = model(x)

    print("x shape:", x.shape)
    print("y shape:", y.shape)
    print("A min/max:", model.get_A().min().item(), model.get_A().max().item())