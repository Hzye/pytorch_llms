"""
S4D — Diagonal State Space Model
 
Reference:
    "On the Parameterization and Initialization of Diagonal State Space Models"
    (Gu, Gupta, Goel, Re, 2022)  https://arxiv.org/abs/2206.11893

VanillaSSM -> Structured SSMs (S4, S4D)

===================================================================================
What S4D adds over VanillaSSM  (two changes only)
===================================================================================
 
Change 1 — Learned per-channel log_dt
--------------------------------------
VanillaSSM:  dt is a fixed float shared across ALL features and set at __init__.
S4D:         log_dt is an nn.Parameter of shape (d_model,) — one learned step
             size per feature.
 
Why it matters:
    For fixed CT decay a_n < 0, the step size Δ controls how that decay maps into
    per-token forgetting:
    - small Δ -> a_n = e ^ Δ a_n closer to 1 -> longer memory / less forgetting
    - large Δ -> a_n smaller -> faster forgetting
    Thus, learned per-channel dt allows different channels to operate at different
    effective DT memory scales.
 
    With a shared fixed dt, every feature is forced to operate at the same
    timescale. With a per-channel learned log_dt, the model can discover these
    different rates through training.
 
    This concept carries directly into Mamba (S6), where dt goes one step
    further and becomes input-dependent — computed as a function of x at each
    timestep, allowing the model to decide dynamically how much to integrate.
 
Log-space for dt:
    We store log_dt rather than dt for the same reason we store log_A:
    unconstrained optimisation over ℝ, with the constraint dt > 0 enforced
    automatically by exp().
 
Initialisation:
    Sample log_dt uniformly in [log(dt_min), log(dt_max)]:
 
        log_dt ~ U(log(0.001), log(0.1))
 
    This gives dt values spread evenly across a 100× range on a log scale —
    the model starts with a diverse set of timescales rather than all features
    at the same rate.
 
    ┌────────────────────────────────────────────────────────────┐
    │  log_dt_min = log(0.001) ≈ -6.9   → dt ≈ 0.001 (slow)   │
    │  log_dt_max = log(0.100) ≈ -2.3   → dt ≈ 0.100 (fast)   │
    │  uniform sample → diverse mix across this range           │
    └────────────────────────────────────────────────────────────┘
 
Change 2 — HiPPO initialisation of A  (S4D-Real)
-------------------------------------------------
VanillaSSM:  log_A = zeros → all a[n] = -1.0 at init. Every state dimension
             starts with the same timescale. Training must discover diversity.
 
S4D:         log_A[n] = log(n+1)  →  a[n] = -(n+1)
 
             n=0:   a = -1     ā = exp(-Δ·1)    slow decay
             n=1:   a = -2     ā = exp(-Δ·2)
             n=2:   a = -3     ā = exp(-Δ·3)
             ...
             n=N-1: a = -N     ā = exp(-Δ·N)    fast decay
 
    Each state dimension is pre-assigned a different timescale. Together they
    form a spectrum from slow (long-range memory) to fast (short-range memory).
    The model starts with useful structure rather than having to learn it.
 
    ┌─────────────────────────────────────────────────────────────────────────┐
    │  Where does log(n+1) come from?                                        │
    │                                                                        │
    │  The HiPPO matrix (Gu et al., 2020) is derived by asking: what is the  │
    │  optimal A matrix for maintaining a polynomial approximation of the    │
    │  input history? The answer for Legendre polynomials (HiPPO-LegS) is a  │
    │  specific structured matrix. S4 uses a low-rank decomposition of this  │
    │  matrix (DPLR). S4D replaces the full DPLR structure with a diagonal   │
    │  approximation of just the eigenvalues, which are approximately        │
    │  -1, -2, -3, ..., -N. The S4D-Real variant uses this directly.        │
    │                                                                        │
    │  This is what lets S4D handle long-range dependencies well from the    │
    │  start of training — the eigenvalue spectrum is already meaningful.    │
    └─────────────────────────────────────────────────────────────────────────┘
 
===================================================================================
Discretisation with per-channel dt
===================================================================================
 
The ZOH formulas are unchanged:
 
    Ā = exp(Δ·A)
    B̄ = (Ā - I) / A · B
 
The only difference is Δ is now a vector (d_model,) rather than a scalar.
 
Broadcasting:
    A:    (d_model, N)
    dt:   (d_model,)  →  dt.unsqueeze(-1)  →  (d_model, 1)
 
    Δ·A:  (d_model, 1) × (d_model, N)  →  (d_model, N)   ✓
 
Everything downstream (kernel, FFT conv, recurrent scan) is identical to
VanillaSSM because A_bar and B_bar still have shape (d_model, N).
"""

import math
import torch
import torch.nn as nn
from models.ssm import VanillaSSM

class S4D(VanillaSSM):
    """
    Diagonal State Space Model.

    Inherits from VanillaSSM and overrides only __init__ and discretise().
    The convolutional and recurrent forward passes are unchanged.
 
    Changes from VanillaSSM:
        1. dt -> log_dt: nn.Parameter(d_model,)   [learned per-channel step size]
        2. log_A init:  log(n+1) for n=0..N-1    [HiPPO-inspired timescale spread]
    """
    def __init__(
        self,
        d_model: int,
        state_dim: int,
        dt_min: float = 0.001,
        dt_max: float = 0.1
    ) -> None:
        """
        d_model - no. features (each gets its own SSM)
        state_dim - hidden state size N
        dt_min - lower bound of initial dt range
        dt_max - upper bound of initial dt range
        """
        super().__init__(d_model=d_model, state_dim=state_dim, dt=0.1)
        # change 1: replace fixed dt scaler with learned log_dt
        del self.dt
        log_dt = torch.rand(d_model) * (math.log(dt_max) - math.log(dt_min)) \
                + math.log(dt_min)
        self.log_dt = nn.Parameter(log_dt)

        # change 2: replace zeros init with HiPPO diagonal init
        n = torch.arange(state_dim, dtype=torch.float)
        log_A_init = torch.log(n + 1)

        log_A_init = log_A_init.unsqueeze(0).expand(d_model, -1).clone()

        self.log_A = nn.Parameter(log_A_init)

    def discretise(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        ZOH discretisation with per-channel learned dt.
        """
        A = -torch.exp(self.log_A)
        dt = torch.exp(self.log_dt)

        A_bar = torch.exp(dt.unsqueeze(-1) * A)
        B_bar = ((A_bar - 1) / A) * self.B_param

        return A_bar, B_bar