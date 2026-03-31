"""
Vanilla State Space Model (SSM)
 
References:
    [1] "Efficiently Modeling Long Sequences with Structured State Spaces" (Gu et al., 2021)
        https://arxiv.org/abs/2111.00396
    [2] "Mamba: Linear-Time Sequence Modeling with Selective State Spaces" (Gu & Dao, 2023)
        https://arxiv.org/abs/2312.00752
    [3] "On the Parameterization and Initialization of Diagonal State Space Models" (Gu et al., 2022)
        https://arxiv.org/abs/2206.11893
 
===================================================================================
Continuous-time State Space Model (from control theory)
===================================================================================
 
The continuous SSM is defined by two coupled differential equations:
 
    h'(t) = A·h(t) + B·x(t)      [state equation]
    y(t)  = C·h(t) + D·x(t)      [output equation]
 
Where:
    x(t) ∈ ℝ         — scalar input signal at time t
    h(t) ∈ ℝᴺ        — hidden state (N-dimensional compressed memory)
    y(t) ∈ ℝ         — scalar output at time t
    A    ∈ ℝᴺˣᴺ      — state transition matrix (governs memory dynamics)
    B    ∈ ℝᴺˣ¹      — input projection (how input writes into state)
    C    ∈ ℝ¹ˣᴺ      — output projection (how state is read)
    D    ∈ ℝ          — skip connection (direct input → output)
 
The intuition:
    - h(t) is a compressed memory of everything seen so far
    - A determines what the model remembers and what it forgets
    - B writes new information into the state
    - C reads relevant information out of the state to produce the output

===================================================================================
Why A must have negative eigenvalues (stability)
===================================================================================
 
For the state equation h'(t) = A·h(t) to be stable, A must have eigenvalues
with strictly negative real parts. For diagonal A (each element a[n]):
 
    Solution: h[n](t) = exp(a[n]·t) · h[n](0)
 
    If a[n] > 0: h[n] grows exponentially   → unstable (state explodes)
    If a[n] < 0: h[n] decays exponentially  → stable   (state forgets gradually)
    If a[n] = 0: h[n] is constant           → neutral  (perfect memory, no decay)
 
We use log-space parameterisation:
    Store: log_A = log(-A)
    Recover: A = -exp(log_A) < 0   always, by construction
 
This means gradient updates can roam freely in ℝ without ever producing an
unstable A. This is a key insight from S4D [3] and carries forward to Mamba.

===================================================================================
Discretisation via Zero-Order Hold (ZOH)
===================================================================================
 
Computers operate in discrete steps. We convert the continuous SSM to a
discrete recurrence using ZOH discretisation with step size Δ (delta).
 
ZOH assumption: the input x is held constant between consecutive time steps.
This is the natural model for token sequences — there is no information between
tokens, so it is correct to treat each token's value as constant until the next.
 
ZOH formulas:
    Ā = exp(Δ·A)                           [matrix exponential]
    B̄ = A⁻¹·(Ā - I)·B                     [exact ZOH formula]
 
Derivation sketch:
    Solve h'(t) = A·h(t) + B·x (with x constant) over interval [0, Δ]:
    h(Δ) = exp(A·Δ)·h(0) + A⁻¹·(exp(A·Δ) - I)·B·x
         = Ā·h(0) + B̄·x
 
For DIAGONAL A (stored as vector a, shape (d_model, N)):
    ā[n] = exp(Δ·a[n])                     [element-wise, no matrix exp]
    b̄[n] = (ā[n] - 1) / a[n] · b[n]      [element-wise, no matrix inverse]
 
Key property: since a[n] < 0 and Δ > 0:
    ā[n] = exp(Δ·a[n]) ∈ (0, 1)   always
 
This is the discrete "forgetting factor" — larger |a[n]| means faster forgetting.
 
The resulting discrete recurrence:
    h[t] = Ā·h[t-1] + B̄·x[t]             [state update]
    y[t] = C·h[t]   + D·x[t]             [output]

===================================================================================
Convolutional view — the key insight enabling parallel training
===================================================================================
 
Unrolling the recurrence from h[-1] = 0:
 
    h[0] = B̄·x[0]
    h[1] = Ā·B̄·x[0] + B̄·x[1]
    h[2] = Ā²·B̄·x[0] + Ā·B̄·x[1] + B̄·x[2]
 
Multiplying each by C and adding D·x[t]:
 
    y[0] = C·B̄·x[0]                                        + D·x[0]
    y[1] = C·Ā·B̄·x[0]  + C·B̄·x[1]                       + D·x[1]
    y[2] = C·Ā²·B̄·x[0] + C·Ā·B̄·x[1] + C·B̄·x[2]        + D·x[2]
 
This is exactly a causal convolution y = K ∗ x + D·x, with the SSM kernel:
 
    K = [C·B̄,  C·Ā·B̄,  C·Ā²·B̄,  ...,  C·Ā^(L-1)·B̄]
    K[t] = C·Ā^t·B̄
 
For DIAGONAL A (element-wise products, then sum over state dimension N):
 
    K[d, t] = Σ_n  C[d,n] · ā[d,n]^t · B̄[d,n]
 
The convolution y = K ∗ x can be computed via FFT in O(L log L):
 
    conv(K, x) = IFFT( FFT(K) · FFT(x) )

===================================================================================
The dual mode: SSM's engineering win
===================================================================================
 
    Training  → convolutional mode → O(L log L), fully parallelisable over L
    Inference → recurrent mode     → O(1) per step, constant memory
 
Both modes compute the EXACT SAME output. This is verified below.
 
Compare to transformers:
    Training  → O(L²) attention (parallel)
    Inference → O(L)  per step (KV cache grows with sequence length)
 
The SSM's recurrent mode uses fixed memory regardless of sequence length,
making it dramatically more efficient for very long sequences at inference.
"""

import torch
import torch.nn as nn


class VanillaSSM(nn.Module):
    """
    Vanilla State Space Model with diagonal A.

    d_model independent SSMs in parallel - one per input feature.
    
    Supports both recurrent (inference) and convolutional (training) forward passes.

    d_model     - no. of input/output features
    N           - state dim (hidden state size per feature)
    L           - seq length
    B           - batch size

    log_A       (d_model, N) : log(-A), so A = -exp(log_A) < 0
    B_param     (d_model, N) : input projection
    C           (d_model, N) : output projection
    D           (d_model,)   : skip connection
    """
    def __init__(self, d_model: int, state_dim: int, dt: float = 0.1) -> None:
        super().__init__()
        self.d_model = d_model
        self.state_dim = state_dim # N
        self.dt = dt # time step
        
        # log space parameterisation
        # init to zero
        self.log_A = nn.Parameter(torch.zeros(d_model, state_dim))
        
        # small random init
        self.B_param = nn.Parameter(torch.randn(d_model, state_dim) * 0.01)

        self.C = nn.Parameter(torch.randn(d_model, state_dim) * 0.01)

        self.D = nn.Parameter(torch.ones(d_model))


    def discretise(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        ZOH
        """
        # recover A from log space param
        A = -torch.exp(self.log_A)

        # A_bar = exp(time_step * A)
        A_bar = torch.exp(self.dt * A)

        # B_bar = (a_bar - 1) / a . b
        B_bar = ((A_bar - 1) / A) * self.B_param

        return A_bar, B_bar
    

    def compute_kernel(self, A_bar: torch.Tensor, B_bar: torch.Tensor, L: int) -> torch.Tensor:
        """
        Build SSM convolutional kernel K of length L
        """
        t = torch.arange(L, device=A_bar.device, dtype=A_bar.dtype) # (L,)

        powers = A_bar.unsqueeze(-1) ** t

        CB = self.C * B_bar

        #K = torch.einsum('dn, dnl->dl', CB, powers)
        K = (CB.unsqueeze(-1) * powers).sum(dim=1)

        return K
    

    def forward_convolution(self, x: torch.Tensor, A_bar: torch.Tensor, B_bar: torch.Tensor) -> torch.Tensor:
        """
        Training mode.
        
        Parallel computation via FFT-based convolution.
        """
        batch, L, d_model = x.shape

        # build kernel
        K = self.compute_kernel(A_bar, B_bar, L)

        # transpose x - time dim moves to last for rfft
        x_t = x.transpose(1, 2)

        ## fft conv
        # pad to 2L to prevent circular aliasing
        fft_size = 2 * L

        # rfft: real-valued fft
        K_fft = torch.fft.rfft(K, n=fft_size, dim=-1)
        x_fft = torch.fft.rfft(x_t, n=fft_size, dim=-1)

        y_fft = K_fft.unsqueeze(0) * x_fft

        y = torch.fft.irfft(y_fft, n=fft_size, dim=-1)
        y = y[:, :, :L]

        y = y + self.D.unsqueeze(-1) * x_t

        return y.transpose(1, 2)
    

    def forward_recurrent(self, x: torch.Tensor, A_bar: torch.Tensor, B_bar: torch.Tensor) -> torch.Tensor:
        """
        Inference mode.

        Sequential recurrence.
        """
        batch, L, d_model = x.shape

        # init hidden state
        h = torch.zeros(batch, d_model, self.state_dim, device=x.device, dtype=x.dtype)

        outputs = []
        for t in range(L):
            x_t = x[:, t, :] # (batch, d_model)

            # state update
            h = A_bar * h + B_bar * x_t.unsqueeze(-1) # (batch, d_model, N)

            # output 
            y_t = (self.C * h).sum(dim=-1) + self.D * x_t # (batch, d_model)

            outputs.append(y_t)

        return torch.stack(outputs, dim=-1)
    
    
    def forward(self, x: torch.Tensor, mode: str = "convolutional") -> torch.Tensor:
        """
        Forward pass.

        x:      (batch, L, d_model)
        mode:   "convolutional" for training.   O(L log L) 
                "recurrent" for inference.      O(1)

        Both modes should produce identical outputs.
        """
        assert mode in ("convolutional", "recurrent"), \
            f"mode must be 'convolutional' or 'recurrent', got {mode}"
        
        A_bar, B_bar = self.discretise()

        if mode == "recurrent":
            return self.forward_recurrent(x, A_bar, B_bar)
        else:
            return self.forward_convolution(x, A_bar, B_bar)