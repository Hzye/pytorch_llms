"""
Selective SSM / S6

Improve LTI aspects of SSMs.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SelectiveSSMCore(nn.Module):
    """
    Minimal real-valued selective SSM (S6-style core).

    Input:
        u: (batch, L, d_model)

    Output:
        y: (batch, L, d_model)

    State:
        h: (batch, d_model, state_dim)

    Global parameters:
        log_A   : (d_model, state_dim)   -> A = -exp(log_A)
        D_skip  : (d_model,)

    Token-dependent parameters (generated from u_t):
        dt_t    : (batch, d_model)
        B_t     : (batch, d_model, state_dim)
        C_t     : (batch, d_model, state_dim)

    """
    def __init__(
        self,
        d_model: int,
        state_dim: int,
        dt_min: float = 1e-3,
        dt_max: float = 1e-1
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.state_dim = state_dim
        
        # ------------------------------------------------------------------
        # 1) global stable diagonal CT dynamics A
        # ------------------------------------------------------------------
        # s4d real style init - A[d, n] = -(n+1)
        n = torch.arange(1, state_dim + 1, dtype=torch.float32)
        log_A_init = torch.log(n).unsqueeze(0).expand(d_model, -1).clone()
        self.log_A = nn.Parameter(log_A_init) # A = -exp(log_A)

        # skip connection
        self.D_skip = nn.Parameter(torch.ones(d_model))

        # ------------------------------------------------------------------
        # 2) input-dependent parameter generators
        # ------------------------------------------------------------------
        # dt_t: (batch, d_model)
        # per channel step size
        self.dt_proj = nn.Linear(d_model, d_model)

        # B_t: (batch, d_model, state_dim)
        # per token write vector
        self.B_proj = nn.Linear(d_model, d_model * state_dim)

        # C_t: (batch, d_model, state_dim)
        # per token read vector
        self.C_proj = nn.Linear(d_model, d_model * state_dim)

        # bias so dt starts in sensible pos range
        log_dt_init = torch.rand(d_model) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        self.dt_bias = nn.Parameter(log_dt_init)


    def get_A(self) -> torch.Tensor:
        """
        Recover stable CT diagonal A.

        Returns:
            A: (d_model, state_dim), with A < 0 elementwise
        """
        return -torch.exp(self.log_A)
    

    def project_step(self, u_t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:        
        """
        Generate selective parameters from the current token representation.

        Args:
            u_t: (batch, d_model)

        Returns:
            dt_t: (batch, d_model), strictly positive via softplus
            B_t : (batch, d_model, state_dim)
            C_t : (batch, d_model, state_dim)
        """
        batch, d_model = u_t.shape
        assert d_model == self.d_model

        dt_t = F.softplus(self.dt_proj(u_t) + self.dt_bias) # (batch, d_model)

        B_t = self.B_proj(u_t).view(batch, d_model, self.state_dim)
        C_t = self.C_proj(u_t).view(batch, d_model, self.state_dim)

        return dt_t, B_t, C_t
    

    def discretise_step(
        self,
        dt_t: torch.Tensor,
        A: torch.Tensor,
        B_t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Exact per-step discretisation for the selective SSM.

        Args:
            A   : (d_model, state_dim), global continuous-time diagonal A
            dt_t: (batch, d_model), positive step size for current token
            B_t : (batch, d_model, state_dim), token-dependent write vector

        Returns:
            A_bar_t: (batch, d_model, state_dim)
            B_bar_t: (batch, d_model, state_dim)
        """
        A_expanded = A.unsqueeze(0) # (1, d_model, state_dim)
        dt_expanded = dt_t.unsqueeze(-1) # (batch, d_model, 1)

        A_bar_t = torch.exp(dt_expanded * A_expanded) # (batch, d_model, state_dim)
        B_bar_t = ((A_bar_t - 1.0) / A_expanded) * B_t
    
        return A_bar_t, B_bar_t
    
    def step(
        self,
        u_t: torch.Tensor,
        h: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One selective SSM step.

        Args:
            u_t : (batch, d_model)
            h   : (batch, d_model, state_dim)

        Returns:
            y_t : (batch, d_model)
            h   : updated hidden state, (batch, d_model, state_dim)
        """
        A = self.get_A()

        # selective params
        dt_t, B_t, C_t = self.project_step(u_t)
        
        # discretise
        A_bar_t, B_bar_t = self.discretise_step(dt_t, A, B_t)

        # update
        # u_t is scalar per channel, broadcast across state dims
        h = A_bar_t * h + B_bar_t * u_t.unsqueeze(-1)

        y_t = (C_t * h).sum(dim=-1) + self.D_skip.unsqueeze(0) * u_t

        return y_t, h
    

    def forward(self, u: torch.Tensor) -> torch.Tensor:

        """
        Selective recurrent scan over a sequence.

        Args:
            u: (batch, L, d_model)

        Returns:
            y: (batch, L, d_model)
        """
        batch, L, d_model = u.shape
        assert d_model == self.d_model

        h = torch.zeros(
            batch,
            d_model,
            self.state_dim,
            device=u.device,
            dtype=u.dtype
        )

        outputs = []
        for t in range(L):
            u_t = u[:, t, :]
            y_t, h = self.step(u_t, h)
            outputs.append(y_t)

        y = torch.stack(outputs, dim=1)
        return y


class S6Block(nn.Module):
    def __init__(self, d_model: int, state_dim: int):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.ssm = SelectiveSSMCore(d_model, state_dim)

    def forward(self, x):
        return x + self.ssm(self.layer_norm(x))


class TinyS6LM(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, state_dim: int, n_layers: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            S6Block(d_model, state_dim) for _ in range(n_layers)
        ])
        self.layer_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # weight tying
        self.lm_head.weight = self.embedding.weight


    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        tokens: (batch, L) integer token ids
        returns: logits (batch, L, vocab_size)
        """
        x = self.embedding(tokens) # (B, L, d_model)
        for layer in self.layers:
            x = layer(x) # (B, L, d_model)

        x = self.layer_norm(x)
        logits = self.lm_head(x) # (B, L, vocab_size)
        return logits
    

    def loss(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Next-token prediction loss.
        toeksn: (batch, L)
        """
        x = tokens[:, :-1]
        targets = tokens[:, 1:]

        logits = self.forward(x)

        loss = F.cross_entropy(
            logits.reshape(-1, self.vocab_size),
            targets.reshape(-1)
        )

        return loss