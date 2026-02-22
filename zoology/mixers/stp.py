"""
STP (Short-Term Plasticity) Mixers for the Zoology Framework
=============================================================

Biologically-inspired short-term synaptic plasticity applied to
transformer sequence mixing and state mixing (MLP).

This file implements ALL STP approaches as drop-in replacements
for Zoology's existing mixers:

SEQUENCE MIXERS (replace attention):
  Approach A: STPAttention      - Pure STP recurrent attention
  Approach B: HybridSTPAttention - Softmax attention + parallel STP with learned gate

STATE MIXERS (replace MLP):
  Approach C: STPMLP            - MLP with plastic first-layer weights

Each can be used independently or combined. For example:
  - A only:     sequence_mixer=STPAttention, state_mixer=MLP
  - B only:     sequence_mixer=HybridSTPAttention, state_mixer=MLP
  - C only:     sequence_mixer=MHA, state_mixer=STPMLP
  - A+C:        sequence_mixer=STPAttention, state_mixer=STPMLP
  - B+C:        sequence_mixer=HybridSTPAttention, state_mixer=STPMLP

Hardware mapping (FeFET devices):
  W_static     -> FeFET polarization (non-volatile, trained via backprop)
  F(t)         -> NQS channel charge (volatile, evolves via plasticity)
  Lambda       -> Overlap capacitance (controls charge decay rate)
  Gamma        -> Coupling strength (controls update magnitude)

References:
  Rodriguez et al., "Short-Term Plasticity Networks" (2024)
  Arora et al., "Zoology: Measuring and Improving Recall" (ICLR 2024)

Author: RISE Lab, Purdue University
Date: February 2026
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# ==============================================================================
# Approach A: Pure STP Attention (Sequence Mixer)
# ==============================================================================

class STPAttention(nn.Module):
    """
    Pure STP recurrent attention. Replaces softmax attention entirely.

    The recurrent state S(t) evolves as:
        S(t) = (1 - Lambda) * S(t-1) + Gamma * v_t @ k_t^T
    Output:
        y(t) = (W_static + S(t)) @ q_t

    Lambda and Gamma are per-element learned parameters, enabling the model
    to selectively retain or forget different associations at different rates.

    This is a linear-time recurrent model (no quadratic attention matrix).

    Usage in Zoology config:
        sequence_mixer=ModuleConfig(
            name="zoology.mixers.stp.STPAttention",
            kwargs={"num_heads": 1}
        )
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int = 1,
        bias: bool = True,
        layer_idx: int = None,
        **kwargs,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        # QKV projections
        self.Wqkv = nn.Linear(d_model, 3 * d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)

        # Per-head STP parameters
        # W_static: static weight matrix (init to zero to force use of S(t))
        self.W_static = nn.Parameter(
            torch.zeros(num_heads, self.head_dim, self.head_dim)
        )

        # Lambda_raw: controls decay rate per element, sigmoid maps to (0,1)
        self.Lambda_raw = nn.Parameter(
            torch.zeros(num_heads, self.head_dim, self.head_dim)
        )
        nn.init.uniform_(self.Lambda_raw, -3.0, 3.0)

        # Gamma: Hebbian learning rate per element
        self.Gamma = nn.Parameter(
            torch.zeros(num_heads, self.head_dim, self.head_dim)
        )
        nn.init.uniform_(
            self.Gamma,
            -0.5 / math.sqrt(self.head_dim),
            0.5 / math.sqrt(self.head_dim),
        )

        # Key normalization scale
        self.scale = 1.0 / math.sqrt(self.head_dim)

    @property
    def Lambda(self):
        return torch.sigmoid(self.Lambda_raw)

    def forward(self, x: torch.Tensor, **kwargs):
        """
        x: (B, L, D)
        Returns: (B, L, D)
        """
        B, L, D = x.shape
        H = self.num_heads
        d = self.head_dim

        # Project to Q, K, V
        qkv = self.Wqkv(x)
        qkv = rearrange(qkv, "b l (three h d) -> three b h l d", three=3, h=H)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each (B, H, L, d)

        # Normalize keys
        k = k * self.scale

        # STP recurrence
        lam = self.Lambda  # (H, d, d)
        retention = 1.0 - lam
        gamma = self.Gamma  # (H, d, d)

        # State: (B, H, d, d)
        state = torch.zeros(B, H, d, d, device=x.device, dtype=x.dtype)

        outputs = []
        for t in range(L):
            k_t = k[:, :, t, :]  # (B, H, d)
            v_t = v[:, :, t, :]  # (B, H, d)
            q_t = q[:, :, t, :]  # (B, H, d)

            # Hebbian outer product: v_t (x) k_t
            hebbian = torch.einsum("bhi,bhj->bhij", v_t, k_t)  # (B, H, d, d)

            # State update: decay + learn
            state = retention.unsqueeze(0) * state + gamma.unsqueeze(0) * hebbian

            # Output: (W_static + S(t)) @ q_t
            effective_W = self.W_static.unsqueeze(0) + state  # (B, H, d, d)
            y_t = torch.einsum("bhij,bhj->bhi", effective_W, q_t)  # (B, H, d)
            outputs.append(y_t)

        y = torch.stack(outputs, dim=2)  # (B, H, L, d)
        y = rearrange(y, "b h l d -> b l (h d)")
        y = self.out_proj(y)
        return y

    def state_size(self, sequence_length: int = 2048):
        return self.num_heads * self.head_dim * self.head_dim


# ==============================================================================
# Approach B: Hybrid STP + Softmax Attention (Sequence Mixer)
# ==============================================================================

class HybridSTPAttention(nn.Module):
    """
    Parallel softmax attention + STP branch with learned gating.

    y(t) = alpha(t) * y_softmax(t) + (1 - alpha(t)) * y_stp(t)

    The gate alpha is learned per-position, allowing the model to decide
    WHEN to use precise recall (softmax) vs adaptive memory (STP).

    This is the most practical approach for augmenting existing transformers:
    add an STP branch and learn when to use it.

    Usage in Zoology config:
        sequence_mixer=ModuleConfig(
            name="zoology.mixers.stp.HybridSTPAttention",
            kwargs={"num_heads": 1}
        )
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int = 1,
        bias: bool = True,
        dropout: float = 0.0,
        layer_idx: int = None,
        **kwargs,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        assert d_model % num_heads == 0

        # Shared QKV projections
        self.Wqkv = nn.Linear(d_model, 3 * d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)

        # Softmax attention dropout
        self.dropout_p = dropout

        # STP parameters (per-head)
        self.W_static = nn.Parameter(
            torch.zeros(num_heads, self.head_dim, self.head_dim)
        )
        self.Lambda_raw = nn.Parameter(
            torch.zeros(num_heads, self.head_dim, self.head_dim)
        )
        nn.init.uniform_(self.Lambda_raw, -3.0, 3.0)
        self.Gamma = nn.Parameter(
            torch.zeros(num_heads, self.head_dim, self.head_dim)
        )
        nn.init.uniform_(
            self.Gamma,
            -0.5 / math.sqrt(self.head_dim),
            0.5 / math.sqrt(self.head_dim),
        )
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # Learned gate: per-position scalar deciding softmax vs STP
        self.gate_proj = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, num_heads),
            nn.Sigmoid(),
        )

    @property
    def Lambda(self):
        return torch.sigmoid(self.Lambda_raw)

    def forward(self, x: torch.Tensor, **kwargs):
        """
        x: (B, L, D)
        Returns: (B, L, D)
        """
        B, L, D = x.shape
        H = self.num_heads
        d = self.head_dim

        # Project to Q, K, V
        qkv = self.Wqkv(x)
        qkv = rearrange(qkv, "b l (three h d) -> three b h l d", three=3, h=H)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # === Softmax branch ===
        softmax_scale = 1.0 / math.sqrt(d)
        scores = torch.einsum("bhid,bhjd->bhij", q, k) * softmax_scale
        causal_mask = torch.triu(
            torch.full((L, L), -10000.0, device=x.device), 1
        )
        scores = scores + causal_mask.to(dtype=scores.dtype)
        attn_weights = torch.softmax(scores, dim=-1, dtype=v.dtype)
        attn_weights = F.dropout(
            attn_weights, self.dropout_p if self.training else 0.0
        )
        y_softmax = torch.einsum("bhij,bhjd->bhid", attn_weights, v)  # (B,H,L,d)

        # === STP branch (recurrent) ===
        k_stp = k * self.scale  # normalized keys for STP
        lam = self.Lambda
        retention = 1.0 - lam
        gamma = self.Gamma

        state = torch.zeros(B, H, d, d, device=x.device, dtype=x.dtype)
        stp_outputs = []

        for t in range(L):
            k_t = k_stp[:, :, t, :]
            v_t = v[:, :, t, :]
            q_t = q[:, :, t, :]

            hebbian = torch.einsum("bhi,bhj->bhij", v_t, k_t)
            state = retention.unsqueeze(0) * state + gamma.unsqueeze(0) * hebbian

            effective_W = self.W_static.unsqueeze(0) + state
            y_t = torch.einsum("bhij,bhj->bhi", effective_W, q_t)
            stp_outputs.append(y_t)

        y_stp = torch.stack(stp_outputs, dim=2)  # (B, H, L, d)

        # === Learned gate ===
        alpha = self.gate_proj(x)  # (B, L, H)
        alpha = rearrange(alpha, "b l h -> b h l")  # (B, H, L)
        alpha = alpha.unsqueeze(-1)  # (B, H, L, 1)

        y = alpha * y_softmax + (1.0 - alpha) * y_stp  # (B, H, L, d)

        y = rearrange(y, "b h l d -> b l (h d)")
        y = self.out_proj(y)
        return y

    def state_size(self, sequence_length: int = 2048):
        # Softmax stores full KV cache; STP stores d*d state per head
        return (
            2 * self.d_model * sequence_length
            + self.num_heads * self.head_dim * self.head_dim
        )


# ==============================================================================
# Approach C: STP-MLP (State Mixer)
# ==============================================================================

class STPMLP(nn.Module):
    """
    MLP with short-term plasticity in the first linear layer.

    The first layer's weight matrix evolves via STP:
        G(t) = W1 + F(t)
        F(t) = (1 - Lambda_F) * F(t-1) + Gamma_F * outer(h_prev, x_t)
        h = GELU(G(t) @ x_t + b1) @ W2 + b2

    This makes the "knowledge" stored in the FFN context-adaptive.

    Hardware mapping:
        W1       -> FeFET polarization (non-volatile)
        F(t)     -> NQS channel charge (volatile)
        Lambda_F -> Overlap capacitance
        Gamma_F  -> Coupling strength

    Usage in Zoology config:
        state_mixer=ModuleConfig(
            name="zoology.mixers.stp.STPMLP",
            kwargs={"hidden_mult": 4}
        )
    """

    def __init__(
        self,
        d_model: int,
        hidden_mult: int = 4,
        activation: str = "gelu",
        layer_idx: int = None,
        **kwargs,
    ):
        super().__init__()
        self.d_model = d_model
        d_ff = d_model * hidden_mult

        # Standard MLP weights
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

        if activation == "gelu":
            self.activation = F.gelu
        elif activation == "silu":
            self.activation = F.silu
        else:
            self.activation = F.gelu

        # STP parameters for fc1
        self.Lambda_raw_F = nn.Parameter(torch.zeros(d_ff, d_model))
        nn.init.uniform_(self.Lambda_raw_F, -3.0, 3.0)

        self.Gamma_F = nn.Parameter(torch.zeros(d_ff, d_model))
        nn.init.uniform_(
            self.Gamma_F,
            -0.5 / math.sqrt(d_model),
            0.5 / math.sqrt(d_model),
        )

    @property
    def Lambda_F(self):
        return torch.sigmoid(self.Lambda_raw_F)

    def forward(self, x: torch.Tensor, **kwargs):
        """
        x: (B, L, D)
        Returns: (B, L, D)
        """
        B, L, D = x.shape

        lam = self.Lambda_F  # (d_ff, d_model)
        retention = 1.0 - lam
        gamma = self.Gamma_F  # (d_ff, d_model)
        W1 = self.fc1.weight  # (d_ff, d_model)
        b1 = self.fc1.bias  # (d_ff,)

        # Plastic state: (B, d_ff, d_model)
        F_state = torch.zeros(B, W1.shape[0], W1.shape[1],
                              device=x.device, dtype=x.dtype)

        outputs = []
        for t in range(L):
            x_t = x[:, t, :]  # (B, d_model)

            # Effective weight = static + plastic
            G = W1.unsqueeze(0) + F_state  # (B, d_ff, d_model)

            # Forward through plastic first layer
            h = torch.einsum("bij,bj->bi", G, x_t)  # (B, d_ff)
            if b1 is not None:
                h = h + b1

            h_pre = h  # Save pre-activation for Hebbian update
            h = self.activation(h)

            # Second layer (standard)
            out = self.fc2(h)  # (B, d_model)
            outputs.append(out)

            # Hebbian update for NEXT timestep
            hebbian = torch.einsum("bi,bj->bij", h_pre, x_t)  # (B, d_ff, d_model)
            F_state = retention.unsqueeze(0) * F_state + gamma.unsqueeze(0) * hebbian

        y = torch.stack(outputs, dim=1)  # (B, L, D)
        return y


# ==============================================================================
# Standard baselines for fair comparison
# ==============================================================================

class LinearAttention(nn.Module):
    """
    Simple linear attention baseline (no decay, no plasticity).
    S(t) = S(t-1) + v_t @ k_t^T  (pure accumulation)
    y(t) = S(t) @ q_t

    Usage in Zoology config:
        sequence_mixer=ModuleConfig(
            name="zoology.mixers.stp.LinearAttention",
            kwargs={"num_heads": 1}
        )
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int = 1,
        bias: bool = True,
        layer_idx: int = None,
        **kwargs,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        assert d_model % num_heads == 0

        self.Wqkv = nn.Linear(d_model, 3 * d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        self.scale = 1.0 / math.sqrt(self.head_dim)

    def forward(self, x: torch.Tensor, **kwargs):
        B, L, D = x.shape
        H = self.num_heads
        d = self.head_dim

        qkv = self.Wqkv(x)
        qkv = rearrange(qkv, "b l (three h d) -> three b h l d", three=3, h=H)
        q, k, v = qkv[0], qkv[1], qkv[2]

        k = k * self.scale
        state = torch.zeros(B, H, d, d, device=x.device, dtype=x.dtype)

        outputs = []
        for t in range(L):
            k_t = k[:, :, t, :]
            v_t = v[:, :, t, :]
            q_t = q[:, :, t, :]

            state = state + torch.einsum("bhi,bhj->bhij", v_t, k_t)
            y_t = torch.einsum("bhij,bhj->bhi", state, q_t)
            outputs.append(y_t)

        y = torch.stack(outputs, dim=2)
        y = rearrange(y, "b h l d -> b l (h d)")
        y = self.out_proj(y)
        return y

    def state_size(self, sequence_length: int = 2048):
        return self.num_heads * self.head_dim * self.head_dim


class RetNetAttention(nn.Module):
    """
    RetNet-style attention baseline (scalar decay, no per-element plasticity).
    S(t) = gamma * S(t-1) + v_t @ k_t^T
    y(t) = S(t) @ q_t

    Usage in Zoology config:
        sequence_mixer=ModuleConfig(
            name="zoology.mixers.stp.RetNetAttention",
            kwargs={"num_heads": 1}
        )
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int = 1,
        bias: bool = True,
        layer_idx: int = None,
        **kwargs,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        assert d_model % num_heads == 0

        self.Wqkv = nn.Linear(d_model, 3 * d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # One scalar decay per head (learned)
        self.gamma_raw = nn.Parameter(torch.zeros(num_heads))
        nn.init.uniform_(self.gamma_raw, 0.0, 2.0)

    @property
    def gamma(self):
        return torch.sigmoid(self.gamma_raw)

    def forward(self, x: torch.Tensor, **kwargs):
        B, L, D = x.shape
        H = self.num_heads
        d = self.head_dim

        qkv = self.Wqkv(x)
        qkv = rearrange(qkv, "b l (three h d) -> three b h l d", three=3, h=H)
        q, k, v = qkv[0], qkv[1], qkv[2]

        k = k * self.scale
        g = self.gamma  # (H,)
        state = torch.zeros(B, H, d, d, device=x.device, dtype=x.dtype)

        outputs = []
        for t in range(L):
            k_t = k[:, :, t, :]
            v_t = v[:, :, t, :]
            q_t = q[:, :, t, :]

            state = g.view(1, H, 1, 1) * state + torch.einsum(
                "bhi,bhj->bhij", v_t, k_t
            )
            y_t = torch.einsum("bhij,bhj->bhi", state, q_t)
            outputs.append(y_t)

        y = torch.stack(outputs, dim=2)
        y = rearrange(y, "b h l d -> b l (h d)")
        y = self.out_proj(y)
        return y

    def state_size(self, sequence_length: int = 2048):
        return self.num_heads * self.head_dim * self.head_dim
