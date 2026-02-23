"""
STP (Short-Term Plasticity) Mixers for the Zoology Framework
=============================================================

Biologically-inspired short-term synaptic plasticity applied to
transformer sequence mixing.

SEQUENCE MIXERS (drop-in replacements for attention):
  Approach A: STPAttention      - Pure STP recurrent attention
  Approach B: HybridSTPAttention - Softmax attention + parallel STP with learned gate

Hardware mapping (FeFET devices):
  W_static     -> FeFET polarization (non-volatile, trained via backprop)
  S(t)         -> NQS channel charge (volatile, evolves via plasticity)
  Lambda       -> Overlap capacitance (controls charge decay rate)
  Gamma        -> Coupling strength (controls update magnitude)

Memory optimization:
  Uses gradient checkpointing on the recurrent loop to avoid storing
  all intermediate states. Sequences are processed in chunks of size
  CHUNK_SIZE; only chunk boundaries are stored for backward pass.
  This reduces memory from O(L * d^2) to O(CHUNK_SIZE * d^2).

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
from torch.utils.checkpoint import checkpoint
from einops import rearrange


# Default chunk size for gradient checkpointing.
# Smaller = less memory, more recomputation.
# 64 is a good balance for seq_len up to 1024.
DEFAULT_CHUNK_SIZE = 64


def _stp_recurrence_chunk(q_chunk, k_chunk, v_chunk, state, W_static, retention, gamma):
    """
    Process a chunk of timesteps through the STP recurrence.
    
    Args:
        q_chunk: (B, H, C, d) — queries for this chunk
        k_chunk: (B, H, C, d) — keys for this chunk
        v_chunk: (B, H, C, d) — values for this chunk
        state:   (B, H, d, d) — recurrent state at chunk start
        W_static: (H, d, d)   — static weight matrix
        retention: (H, d, d)  — 1 - Lambda (decay complement)
        gamma:   (H, d, d)    — Hebbian learning rate
    
    Returns:
        outputs: (B, H, C, d) — output for this chunk
        state:   (B, H, d, d) — recurrent state at chunk end
    """
    B, H, C, d = q_chunk.shape
    outputs = []

    for t in range(C):
        k_t = k_chunk[:, :, t, :]  # (B, H, d)
        v_t = v_chunk[:, :, t, :]  # (B, H, d)
        q_t = q_chunk[:, :, t, :]  # (B, H, d)

        # Hebbian outer product: v_t (x) k_t
        hebbian = torch.einsum("bhi,bhj->bhij", v_t, k_t)  # (B, H, d, d)

        # State update: decay + learn
        state = retention.unsqueeze(0) * state + gamma.unsqueeze(0) * hebbian

        # Output: (W_static + S(t)) @ q_t
        effective_W = W_static.unsqueeze(0) + state
        y_t = torch.einsum("bhij,bhj->bhi", effective_W, q_t)  # (B, H, d)
        outputs.append(y_t)

    outputs = torch.stack(outputs, dim=2)  # (B, H, C, d)
    return outputs, state


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

    Uses gradient checkpointing to handle long sequences without OOM.

    Usage in Zoology config:
        sequence_mixer=ModuleConfig(
            name="zoology.mixers.stp.STPAttention",
            kwargs={"num_heads": 2}
        )
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int = 2,
        bias: bool = True,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        layer_idx: int = None,
        **kwargs,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.chunk_size = chunk_size
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

        # STP parameters
        retention = 1.0 - self.Lambda  # (H, d, d)
        gamma = self.Gamma  # (H, d, d)

        # Process in chunks with gradient checkpointing
        state = torch.zeros(B, H, d, d, device=x.device, dtype=x.dtype)
        all_outputs = []

        for start in range(0, L, self.chunk_size):
            end = min(start + self.chunk_size, L)
            q_chunk = q[:, :, start:end, :]
            k_chunk = k[:, :, start:end, :]
            v_chunk = v[:, :, start:end, :]

            if self.training:
                # Gradient checkpointing: recompute forward during backward
                chunk_out, state = checkpoint(
                    _stp_recurrence_chunk,
                    q_chunk, k_chunk, v_chunk, state,
                    self.W_static, retention, gamma,
                    use_reentrant=False,
                )
            else:
                chunk_out, state = _stp_recurrence_chunk(
                    q_chunk, k_chunk, v_chunk, state,
                    self.W_static, retention, gamma,
                )

            all_outputs.append(chunk_out)

        y = torch.cat(all_outputs, dim=2)  # (B, H, L, d)
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

    Uses gradient checkpointing for the STP branch.

    Usage in Zoology config:
        sequence_mixer=ModuleConfig(
            name="zoology.mixers.stp.HybridSTPAttention",
            kwargs={"num_heads": 2}
        )
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int = 2,
        bias: bool = True,
        dropout: float = 0.0,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        layer_idx: int = None,
        **kwargs,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.chunk_size = chunk_size
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

        # === STP branch (recurrent with checkpointing) ===
        k_stp = k * self.scale  # normalized keys for STP
        retention = 1.0 - self.Lambda
        gamma = self.Gamma

        state = torch.zeros(B, H, d, d, device=x.device, dtype=x.dtype)
        stp_outputs = []

        for start in range(0, L, self.chunk_size):
            end = min(start + self.chunk_size, L)
            q_chunk = q[:, :, start:end, :]
            k_chunk = k_stp[:, :, start:end, :]
            v_chunk = v[:, :, start:end, :]

            if self.training:
                chunk_out, state = checkpoint(
                    _stp_recurrence_chunk,
                    q_chunk, k_chunk, v_chunk, state,
                    self.W_static, retention, gamma,
                    use_reentrant=False,
                )
            else:
                chunk_out, state = _stp_recurrence_chunk(
                    q_chunk, k_chunk, v_chunk, state,
                    self.W_static, retention, gamma,
                )

            stp_outputs.append(chunk_out)

        y_stp = torch.cat(stp_outputs, dim=2)  # (B, H, L, d)

        # === Learned gate ===
        alpha = self.gate_proj(x)  # (B, L, H)
        alpha = rearrange(alpha, "b l h -> b h l")  # (B, H, L)
        alpha = alpha.unsqueeze(-1)  # (B, H, L, 1)

        y = alpha * y_softmax + (1.0 - alpha) * y_stp  # (B, H, L, d)

        y = rearrange(y, "b h l d -> b l (h d)")
        y = self.out_proj(y)
        return y

    def state_size(self, sequence_length: int = 2048):
        return (
            2 * self.d_model * sequence_length
            + self.num_heads * self.head_dim * self.head_dim
        )


# ==============================================================================
# Approach C: STP-MLP (State Mixer) — for future experiments
# ==============================================================================

class STPMLP(nn.Module):
    """
    MLP with short-term plasticity in the first linear layer.

    The first layer's weight matrix evolves via STP:
        G(t) = W1 + F(t)
        F(t) = (1 - Lambda_F) * F(t-1) + Gamma_F * outer(h_prev, x_t)
        h = GELU(G(t) @ x_t + b1) @ W2 + b2

    Uses gradient checkpointing to handle long sequences.

    NOT used in the current MQAR experiment (original benchmark has no MLP).
    Reserved for future experiments where MLP is part of the architecture.

    Usage in Zoology config:
        state_mixer=ModuleConfig(
            name="zoology.mixers.stp.STPMLP",
            kwargs={"hidden_mult": 2}
        )
    """

    def __init__(
        self,
        d_model: int,
        hidden_mult: int = 2,
        activation: str = "gelu",
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        layer_idx: int = None,
        **kwargs,
    ):
        super().__init__()
        self.d_model = d_model
        self.chunk_size = chunk_size
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

    def _forward_chunk(self, x_chunk, F_state):
        """Process a chunk of timesteps."""
        B, C, D = x_chunk.shape
        lam = self.Lambda_F
        retention = 1.0 - lam
        gamma = self.Gamma_F
        W1 = self.fc1.weight
        b1 = self.fc1.bias

        outputs = []
        for t in range(C):
            x_t = x_chunk[:, t, :]
            G = W1.unsqueeze(0) + F_state
            h = torch.einsum("bij,bj->bi", G, x_t)
            if b1 is not None:
                h = h + b1
            h_pre = h
            h = self.activation(h)
            out = self.fc2(h)
            outputs.append(out)
            hebbian = torch.einsum("bi,bj->bij", h_pre, x_t)
            F_state = retention.unsqueeze(0) * F_state + gamma.unsqueeze(0) * hebbian

        return torch.stack(outputs, dim=1), F_state

    def forward(self, x: torch.Tensor, **kwargs):
        B, L, D = x.shape
        F_state = torch.zeros(B, self.fc1.weight.shape[0], self.fc1.weight.shape[1],
                              device=x.device, dtype=x.dtype)

        all_outputs = []
        for start in range(0, L, self.chunk_size):
            end = min(start + self.chunk_size, L)
            x_chunk = x[:, start:end, :]

            if self.training:
                chunk_out, F_state = checkpoint(
                    self._forward_chunk, x_chunk, F_state,
                    use_reentrant=False,
                )
            else:
                chunk_out, F_state = self._forward_chunk(x_chunk, F_state)

            all_outputs.append(chunk_out)

        return torch.cat(all_outputs, dim=1)
