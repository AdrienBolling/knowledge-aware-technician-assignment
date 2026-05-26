"""Modern Transformer encoder for token-ID observations.

Implements a stack of pre-norm decoder-style blocks with the
optimisations that have become standard in recent open-weights
language models (Llama / Mistral / Qwen).  Specifically:

* **RMSNorm** instead of LayerNorm (no mean, no learnable bias).
* **Rotary positional embeddings (RoPE)** applied per-head inside the
  attention module — additive, length-extrapolating.
* **SwiGLU** feed-forward (gated linear unit with SiLU activation),
  which consistently outperforms ReLU/GELU MLPs at the same parameter
  budget.
* **Pre-norm residual blocks** + **scaled residual init** for stable
  deep-stack training.
* **Causal=False** (encoder-only); a key padding mask blocks
  attention on `<PAD>` tokens.
* **CLS-token pooling** so the head sees a single learned summary
  embedding (rather than a mean over variable-length tokens).

Designed to be scriptable, mixed-precision friendly, and big enough
for a "complex and big" agent — ~700k–1M parameters at the default
config (d_model=192, n_layers=6, n_heads=6).
"""

from __future__ import annotations

import math

import torch
from torch import nn


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


class RMSNorm(nn.Module):
    """Root-Mean-Square layer norm without bias (Llama-style)."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.float().pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).to(x.dtype) * self.weight


class SwiGLU(nn.Module):
    """SwiGLU feed-forward: ``Swish(W_gate · x) ⊙ W_up · x → W_down``."""

    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.w_gate = nn.Linear(dim, hidden_dim, bias=False)
        self.w_up = nn.Linear(dim, hidden_dim, bias=False)
        self.w_down = nn.Linear(hidden_dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w_down(nn.functional.silu(self.w_gate(x)) * self.w_up(x)))


def _build_rope_cache(
    head_dim: int,
    max_seq_len: int,
    base: float = 10000.0,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pre-compute (cos, sin) tables for RoPE."""
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(max_seq_len, device=device).float()
    freqs = torch.outer(t, inv_freq)  # (S, head_dim/2)
    return freqs.cos(), freqs.sin()


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply rotary positional embedding to ``x`` of shape ``(B, H, S, D)``."""
    # x split into even/odd along the last dim
    x1, x2 = x[..., 0::2], x[..., 1::2]
    cos = cos[None, None, : x.size(-2), :]  # (1, 1, S, D/2)
    sin = sin[None, None, : x.size(-2), :]
    rotated_even = x1 * cos - x2 * sin
    rotated_odd = x1 * sin + x2 * cos
    out = torch.empty_like(x)
    out[..., 0::2] = rotated_even
    out[..., 1::2] = rotated_odd
    return out


class RoPESelfAttention(nn.Module):
    """Multi-head self-attention with rotary positional embeddings.

    Uses ``F.scaled_dot_product_attention`` so it benefits from the
    PyTorch fused / Flash kernel when available (CUDA, MPS), with a
    portable fallback on CPU.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        max_seq_len: int,
        dropout: float = 0.0,
        rope_base: float = 10000.0,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            msg = f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
            raise ValueError(msg)
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout_p = dropout

        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        cos, sin = _build_rope_cache(self.head_dim, max_seq_len, base=rope_base)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        b, s, _ = x.shape
        qkv = self.qkv_proj(x).view(b, s, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)  # each: (B, S, H, D_h)
        q = q.transpose(1, 2)  # (B, H, S, D_h)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        q = _apply_rope(q, self.rope_cos, self.rope_sin)
        k = _apply_rope(k, self.rope_cos, self.rope_sin)

        # SDPA mask: True means "blocked".  Convert padding mask
        # (B, S) -> additive (B, 1, 1, S) for broadcast across heads.
        attn_mask: torch.Tensor | None = None
        if key_padding_mask is not None:
            attn_mask = key_padding_mask[:, None, None, :].to(torch.bool)
            # SDPA expects a bool/float attn_mask where True = mask out
        out = nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=~attn_mask if attn_mask is not None else None,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=False,
        )
        out = out.transpose(1, 2).contiguous().view(b, s, self.d_model)
        return self.out_proj(out)


class TransformerBlock(nn.Module):
    """Pre-norm block: x ← x + Attn(RMS(x)); x ← x + FFN(RMS(x))."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        max_seq_len: int,
        dropout: float = 0.0,
        rope_base: float = 10000.0,
    ) -> None:
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = RoPESelfAttention(
            d_model,
            n_heads,
            max_seq_len=max_seq_len,
            dropout=dropout,
            rope_base=rope_base,
        )
        self.norm2 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, d_ff, dropout=dropout)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x + self.dropout(self.attn(self.norm1(x), key_padding_mask=key_padding_mask))
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------


class ModernTransformerEncoder(nn.Module):
    """Encoder-only transformer with RMSNorm + SwiGLU + RoPE + CLS pooling.

    Forward: ``(B, S)`` token-ID tensor → ``(B, d_model)`` summary.

    Parameters
    ----------
    vocab_size:
        Size of the token vocabulary.
    d_model:
        Width of the residual stream.
    n_heads:
        Number of attention heads (must divide ``d_model``).
    n_layers:
        Depth of the encoder stack.
    d_ff:
        SwiGLU hidden dimension; defaults to roughly ``2.667 * d_model``
        following Llama (which keeps total FFN params close to the
        original ``4 * d_model`` GeLU MLP budget).
    max_seq_len:
        Upper bound on the input sequence length.  RoPE tables are
        pre-computed up to this length.
    dropout:
        Dropout applied to attention output, FFN output, and embeddings.
    pad_token_id:
        Token id treated as padding (excluded from attention).
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 192,
        n_heads: int = 6,
        n_layers: int = 6,
        d_ff: int | None = None,
        max_seq_len: int = 256,
        dropout: float = 0.1,
        pad_token_id: int = 0,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.pad_token_id = pad_token_id

        if d_ff is None:
            # Llama-style: ~8/3 · d_model rounded to a multiple of 64
            d_ff = int(round((8 * d_model / 3) / 64.0)) * 64

        self.token_embedding = nn.Embedding(
            vocab_size, d_model, padding_idx=pad_token_id
        )
        self.embed_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Learnable CLS token prepended to every sequence
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    n_heads=n_heads,
                    d_ff=d_ff,
                    max_seq_len=max_seq_len + 1,  # +1 for the CLS slot
                    dropout=dropout,
                )
                for _ in range(n_layers)
            ]
        )
        self.final_norm = RMSNorm(d_model)
        self._init_weights()

    @property
    def output_dim(self) -> int:
        return self.d_model

    def _init_weights(self) -> None:
        # Llama-style scaled init: deeper layers get a smaller std on
        # the residual projections to keep activation variance bounded.
        std_base = 0.02
        for layer_idx, block in enumerate(self.blocks):
            depth_scale = 1.0 / math.sqrt(2 * (layer_idx + 1))
            for module in (block.attn.out_proj, block.ffn.w_down):
                nn.init.normal_(module.weight, mean=0.0, std=std_base * depth_scale)
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=std_base)
        nn.init.normal_(self.cls_token, mean=0.0, std=std_base)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Encode and return the CLS-pooled summary ``(B, d_model)``."""
        cls, _ = self.encode(token_ids)
        return cls

    def encode(
        self,
        token_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the encoder and return both pooled and per-token hidden states.

        Returns
        -------
        (cls, tokens):
            ``cls`` has shape ``(B, d_model)`` — the CLS pooled summary
            used by RL heads.
            ``tokens`` has shape ``(B, S, d_model)`` and contains the
            final hidden state for each *real* input token (i.e. with
            the CLS slot stripped).  Pad positions are present but were
            masked out of attention.  Used by masked-token-modelling
            heads for representation pretraining.
        """
        if token_ids.shape[1] > self.max_seq_len:
            token_ids = token_ids[:, -self.max_seq_len :]
        pad_mask = token_ids == self.pad_token_id  # (B, S) — True == pad
        embeddings = self.token_embedding(token_ids)
        return self.encode_from_embeddings(embeddings, pad_mask)

    def encode_from_embeddings(
        self,
        embeddings: torch.Tensor,
        pad_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the transformer stack on *pre-built* token embeddings.

        Used by :class:`agents.networks.hybrid_encoder.HybridTokenEncoder`
        when the embedding sequence has been assembled from a mix of
        categorical lookups and continuous-feature encoders (PLE /
        Time2Vec / Fourier).  Same return contract as :meth:`encode`.

        Parameters
        ----------
        embeddings:
            ``(B, S, d_model)`` already-built per-token embeddings.  The
            CLS token will be prepended internally.
        pad_mask:
            ``(B, S)`` bool tensor, ``True`` at padded positions (which
            attention will ignore).
        """
        b = embeddings.size(0)
        if embeddings.size(1) > self.max_seq_len:
            embeddings = embeddings[:, -self.max_seq_len :]
            pad_mask = pad_mask[:, -self.max_seq_len :]

        cls = self.cls_token.expand(b, -1, -1)
        x = torch.cat([cls, embeddings], dim=1)
        cls_mask = torch.zeros(b, 1, dtype=torch.bool, device=embeddings.device)
        full_pad_mask = torch.cat([cls_mask, pad_mask], dim=1)
        x = self.embed_dropout(x)

        for block in self.blocks:
            x = block(x, key_padding_mask=full_pad_mask)

        x = self.final_norm(x)
        return x[:, 0], x[:, 1:]


__all__ = ["ModernTransformerEncoder", "RMSNorm", "SwiGLU", "TransformerBlock"]
