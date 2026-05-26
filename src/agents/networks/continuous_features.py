"""Continuous-feature encoders for hybrid token observations.

Implements three encoders from the recent tabular-DL / continuous-feature
literature that replace the crude one-hot "bucket tokens" used previously:

* :class:`PiecewiseLinearEncoding` — Gorishniy et al. (NeurIPS 2022).
  Replaces one-hot bucketing with an *ordered, smooth* K-dim encoding
  that preserves the relative position of a value inside its bin.
* :class:`Time2Vec` — Kazemi et al. 2019.  A learnable linear + sine
  basis suitable for *recent* event timings where both magnitude and
  recurrence matter.
* :class:`FourierFeatures` — Tancik et al. (NeurIPS 2020).  Fixed
  random-frequency cos/sin features, optimal for long-horizon scalar
  signals where the NTK should remain well-conditioned across a wide
  dynamic range.

All three accept a ``(B, S)`` float tensor of scalar values and return
``(B, S, d_model)`` embeddings ready to be inserted into the residual
stream of a Transformer.
"""

from __future__ import annotations

import math
from typing import Iterable

import torch
from torch import nn


# ---------------------------------------------------------------------------
# Piecewise-linear encoding
# ---------------------------------------------------------------------------


class PiecewiseLinearEncoding(nn.Module):
    """K-dim ramp encoding + linear projection to ``d_model``.

    Bin edges are passed at construction.  For an input value ``x`` and
    edges ``b_0 < b_1 < ... < b_K`` the encoding vector ``p ∈ R^K``
    satisfies, for each ``k = 0, …, K-1``::

        p_k(x) = 0                                   if x ≤ b_k
        p_k(x) = (x - b_k) / (b_{k+1} - b_k)         if b_k < x ≤ b_{k+1}
        p_k(x) = 1                                   if x > b_{k+1}

    Values outside ``[b_0, b_K]`` are clamped — this is intentional;
    PLE is for *bounded* features.  For long-horizon time signals use
    :class:`FourierFeatures` instead.

    The resulting ``(B, S, K)`` ramp tensor is projected to
    ``(B, S, d_model)`` by a learnable linear layer + GELU + linear
    (a small MLP — Gorishniy 2022 found a tiny non-linearity helps).

    Parameters
    ----------
    bin_edges:
        Monotonically increasing edges defining the bins.  ``K = len(edges) - 1``.
    d_model:
        Output dimensionality of the projection.
    hidden_dim:
        Width of the internal MLP.  Defaults to ``d_model``.
    """

    def __init__(
        self,
        bin_edges: Iterable[float],
        d_model: int,
        hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        edges = torch.as_tensor(list(bin_edges), dtype=torch.float32)
        if edges.ndim != 1 or len(edges) < 2:
            msg = "bin_edges must be a 1-D iterable with at least 2 values"
            raise ValueError(msg)
        if not torch.all(edges[1:] > edges[:-1]):
            msg = "bin_edges must be strictly increasing"
            raise ValueError(msg)

        self.register_buffer("edges", edges)
        self.k = int(len(edges) - 1)
        widths = (edges[1:] - edges[:-1]).clamp(min=1e-8)
        self.register_buffer("widths", widths)

        hidden_dim = hidden_dim or d_model
        self.proj = nn.Sequential(
            nn.Linear(self.k, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_model),
        )

    @property
    def output_dim(self) -> int:
        return self.proj[-1].out_features

    def encode_raw(self, x: torch.Tensor) -> torch.Tensor:
        """Return the K-dim ramp encoding ``(..., K)`` for ``x`` ``(...)``."""
        # Broadcast: (..., 1) vs edges (K+1,) -> (..., K+1)
        x_b = x.unsqueeze(-1)
        lower = self.edges[:-1]  # (K,)
        ramp = (x_b - lower) / self.widths
        return ramp.clamp(min=0.0, max=1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode and project: ``(...)`` → ``(..., d_model)``."""
        raw = self.encode_raw(x)
        return self.proj(raw)


# ---------------------------------------------------------------------------
# Time2Vec
# ---------------------------------------------------------------------------


class Time2Vec(nn.Module):
    """Learnable linear + sine basis encoding for scalar time inputs.

    Following Kazemi et al. 2019, a scalar ``t`` is mapped to a
    ``K``-dim vector::

        Time2Vec(t)_0 = ω_0 · t + φ_0           (linear component)
        Time2Vec(t)_k = sin(ω_k · t + φ_k)      for k = 1, ..., K-1

    where ``ω`` and ``φ`` are both learnable.  The linear term captures
    monotonic "age" structure; the sine bank captures repeated patterns
    at multiple time-scales.

    The K-dim Time2Vec output is then projected to ``d_model``.

    Parameters
    ----------
    d_model:
        Output dimensionality after the projection.
    n_freqs:
        Number of sine basis functions (K = 1 linear + n_freqs).
        Defaults to 16.
    """

    def __init__(self, d_model: int, n_freqs: int = 16) -> None:
        super().__init__()
        if n_freqs < 1:
            msg = "n_freqs must be >= 1"
            raise ValueError(msg)
        self.n_freqs = int(n_freqs)
        # ω_0 (linear) + ω_1..ω_K (periodic) → K+1 frequencies in total.
        self.omega = nn.Parameter(torch.randn(self.n_freqs + 1) * 0.1)
        self.phi = nn.Parameter(torch.randn(self.n_freqs + 1) * 0.1)
        self.proj = nn.Linear(self.n_freqs + 1, d_model)

    @property
    def output_dim(self) -> int:
        return self.proj.out_features

    def encode_raw(self, t: torch.Tensor) -> torch.Tensor:
        """Return the ``(K+1)``-dim Time2Vec vector ``(..., K+1)``."""
        t_b = t.unsqueeze(-1)  # (..., 1)
        v = t_b * self.omega + self.phi  # (..., K+1)
        # First dimension is linear, the rest are sine.
        out = torch.cat([v[..., :1], torch.sin(v[..., 1:])], dim=-1)
        return out

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Encode and project: ``(...)`` → ``(..., d_model)``."""
        return self.proj(self.encode_raw(t))


# ---------------------------------------------------------------------------
# Fourier features
# ---------------------------------------------------------------------------


class FourierFeatures(nn.Module):
    """Random-frequency cos/sin features (Tancik et al. NeurIPS 2020).

    A scalar ``t`` is mapped to ``[cos(2π B t), sin(2π B t)]`` ∈ R^{2K}
    where ``B ∼ 𝒩(0, σ²)^K`` is sampled once at construction and *not*
    learned thereafter.  The fixed-frequency spectrum keeps the network's
    NTK well-conditioned over a wide range of input magnitudes, which is
    why this is the standard recipe for positional encoding in NeRFs and
    long-horizon scalar inputs.

    The 2K-dim feature vector is then projected to ``d_model`` by a
    learnable linear layer.

    Parameters
    ----------
    d_model:
        Output dimensionality after the projection.
    n_freqs:
        Number of independent (cos, sin) pairs (so the raw feature dim
        is ``2 * n_freqs``).  Defaults to 16.
    sigma:
        Standard deviation of the Gaussian frequency prior.  Larger
        sigma → higher frequencies / better NTK conditioning for very
        smooth inputs.  Tancik 2020 reports σ ∈ [1, 10] as the useful
        range for normalised inputs.
    input_scale:
        Inputs are divided by ``input_scale`` before the Fourier pass.
        For SIM_TIME-style inputs ranging up to ``max_sim_time`` this
        should be set to that range so frequencies remain in a useful
        regime.  Defaults to 1.0 (no rescaling).
    """

    def __init__(
        self,
        d_model: int,
        n_freqs: int = 16,
        sigma: float = 1.0,
        input_scale: float = 1.0,
    ) -> None:
        super().__init__()
        if n_freqs < 1:
            msg = "n_freqs must be >= 1"
            raise ValueError(msg)
        self.n_freqs = int(n_freqs)
        self.input_scale = float(input_scale)
        b = torch.randn(self.n_freqs) * float(sigma)
        self.register_buffer("B", b)  # frozen frequencies
        self.proj = nn.Linear(2 * self.n_freqs, d_model)

    @property
    def output_dim(self) -> int:
        return self.proj.out_features

    def encode_raw(self, t: torch.Tensor) -> torch.Tensor:
        """Return the ``(2K)``-dim Fourier feature vector ``(..., 2K)``."""
        t_b = t.unsqueeze(-1) / self.input_scale  # (..., 1)
        angles = 2.0 * math.pi * (t_b * self.B)  # (..., K)
        return torch.cat([torch.cos(angles), torch.sin(angles)], dim=-1)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Encode and project: ``(...)`` → ``(..., d_model)``."""
        return self.proj(self.encode_raw(t))


# ---------------------------------------------------------------------------
# Default kind constants (single source of truth shared with env.py)
# ---------------------------------------------------------------------------


class ContKind:
    """Integer codes identifying the continuous-encoder for each position."""

    CATEGORICAL = 0  # use token-id embedding (no continuous channel)
    RATIO_PLE = 1
    COUNT_PLE = 2
    TIME2VEC = 3
    FOURIER = 4


__all__ = [
    "ContKind",
    "FourierFeatures",
    "PiecewiseLinearEncoding",
    "Time2Vec",
]
