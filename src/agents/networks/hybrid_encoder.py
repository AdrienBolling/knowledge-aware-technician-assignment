"""Hybrid token / continuous-feature transformer encoder.

Composes a :class:`ModernTransformerEncoder` backbone with the three
continuous-feature modules from
:mod:`agents.networks.continuous_features` so the residual stream can
carry, position-by-position, either a categorical token embedding or a
smooth real-valued encoding routed to the right encoder by a
``cont_kinds`` channel emitted by the env.

The categorical pipeline is unchanged — the same token-id embedding,
the same vocabulary, the same MLM-compatible interface.  What changes
is that bucket *value* tokens (``R_0_10``, ``T_500_1K``, ``C_2_3``) are
gone: instead the env emits a single ``<NUM>`` placeholder at each
numerical-value position alongside the raw scalar in ``cont_values``
and a kind code in ``cont_kinds``.  The encoder rewrites those
placeholder embeddings with PLE / Time2Vec / Fourier outputs before
running attention.
"""

from __future__ import annotations

from typing import Sequence

import math

import torch
from torch import nn

from agents.networks.continuous_features import (
    ContKind,
    FourierFeatures,
    PiecewiseLinearEncoding,
    Time2Vec,
)
from agents.networks.modern_transformer import ModernTransformerEncoder


# Sensible defaults for the KATA token vocabulary — exposed at module
# level so tokenizer / env / encoder can stay in sync without passing
# them through three layers of configuration.
DEFAULT_RATIO_EDGES: tuple[float, ...] = (
    0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
)
DEFAULT_COUNT_LOG_EDGES: tuple[float, ...] = (
    # log1p of {0, 1, 2, 5, 10, 20, 50, 100, 500, 1000, 10_000}
    math.log1p(0.0),
    math.log1p(1.0),
    math.log1p(2.0),
    math.log1p(5.0),
    math.log1p(10.0),
    math.log1p(20.0),
    math.log1p(50.0),
    math.log1p(100.0),
    math.log1p(500.0),
    math.log1p(1000.0),
    math.log1p(10_000.0),
)


class HybridTokenEncoder(nn.Module):
    """Transformer encoder over mixed categorical + continuous inputs.

    Parameters
    ----------
    vocab_size:
        Size of the categorical vocabulary (matches the tokenizer).
    d_model, n_heads, n_layers, max_seq_len, dropout, pad_token_id:
        Forwarded to :class:`ModernTransformerEncoder`.
    sim_time_scale:
        Inputs to :class:`FourierFeatures` are divided by this scale
        before the cos/sin pass.  Setting it to your env's
        ``max_sim_time`` keeps the random-frequency NTK in a useful
        range.  Defaults to 200000 to match ``factory_long.json``.
    ratio_bin_edges / count_bin_edges_log1p:
        Override the default PLE binning if your scalar ranges differ.
    n_time2vec_freqs / n_fourier_freqs:
        Size of the sine bases for Time2Vec / Fourier.
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
        *,
        sim_time_scale: float = 200_000.0,
        ratio_bin_edges: Sequence[float] | None = None,
        count_bin_edges_log1p: Sequence[float] | None = None,
        n_time2vec_freqs: int = 16,
        n_fourier_freqs: int = 16,
        fourier_sigma: float = 1.0,
    ) -> None:
        super().__init__()
        self.backbone = ModernTransformerEncoder(
            vocab_size=vocab_size,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            max_seq_len=max_seq_len,
            dropout=dropout,
            pad_token_id=pad_token_id,
        )

        edges_r = list(ratio_bin_edges or DEFAULT_RATIO_EDGES)
        edges_c = list(count_bin_edges_log1p or DEFAULT_COUNT_LOG_EDGES)
        self.ratio_ple = PiecewiseLinearEncoding(edges_r, d_model=d_model)
        self.count_ple = PiecewiseLinearEncoding(edges_c, d_model=d_model)
        self.time2vec = Time2Vec(d_model=d_model, n_freqs=n_time2vec_freqs)
        self.fourier = FourierFeatures(
            d_model=d_model,
            n_freqs=n_fourier_freqs,
            sigma=fourier_sigma,
            input_scale=float(sim_time_scale),
        )

    @property
    def output_dim(self) -> int:
        return self.backbone.output_dim

    @property
    def max_seq_len(self) -> int:
        return self.backbone.max_seq_len

    @property
    def pad_token_id(self) -> int:
        return self.backbone.pad_token_id

    # ------------------------------------------------------------------

    def _fuse(
        self,
        token_ids: torch.Tensor,
        cont_values: torch.Tensor,
        cont_kinds: torch.Tensor,
    ) -> torch.Tensor:
        """Build the ``(B, S, d_model)`` mixed-source embedding tensor.

        For each position, the final embedding is the one produced by
        the encoder whose code matches ``cont_kinds[b, s]``.  The
        per-position selection is a masked sum (cheap, no scatter /
        index_put).

        For efficiency, each *continuous* encoder is only evaluated
        when at least one position in the batch routes to it --- a
        common case is that a batch contains no time2vec / fourier
        positions, in which case skipping the encoder saves
        ``B * S * d_model`` of needless work and a Linear projection.
        The categorical path always runs because it is also the
        embedding source for the ``<NUM>`` placeholder, which sits at
        *every* continuous-value position and contributes nothing to
        the final sum but is cheap.
        """
        # Categorical path: always-on, indexes the embedding table.
        x_cat = self.backbone.token_embedding(token_ids)  # (B, S, D)

        m_cat = (cont_kinds == ContKind.CATEGORICAL).unsqueeze(-1).float()
        out = x_cat * m_cat

        # Continuous paths: each kind contributes only if any position
        # actually routes to it.  ``mask.any()`` is a cheap reduction;
        # skipping the encoder saves a Linear-projection forward.
        m_r = cont_kinds == ContKind.RATIO_PLE
        if m_r.any():
            out = out + self.ratio_ple(
                cont_values.clamp(min=0.0, max=1.0)
            ) * m_r.unsqueeze(-1).float()

        m_c = cont_kinds == ContKind.COUNT_PLE
        if m_c.any():
            out = out + self.count_ple(
                torch.log1p(cont_values.clamp(min=0.0))
            ) * m_c.unsqueeze(-1).float()

        m_t = cont_kinds == ContKind.TIME2VEC
        if m_t.any():
            out = out + self.time2vec(cont_values) * m_t.unsqueeze(-1).float()

        m_f = cont_kinds == ContKind.FOURIER
        if m_f.any():
            out = out + self.fourier(cont_values) * m_f.unsqueeze(-1).float()

        return out

    # ------------------------------------------------------------------

    def encode(
        self,
        token_ids: torch.Tensor,
        cont_values: torch.Tensor,
        cont_kinds: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(cls, per_token_hidden)`` as :class:`ModernTransformerEncoder` does.

        Inputs are aligned ``(B, S)`` tensors emitted by the env's
        ``hybrid`` observation mode:

        * ``token_ids`` — categorical IDs (or ``<NUM>`` at numerical positions).
        * ``cont_values`` — raw scalar (``0`` where categorical).
        * ``cont_kinds`` — :class:`ContKind` code per position.
        """
        if token_ids.shape[1] > self.backbone.max_seq_len:
            token_ids = token_ids[:, -self.backbone.max_seq_len :]
            cont_values = cont_values[:, -self.backbone.max_seq_len :]
            cont_kinds = cont_kinds[:, -self.backbone.max_seq_len :]

        pad_mask = token_ids == self.backbone.pad_token_id
        embeddings = self._fuse(token_ids, cont_values, cont_kinds)
        return self.backbone.encode_from_embeddings(embeddings, pad_mask)

    def forward(
        self,
        token_ids: torch.Tensor,
        cont_values: torch.Tensor,
        cont_kinds: torch.Tensor,
    ) -> torch.Tensor:
        cls, _ = self.encode(token_ids, cont_values, cont_kinds)
        return cls

    @property
    def token_embedding(self) -> nn.Embedding:
        """Convenience: expose the categorical embedding for MLM heads."""
        return self.backbone.token_embedding


__all__ = ["HybridTokenEncoder", "DEFAULT_RATIO_EDGES", "DEFAULT_COUNT_LOG_EDGES"]
