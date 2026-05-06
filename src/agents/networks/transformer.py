"""Generic Transformer encoder for processing tokenized observations.

The architecture follows the standard encoder-only Transformer:
token embedding + positional encoding -> N Transformer blocks -> pooled output.

The output is a fixed-size vector suitable for policy or value heads.
"""

from __future__ import annotations

import math

import torch
from torch import nn


class TransformerEncoder(nn.Module):
    """Encoder-only Transformer for tokenized observations.

    Parameters
    ----------
    vocab_size:
        Number of tokens in the vocabulary (including special tokens).
    d_model:
        Embedding dimension.
    n_heads:
        Number of attention heads.
    n_layers:
        Number of Transformer encoder layers.
    d_ff:
        Feed-forward hidden dimension.  Defaults to ``4 * d_model``.
    max_seq_len:
        Maximum sequence length (for positional encoding).
    dropout:
        Dropout rate.
    pad_token_id:
        Token ID used for padding (masked out in attention).

    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int | None = None,
        max_seq_len: int = 128,
        dropout: float = 0.1,
        pad_token_id: int = 0,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.pad_token_id = pad_token_id

        self.token_embedding = nn.Embedding(
            vocab_size, d_model, padding_idx=pad_token_id
        )
        self.pos_encoding = SinusoidalPositionalEncoding(d_model, max_seq_len)
        self.dropout = nn.Dropout(dropout)

        d_ff = d_ff or 4 * d_model
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.output_norm = nn.LayerNorm(d_model)

    @property
    def output_dim(self) -> int:
        """Dimension of the output vector."""
        return self.d_model

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Encode a batch of token ID sequences.

        Parameters
        ----------
        token_ids:
            ``(batch, seq_len)`` tensor of integer token IDs.

        Returns
        -------
        torch.Tensor
            ``(batch, d_model)`` pooled representation.

        """
        # Padding mask: True where token is PAD (will be ignored)
        pad_mask = token_ids == self.pad_token_id  # (B, S)

        x = self.token_embedding(token_ids)  # (B, S, D)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        x = self.encoder(x, src_key_padding_mask=pad_mask)  # (B, S, D)
        x = self.output_norm(x)

        # Mean-pool over non-padding positions
        mask = (~pad_mask).unsqueeze(-1).float()  # (B, S, 1)
        pooled = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)  # (B, D)
        return pooled


class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 512) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: d_model // 2])
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input embeddings."""
        return x + self.pe[:, : x.size(1)]
