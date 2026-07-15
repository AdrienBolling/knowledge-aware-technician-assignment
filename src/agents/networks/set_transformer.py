"""Three-stream Set-Transformer encoder + pointer-network action head.

Architecture
------------

The observation produced by ``KataEnv`` in ``observation_representation
= "set"`` mode arrives as three grouped streams::

    tech_*    : (B, max_techs,     L_tech)   token_ids / cont_values / cont_kinds + (B, max_techs)     mask
    machine_* : (B, max_machines,  L_mach)   ...                                    + (B, max_machines) mask
    env_*     : (B, L_env)         ...

This encoder maps that to a per-technician logit vector ``(B,
max_techs)`` plus a scalar value estimate.  The flow is:

1. **Per-slot fusion.**  Each slot's ``L``-position hybrid sub-sequence
   is fused into ``L`` token-level embeddings using the same
   PLE / Time2Vec / Fourier routing as
   :class:`agents.networks.hybrid_encoder.HybridTokenEncoder`.

2. **Per-slot pooling.**  The ``L`` per-position embeddings are
   mean-pooled into one vector per slot.  Padded positions (whose
   ``cont_kinds`` is ``CATEGORICAL`` and whose ``token_id`` is the pad
   id) contribute their pad-embedding to the mean — fine because the
   slot's *own* validity is tracked separately by the slot mask.

3. **Cross-slot Set Transformer.**  Per-slot vectors are processed by
   a small Transformer encoder with the slot mask as a key-padding
   mask, producing contextualised per-slot embeddings + a CLS summary
   (``slot_cls_*``).  Two such stacks: one for techs, one for machines.

4. **Env MLP.**  The env stream has no slot-set structure, so its
   ``L_env`` positions are fused, mean-pooled, and passed through a
   small MLP into a fixed-size summary.

5. **Context fusion.**  The three summaries
   ``(tech_cls, machine_cls, env_summary)`` are concatenated and
   projected to a ``d_core`` policy-context vector ``s``.

6. **Pointer action head.**  For each technician slot ``i`` with
   contextualised embedding ``e_i`` and slot mask ``valid_i``, the
   logit is::

       logit_i = (W_q s)^T (W_k e_i) / sqrt(d_core)

   Invalid slots (mask = 0) get ``-inf`` so softmax ignores them.  The
   number of parameters in this head is independent of
   ``max_techs`` — it transfers across environments with different
   fleet sizes.

7. **Value head.**  A small MLP on ``s``.

Design choices
--------------

* The categorical embedding table is shared across all three streams
  (one vocab fits all three sub-vocabularies).
* The per-position continuous encoders (PLE, Time2Vec, Fourier) are
  shared too — there is no reason a "fatigue ratio" would need a
  different bin schedule on technicians vs the env stream.
* Mean-pooling within slots avoids the cost of per-slot transformers;
  the cross-slot transformer is where the heavy lifting happens.
* CLS-token pooling in the cross-slot transformer is used for the
  context summary — same recipe as BERT and the project's existing
  ``ModernTransformerEncoder``.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch
from torch import nn

from agents.networks.continuous_features import (
    ContKind,
    FourierFeatures,
    PiecewiseLinearEncoding,
    Time2Vec,
)
from agents.networks.hybrid_encoder import (
    DEFAULT_COUNT_LOG_EDGES,
    DEFAULT_RATIO_EDGES,
)
from agents.networks.modern_transformer import (
    RMSNorm,
    TransformerBlock,
)


class _SlotFuser(nn.Module):
    """Hybrid-fuse a ``(B, S, L)`` set of slot sub-sequences.

    Output: ``(B, S, L, d_model)`` per-position embeddings.  The
    categorical token embedding, PLE, Time2Vec and Fourier modules are
    shared across all three streams (techs / machines / env).
    """

    def __init__(
        self,
        token_embedding: nn.Embedding,
        ratio_ple: PiecewiseLinearEncoding,
        count_ple: PiecewiseLinearEncoding,
        time2vec: Time2Vec,
        fourier: FourierFeatures,
    ) -> None:
        super().__init__()
        self.token_embedding = token_embedding
        self.ratio_ple = ratio_ple
        self.count_ple = count_ple
        self.time2vec = time2vec
        self.fourier = fourier

    def forward(
        self,
        token_ids: torch.Tensor,
        cont_values: torch.Tensor,
        cont_kinds: torch.Tensor,
    ) -> torch.Tensor:
        # token_ids: (..., L) int64
        # cont_values: (..., L) float32
        # cont_kinds: (..., L) int (small)
        x_cat = self.token_embedding(token_ids)  # (..., L, D)
        m_cat = (cont_kinds == ContKind.CATEGORICAL).unsqueeze(-1).float()
        out = x_cat * m_cat

        m_r = cont_kinds == ContKind.RATIO_PLE
        if m_r.any():
            v = cont_values.clamp(min=0.0, max=1.0)
            out = out + self.ratio_ple(v) * m_r.unsqueeze(-1).float()

        m_c = cont_kinds == ContKind.COUNT_PLE
        if m_c.any():
            v = torch.log1p(cont_values.clamp(min=0.0))
            out = out + self.count_ple(v) * m_c.unsqueeze(-1).float()

        m_t = cont_kinds == ContKind.TIME2VEC
        if m_t.any():
            out = out + self.time2vec(cont_values) * m_t.unsqueeze(-1).float()

        m_f = cont_kinds == ContKind.FOURIER
        if m_f.any():
            out = out + self.fourier(cont_values) * m_f.unsqueeze(-1).float()

        return out


class _SetEncoder(nn.Module):
    """Stack of TransformerBlocks with a learnable CLS token and key-padding mask.

    Returns ``(cls, slots)`` where ``cls`` is ``(B, d_model)`` and
    ``slots`` is ``(B, S, d_model)`` of contextualised per-slot
    embeddings.  Invalid slots (per the mask) attend to nothing and
    produce undefined values — *consumers must apply the mask
    themselves* before any reduction that mixes valid and invalid
    slots.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_layers: int,
        d_ff: int | None,
        max_slots: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if d_ff is None:
            d_ff = int(round((8 * d_model / 3) / 64.0)) * 64
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    n_heads=n_heads,
                    d_ff=d_ff,
                    max_seq_len=max_slots + 1,
                    dropout=dropout,
                )
                for _ in range(n_layers)
            ]
        )
        self.final_norm = RMSNorm(d_model)

    def forward(
        self, slot_embeds: torch.Tensor, slot_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``slot_embeds``: (B, S, D);  ``slot_mask``: (B, S) bool/int (1 = valid)."""
        b = slot_embeds.size(0)
        cls = self.cls_token.expand(b, -1, -1)
        x = torch.cat([cls, slot_embeds], dim=1)
        # key_padding_mask convention in TransformerBlock matches
        # nn.MultiheadAttention: True == ignore.
        pad = ~slot_mask.bool()
        cls_pad = torch.zeros(b, 1, dtype=torch.bool, device=slot_embeds.device)
        full_pad = torch.cat([cls_pad, pad], dim=1)
        for block in self.blocks:
            x = block(x, key_padding_mask=full_pad)
        x = self.final_norm(x)
        return x[:, 0], x[:, 1:]


class _CrossAttentionRefiner(nn.Module):
    """Single-block cross-attention refinement of per-tech embeddings.

    Each technician slot attends over the *concatenated* machine slots
    and env summary, gaining direct access to "what's broken, what's
    queued, what's the current ticket" before the pointer head scores
    it.  Without this layer, ``e_i`` only sees other technicians (via
    the Tech SetTransformer's self-attention) and the global state
    only reaches the pointer head via the scalar context ``s`` — which
    is identical for every candidate and so cannot tilt one tech's
    score over another based on a tech-specific knowledge match.

    Architecture: standard pre-norm transformer block with one
    cross-attention sub-layer + one feed-forward sub-layer, residually
    connected.  Uses :class:`torch.nn.MultiheadAttention` (no RoPE —
    positions in a *set* of techs or machines have no order to encode).
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
        d_ff: int | None = None,
    ) -> None:
        super().__init__()
        if d_ff is None:
            d_ff = int(round((8 * d_model / 3) / 64.0)) * 64
        self.norm_q = RMSNorm(d_model)
        self.norm_kv = RMSNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm_ffn = RMSNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )
        self.attn_drop = (
            nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        )

    def forward(
        self,
        queries: torch.Tensor,         # (B, S_q, d)
        kv: torch.Tensor,              # (B, S_k, d)
        kv_pad_mask: torch.Tensor,     # (B, S_k) bool — True == ignore
    ) -> torch.Tensor:
        # Cross-attention sub-layer (pre-norm + residual)
        q = self.norm_q(queries)
        k = self.norm_kv(kv)
        attn_out, _ = self.attn(
            q, k, k, key_padding_mask=kv_pad_mask, need_weights=False
        )
        x = queries + self.attn_drop(attn_out)
        # Feed-forward sub-layer
        x = x + self.ffn(self.norm_ffn(x))
        return x


class _ContextProjection(nn.Module):
    """Concat-projection block: ``(3·d_model) → d_core``.

    Replaces the bare ``torch.cat`` with a small 2-layer MLP behind an
    RMSNorm so the heads consume a shared, fixed-size *policy state*
    ``s ∈ ℝ^{d_core}``.  Cheap (≤ 100k params for typical sizes) but
    gives an interpretable shared bottleneck and uniform scaling of
    the three stream summaries.
    """

    def __init__(self, d_in: int, d_out: int) -> None:
        super().__init__()
        self.norm = RMSNorm(d_in)
        self.fc1 = nn.Linear(d_in, 2 * d_out)
        self.fc2 = nn.Linear(2 * d_out, d_out)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(self.norm(x))))


class SetTransformerEncoder(nn.Module):
    """Three-stream encoder over the ``set`` observation.

    Parameters
    ----------
    vocab_size:
        Size of the shared categorical vocabulary.
    d_model:
        Width of the residual stream.  Same in all three streams.
    n_heads, n_layers, d_ff, dropout:
        Forwarded to the cross-slot Transformer encoders.
    max_techs, max_machines, env_length:
        Fixed slot/length caps matching the env's set-mode config.
    sim_time_scale:
        Fourier-feature input scaler — match the env's ``max_sim_time``.
    d_core:
        Output dimensionality of the context projection block.  The
        policy state ``s`` produced by ``Concat + Proj`` lives in
        ``ℝ^{d_core}``; both heads consume it.  Defaults to ``d_model``.
    use_cross_attention:
        When True (default) a cross-attention refiner contextualises
        per-tech embeddings with the machine + env state before the
        pointer head sees them.  Set False to ablate the layer.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 192,
        n_heads: int = 6,
        n_layers: int = 4,
        d_ff: int | None = None,
        dropout: float = 0.1,
        *,
        max_techs: int = 30,
        max_machines: int = 100,
        env_length: int = 8,
        pad_token_id: int = 0,
        sim_time_scale: float = 200_000.0,
        ratio_bin_edges: Sequence[float] | None = None,
        count_bin_edges_log1p: Sequence[float] | None = None,
        n_time2vec_freqs: int = 16,
        n_fourier_freqs: int = 16,
        fourier_sigma: float = 1.0,
        d_core: int | None = None,
        use_cross_attention: bool = True,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.max_techs = max_techs
        self.max_machines = max_machines
        self.env_length = env_length
        self.pad_token_id = pad_token_id
        self.d_core = int(d_core if d_core is not None else d_model)
        self.use_cross_attention = bool(use_cross_attention)

        # Shared embedding + continuous encoders across all 3 streams.
        self.token_embedding = nn.Embedding(
            vocab_size, d_model, padding_idx=pad_token_id
        )
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)

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

        self._fuser = _SlotFuser(
            token_embedding=self.token_embedding,
            ratio_ple=self.ratio_ple,
            count_ple=self.count_ple,
            time2vec=self.time2vec,
            fourier=self.fourier,
        )

        # Cross-slot encoders, one per slot-set.
        self.tech_encoder = _SetEncoder(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            max_slots=max_techs,
            dropout=dropout,
        )
        self.machine_encoder = _SetEncoder(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            max_slots=max_machines,
            dropout=dropout,
        )

        # Env stream: just an MLP on the mean-pooled fused embedding.
        self.env_mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        # Optional cross-attention refiner: per-tech embeddings attend
        # over (machine slots + env summary) so the pointer head sees
        # state-conditional tech features.
        self.cross_attn: _CrossAttentionRefiner | None = (
            _CrossAttentionRefiner(
                d_model=d_model, n_heads=n_heads, dropout=dropout, d_ff=d_ff
            )
            if self.use_cross_attention
            else None
        )

        # Concat → norm → 2-layer MLP → policy state ``s``.
        self.context_proj = _ContextProjection(
            d_in=3 * d_model, d_out=self.d_core
        )

    @property
    def output_dim(self) -> int:
        """Dim of the policy-state vector ``s`` produced by Concat+Proj."""
        return self.d_core

    # ------------------------------------------------------------------

    def _pool_slot(
        self,
        token_ids: torch.Tensor,
        cont_values: torch.Tensor,
        cont_kinds: torch.Tensor,
    ) -> torch.Tensor:
        """Fuse and mean-pool within each slot's L positions.

        Inputs are ``(B, S, L)``.  Output is ``(B, S, D)``.  Pad
        positions (``token_id == pad``) are excluded from the mean.
        """
        embeds = self._fuser(token_ids, cont_values, cont_kinds)
        # (B, S, L, D)
        pos_valid = (token_ids != self.pad_token_id).float().unsqueeze(-1)
        # If ALL positions in a slot are pad, denom would be 0 — clamp.
        denom = pos_valid.sum(dim=-2).clamp(min=1.0)
        pooled = (embeds * pos_valid).sum(dim=-2) / denom
        return pooled  # (B, S, D)

    def _pool_env(
        self,
        token_ids: torch.Tensor,
        cont_values: torch.Tensor,
        cont_kinds: torch.Tensor,
    ) -> torch.Tensor:
        """Fuse and mean-pool the env stream into a ``(B, D)`` vector."""
        embeds = self._fuser(token_ids, cont_values, cont_kinds)
        # (B, L_env, D)
        pos_valid = (token_ids != self.pad_token_id).float().unsqueeze(-1)
        denom = pos_valid.sum(dim=-2).clamp(min=1.0)
        return (embeds * pos_valid).sum(dim=-2) / denom

    def forward(
        self,
        obs: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(context, tech_slot_embeds, tech_mask)``.

        * ``context`` ``(B, d_core)`` — the policy state ``s`` produced
          by the Concat+Proj block from the three stream summaries.
          Feeds the value head and (with the per-tech embeddings) the
          pointer action head.
        * ``tech_slot_embeds`` ``(B, max_techs, d_model)`` — contextua-
          lised per-technician embeddings used by the pointer head.
          Optionally refined by cross-attention over the machine + env
          state (see ``use_cross_attention``).
        * ``tech_mask`` ``(B, max_techs)`` — bool, 1 == valid slot.
          Passed through unchanged so the action head can mask
          invalid logits to ``-inf``.
        """
        tech_pooled = self._pool_slot(
            obs["tech_token_ids"], obs["tech_cont_values"], obs["tech_cont_kinds"]
        )
        machine_pooled = self._pool_slot(
            obs["machine_token_ids"],
            obs["machine_cont_values"],
            obs["machine_cont_kinds"],
        )

        tech_mask = obs["tech_mask"].bool()
        machine_mask = obs["machine_mask"].bool()

        tech_cls, tech_slots = self.tech_encoder(tech_pooled, tech_mask)
        machine_cls, machine_slots = self.machine_encoder(
            machine_pooled, machine_mask
        )

        env_pooled = self._pool_env(
            obs["env_token_ids"], obs["env_cont_values"], obs["env_cont_kinds"]
        )
        env_summary = self.env_mlp(env_pooled)

        # Cross-attention refinement: per-tech embeddings attend over
        # (machine slots + env).  After this, ``e_i`` is state-aware
        # and the pointer score ``s · e_i`` becomes tech-specific in
        # the context of the current machine and ticket state.
        if self.cross_attn is not None:
            b = tech_slots.size(0)
            # Build KV as the concatenation of machine slots and the
            # single env summary token.
            env_kv = env_summary.unsqueeze(1)  # (B, 1, d)
            kv = torch.cat([machine_slots, env_kv], dim=1)
            machine_pad = ~machine_mask
            env_pad = torch.zeros(b, 1, dtype=torch.bool, device=kv.device)
            kv_pad = torch.cat([machine_pad, env_pad], dim=1)
            # Guard against an all-pad KV row (would produce NaN in
            # softmax).  This can only happen if max_machines=0, which
            # is never the case for FactoReal — but be safe.
            all_pad = kv_pad.all(dim=-1)
            if all_pad.any():
                kv_pad = kv_pad.clone()
                kv_pad[all_pad, -1] = False  # un-pad the env slot
            tech_slots = self.cross_attn(tech_slots, kv, kv_pad)
            # Recompute the tech CLS as the masked mean of the refined
            # slots so the context vector also benefits from the
            # cross-attention.
            valid = tech_mask.float().unsqueeze(-1)
            denom = valid.sum(dim=-2).clamp(min=1.0)
            tech_cls = (tech_slots * valid).sum(dim=-2) / denom

        context_raw = torch.cat([tech_cls, machine_cls, env_summary], dim=-1)
        context = self.context_proj(context_raw)
        return context, tech_slots, tech_mask


class PointerActionHead(nn.Module):
    """Pointer-network scorer.

    Given a policy-context ``s ∈ ℝ^{d_core}`` and per-tech embeddings
    ``e_i ∈ ℝ^{d_slot}``, produces one logit per technician slot::

        logit_i = (W_q s)^T (W_k e_i) / sqrt(d_attn)

    Invalid slots receive ``-inf`` so softmax assigns them zero prob.
    Parameter count is independent of the number of technicians, so
    the head transfers across environments of different fleet sizes.
    """

    def __init__(self, d_context: int, d_slot: int, d_attn: int = 64) -> None:
        super().__init__()
        self.q_proj = nn.Linear(d_context, d_attn, bias=False)
        self.k_proj = nn.Linear(d_slot, d_attn, bias=False)
        self._scale = 1.0 / math.sqrt(d_attn)
        nn.init.orthogonal_(self.q_proj.weight, gain=0.01)
        nn.init.orthogonal_(self.k_proj.weight, gain=0.01)

    def forward(
        self,
        context: torch.Tensor,        # (B, d_context)
        slot_embeds: torch.Tensor,    # (B, S, d_slot)
        slot_mask: torch.Tensor,      # (B, S) bool, 1 = valid
    ) -> torch.Tensor:
        q = self.q_proj(context).unsqueeze(1)        # (B, 1, d_attn)
        k = self.k_proj(slot_embeds)                  # (B, S, d_attn)
        scores = (q * k).sum(dim=-1) * self._scale    # (B, S)
        scores = scores.masked_fill(~slot_mask, float("-inf"))
        return scores


class SetTransformerActorCritic(nn.Module):
    """Encoder + optional recurrent context + pointer policy head + value head.

    Forward signature:
    ``obs: dict[str, Tensor], hidden -> (logits, value, hidden_out)``.

    ``rnn_type`` (opt-in, default ``"none"``) inserts a single-layer GRU
    or LSTM between the encoder's pooled context and both heads, giving
    the policy a within-episode memory of past decisions.  When
    disabled the architecture (and therefore every checkpoint) is
    identical to the historical version and ``hidden`` passes through
    untouched, so old checkpoints remain loadable.
    """

    def __init__(
        self,
        encoder: SetTransformerEncoder,
        value_hidden: int = 256,
        pointer_d_attn: int = 64,
        rnn_type: str = "none",
        rnn_hidden: int = 128,
    ) -> None:
        super().__init__()
        if rnn_type not in ("none", "gru", "lstm"):
            msg = f"rnn_type must be 'none', 'gru', or 'lstm' (got {rnn_type!r})"
            raise ValueError(msg)
        self.encoder = encoder
        self.rnn_type = rnn_type
        self.rnn_hidden = int(rnn_hidden)
        d_ctx = encoder.output_dim
        d_slot = encoder.d_model
        if rnn_type == "gru":
            self.rnn: nn.Module | None = nn.GRU(
                d_ctx, self.rnn_hidden, batch_first=True
            )
            d_head = self.rnn_hidden
        elif rnn_type == "lstm":
            self.rnn = nn.LSTM(d_ctx, self.rnn_hidden, batch_first=True)
            d_head = self.rnn_hidden
        else:
            self.rnn = None
            d_head = d_ctx
        self.policy_head = PointerActionHead(d_head, d_slot, d_attn=pointer_d_attn)
        self.value_head = nn.Sequential(
            nn.Linear(d_head, value_hidden),
            nn.GELU(),
            nn.Linear(value_hidden, 1),
        )
        nn.init.orthogonal_(self.value_head[-1].weight, gain=1.0)
        nn.init.zeros_(self.value_head[-1].bias)

    def initial_hidden(self, batch_size: int, device: torch.device):
        """Zero hidden state for ``batch_size`` parallel streams."""
        if self.rnn_type == "gru":
            return torch.zeros(1, batch_size, self.rnn_hidden, device=device)
        if self.rnn_type == "lstm":
            return (
                torch.zeros(1, batch_size, self.rnn_hidden, device=device),
                torch.zeros(1, batch_size, self.rnn_hidden, device=device),
            )
        return None

    def forward(
        self,
        obs: dict[str, torch.Tensor],
        hidden=None,
    ) -> tuple[torch.Tensor, torch.Tensor, object]:
        context, tech_slots, tech_mask = self.encoder(obs)
        if self.rnn is not None:
            if hidden is None:
                hidden = self.initial_hidden(context.shape[0], context.device)
            # One recurrent step per decision: (B, 1, d_ctx) -> (B, 1, H)
            out, hidden_out = self.rnn(context.unsqueeze(1), hidden)
            head_in = out.squeeze(1)
        else:
            head_in = context
            hidden_out = hidden
        logits = self.policy_head(head_in, tech_slots, tech_mask)
        value = self.value_head(head_in).squeeze(-1)
        return logits, value, hidden_out


__all__ = [
    "SetTransformerEncoder",
    "PointerActionHead",
    "SetTransformerActorCritic",
]
