"""Flat feature encoding + plain-MLP heads for the traditional baselines.

The learned agents consume the ``set`` observation mode of
:class:`kata.env.KataEnv` through the SetTransformer encoder.  The
traditional MLP baselines (PPO-MLP / REINFORCE-MLP / DQN-MLP) need the
same information as one fixed-width vector instead.  This module
provides:

``SetObsFlattener``
    A stateless (no learnable parameters) ``nn.Module`` that turns the
    env's ``set`` observation dict into a constant-width float vector:
    every categorical / boolean slot position becomes a one-hot over
    its own category's cardinality (derived from the frozen set vocab),
    every continuous position becomes a single scalar (symlog-squashed
    for COUNT / TIME / FOURIER kinds, raw for RATIO kinds).

``MLPTrunk`` / ``MLPActorCritic`` / ``MLPPolicy`` / ``MLPQNetwork``
    Plain Linear-LayerNorm-ReLU stacks with the policy / value / Q
    heads the three baseline algorithms need.

Why one-hot from the vocab (the D11 lesson)
-------------------------------------------

In the set mode, categorical and boolean state lives ONLY in the token
id — the continuous channel is 0.0 at those positions
(``kata.env._SetEmitter.cat`` / ``.bool``, env.py:344-362).  A
flattener that dropped or hashed the ids would silently erase
BUSY / BROKEN / IS_CURRENT exactly the way defect D11 did for every
pre-fix checkpoint.  Building the per-position id->one-hot tables from
the canonical frozen vocab (``run_configs/vocab/set_vocab.json``)
guarantees a BUSY flip changes the output vector; any id outside the
position's own category (``<PAD>``, ``<UNK>``, legacy TRUE/FALSE
spellings) routes to a dedicated trailing OTHER bin instead of
aliasing a real value.

Slot layouts
------------

The per-slot emission order is frozen by ``KataEnv._set_obs``
(env.py:1159-1327) and mirrored here as module-level constants.  Slot
positions beyond the layout length are structural padding (PAD id,
value 0.0) and are skipped entirely.  Keep these tables in sync with
``_set_obs`` if the observation schema ever changes.

Because the layouts are mirrored rather than derived, an inserted or
reordered ``_set_obs`` emission would shift every downstream column of
the flat vector *silently*.  :meth:`SetObsFlattener.forward` therefore
cross-checks the observation's own ``*_cont_kinds`` channel against the
layouts once (the first forward that carries it) and raises instead of
encoding a drifted schema.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from agents.networks.continuous_features import ContKind
from kata.core.tokenizer import PAD_ID


@dataclass(frozen=True)
class SlotPosition:
    """One position of a frozen slot layout: emitter key + ContKind."""

    key: str
    kind: int


_KIND_NAMES: dict[int, str] = {
    int(v): k
    for k, v in vars(ContKind).items()
    if not k.startswith("_") and isinstance(v, int)
}


def _kind_repr(kind: int) -> str:
    """``3`` -> ``"3 (TIME2VEC)"`` for readable divergence errors."""
    return f"{int(kind)} ({_KIND_NAMES.get(int(kind), 'UNKNOWN')})"


# Emission order of KataEnv._set_obs (env.py:1170-1223).  14 positions,
# slot length 16 — positions 14-15 are always padding.
TECH_SLOT_LAYOUT: tuple[SlotPosition, ...] = (
    SlotPosition("TEMPLATE", ContKind.CATEGORICAL),
    SlotPosition("BUSY", ContKind.CATEGORICAL),
    SlotPosition("DISRUPT", ContKind.CATEGORICAL),
    SlotPosition("FATIGUE", ContKind.RATIO_PLE),
    SlotPosition("ASSIGNS", ContKind.COUNT_PLE),
    SlotPosition("KNOW_VOL", ContKind.COUNT_PLE),
    SlotPosition("KNOW_MAX", ContKind.COUNT_PLE),
    SlotPosition("KNOW_SPEC", ContKind.RATIO_PLE),
    SlotPosition("KNOW_ENT", ContKind.COUNT_PLE),
    SlotPosition("MATCH", ContKind.RATIO_PLE),
    SlotPosition("MATCH_N1", ContKind.RATIO_PLE),
    SlotPosition("MATCH_N2", ContKind.RATIO_PLE),
    SlotPosition("ETA", ContKind.TIME2VEC),
    SlotPosition("LAST_AGE", ContKind.TIME2VEC),
)

# Emission order of KataEnv._set_obs (env.py:1242-1282).  11 positions,
# slot length 12 — position 11 is always padding.
MACHINE_SLOT_LAYOUT: tuple[SlotPosition, ...] = (
    SlotPosition("M_TYPE", ContKind.CATEGORICAL),
    SlotPosition("BROKEN", ContKind.CATEGORICAL),
    SlotPosition("PROC", ContKind.CATEGORICAL),
    SlotPosition("IS_CURRENT", ContKind.CATEGORICAL),
    SlotPosition("CUR_COMP", ContKind.CATEGORICAL),
    SlotPosition("PROC_TOT", ContKind.COUNT_PLE),
    SlotPosition("IN_BUF", ContKind.COUNT_PLE),
    SlotPosition("OUT_BUF", ContKind.COUNT_PLE),
    SlotPosition("BD_COUNT", ContKind.COUNT_PLE),
    SlotPosition("DOWNTIME", ContKind.TIME2VEC),
    SlotPosition("MEAN_TBF", ContKind.TIME2VEC),
)

# Emission order of KataEnv._set_obs (env.py:1288-1327).  14 positions,
# env length 16 — positions 14-15 are always padding.
ENV_SLOT_LAYOUT: tuple[SlotPosition, ...] = (
    SlotPosition("HAS_T", ContKind.CATEGORICAL),
    SlotPosition("T_M_TYPE", ContKind.CATEGORICAL),
    SlotPosition("T_C_TYPE", ContKind.CATEGORICAL),
    SlotPosition("SIM_T", ContKind.FOURIER),
    SlotPosition("T_AGE", ContKind.TIME2VEC),
    SlotPosition("Q_SIZE", ContKind.COUNT_PLE),
    SlotPosition("BROKEN_N", ContKind.COUNT_PLE),
    SlotPosition("PROC_N", ContKind.COUNT_PLE),
    SlotPosition("N1_M_TYPE", ContKind.CATEGORICAL),
    SlotPosition("N1_C_TYPE", ContKind.CATEGORICAL),
    SlotPosition("N1_AGE", ContKind.TIME2VEC),
    SlotPosition("N2_M_TYPE", ContKind.CATEGORICAL),
    SlotPosition("N2_C_TYPE", ContKind.CATEGORICAL),
    SlotPosition("N2_AGE", ContKind.TIME2VEC),
)

_STREAM_LAYOUTS: dict[str, tuple[SlotPosition, ...]] = {
    "tech": TECH_SLOT_LAYOUT,
    "machine": MACHINE_SLOT_LAYOUT,
    "env": ENV_SLOT_LAYOUT,
}


def symlog(x: torch.Tensor) -> torch.Tensor:
    """Sign-preserving log squash: ``sign(x) * log1p(|x|)``.

    Stateless replacement for the learned PLE / Time2Vec / Fourier
    encoders on wide-ranging scalars (counts up to ~75k knowledge,
    sim times up to 5M): compresses magnitude without normaliser
    state that would drift between train and eval.
    """
    return torch.sign(x) * torch.log1p(torch.abs(x))


class SetObsFlattener(nn.Module):
    """Flatten the ``set`` observation dict into one fixed-width vector.

    Output layout (offsets exposed via :meth:`position_slice`)::

        [tech slot 0 | tech slot 1 | ... | machine slot 0 | ... | env]

    where each tech / machine slot is ``[mask_bit, features...]`` with
    the features zeroed on padded (mask == 0) slots, so the vector
    width is constant regardless of the real fleet size.  The env slot
    has no mask bit (it is always present).

    Parameters
    ----------
    vocab:
        The frozen set vocabulary — either a ``{token: id}`` dict or a
        path to a JSON file (the canonical
        ``run_configs/vocab/set_vocab.json`` wrapper with a ``vocab``
        field, or a bare mapping).
    max_techs / max_machines:
        Slot caps — must match the env config that produced the
        observations (defaults mirror ``GymEnvConfig``: 30 / 100).
    tech_slot_len / machine_slot_len / env_len:
        Per-slot triple counts (defaults mirror ``GymEnvConfig``:
        16 / 12 / 16).  The frozen layouts must fit inside them.
    """

    def __init__(
        self,
        vocab: Mapping[str, int] | str | Path,
        *,
        max_techs: int = 30,
        max_machines: int = 100,
        tech_slot_len: int = 16,
        machine_slot_len: int = 12,
        env_len: int = 16,
    ) -> None:
        super().__init__()
        if not isinstance(vocab, Mapping):
            vocab = self.load_vocab(vocab)
        self._vocab: dict[str, int] = {str(k): int(v) for k, v in vocab.items()}
        self._n_vocab = max(self._vocab.values()) + 1
        self.max_techs = int(max_techs)
        self.max_machines = int(max_machines)
        self._slot_lens = {
            "tech": int(tech_slot_len),
            "machine": int(machine_slot_len),
            "env": int(env_len),
        }
        self._n_slots = {"tech": self.max_techs, "machine": self.max_machines, "env": 1}
        # env slot carries no mask bit — it is always present.
        self._has_mask_bit = {"tech": True, "machine": True, "env": False}

        # Per-stream: ordered (position, spec, one-hot width, within-slot
        # feature offset) plus the id->local-index lookup tables.
        self._specs: dict[str, list[tuple[int, SlotPosition, int]]] = {}
        self._within: dict[str, dict[str, tuple[int, int]]] = {}
        self._cats: dict[str, dict[str, tuple[str, ...]]] = {}
        self._slot_width: dict[str, int] = {}
        for stream, layout in _STREAM_LAYOUTS.items():
            if len(layout) > self._slot_lens[stream]:
                msg = (
                    f"{stream} layout ({len(layout)} positions) exceeds the "
                    f"slot length {self._slot_lens[stream]}"
                )
                raise ValueError(msg)
            self._build_stream(stream, layout)

        # Stream base offsets inside the flat vector.
        tech_block = self.max_techs * self._slot_width["tech"]
        machine_block = self.max_machines * self._slot_width["machine"]
        self._stream_base = {
            "tech": 0,
            "machine": tech_block,
            "env": tech_block + machine_block,
        }
        self._out_dim = tech_block + machine_block + self._slot_width["env"]
        # Layout-drift guard: flipped by the first forward that carries
        # the observation's own ``*_cont_kinds`` channels (see
        # :meth:`_check_cont_kinds_once`).
        self._kinds_checked = False
        # Device probe: `.to(device)` moves buffers, inputs follow it.
        self.register_buffer(
            "_dev_probe", torch.zeros(1, dtype=torch.float32), persistent=False
        )

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------
    @staticmethod
    def load_vocab(path: str | Path) -> dict[str, int]:
        """Load a ``{token: id}`` dict from a vocab JSON file.

        Accepts both the canonical wrapper format
        (``{"version": ..., "n_tokens": ..., "vocab": {...}}``) and a
        bare token->id mapping.
        """
        data = json.loads(Path(path).read_text())
        mapping = data.get("vocab", data) if isinstance(data, dict) else data
        return {str(k): int(v) for k, v in mapping.items()}

    def _values_for_key(self, key: str) -> list[tuple[str, int]]:
        """All ``KEY=VALUE`` vocab tokens for ``key``, sorted by id."""
        prefix = f"{key}="
        values = sorted(
            ((tok, tid) for tok, tid in self._vocab.items() if tok.startswith(prefix)),
            key=lambda kv: kv[1],
        )
        if not values:
            msg = f"vocab has no '{prefix}*' tokens — wrong vocab file?"
            raise ValueError(msg)
        return values

    def _build_stream(self, stream: str, layout: tuple[SlotPosition, ...]) -> None:
        specs: list[tuple[int, SlotPosition, int]] = []
        within: dict[str, tuple[int, int]] = {}
        cats: dict[str, tuple[str, ...]] = {}
        offset = 0
        for pos, sp in enumerate(layout):
            if sp.kind == ContKind.CATEGORICAL:
                values = self._values_for_key(sp.key)
                width = len(values) + 1  # trailing OTHER bin (PAD/UNK/unseen)
                lut = torch.full((self._n_vocab,), len(values), dtype=torch.long)
                for local, (_tok, tid) in enumerate(values):
                    lut[tid] = local
                self.register_buffer(f"_lut_{stream}_{pos}", lut, persistent=False)
                cats[sp.key] = tuple(tok for tok, _tid in values)
            else:
                width = 1
            specs.append((pos, sp, width))
            within[sp.key] = (offset, width)
            offset += width
        # Expected ContKind per layout position — compared against the
        # observation's own kinds channel on the first forward.
        self.register_buffer(
            f"_kinds_{stream}",
            torch.tensor([sp.kind for sp in layout], dtype=torch.long),
            persistent=False,
        )
        self._specs[stream] = specs
        self._within[stream] = within
        self._cats[stream] = cats
        self._slot_width[stream] = offset + (1 if self._has_mask_bit[stream] else 0)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    @property
    def out_dim(self) -> int:
        """Width of the flattened vector (constant for a given config)."""
        return self._out_dim

    def categories(self, stream: str, key: str) -> tuple[str, ...]:
        """One-hot bin order (token strings) for a categorical position.

        The implicit OTHER bin sits at index ``len(categories(...))``.
        """
        return self._cats[stream][key]

    def position_slice(self, stream: str, key: str, slot: int = 0) -> slice:
        """Flat-vector slice holding ``key``'s features for slot ``slot``."""
        off, width = self._within[stream][key]
        start = (
            self._stream_base[stream]
            + slot * self._slot_width[stream]
            + (1 if self._has_mask_bit[stream] else 0)
            + off
        )
        return slice(start, start + width)

    def mask_bit_index(self, stream: str, slot: int) -> int:
        """Flat-vector index of the validity bit for slot ``slot``."""
        if not self._has_mask_bit[stream]:
            msg = f"stream '{stream}' has no mask bit"
            raise ValueError(msg)
        return self._stream_base[stream] + slot * self._slot_width[stream]

    # ------------------------------------------------------------------
    # Action mask
    # ------------------------------------------------------------------
    def extract_action_mask(self, obs: Mapping[str, Any]) -> np.ndarray:
        """Return the ``(max_techs,)`` boolean action mask from ``obs``.

        Prefers ``obs['action_mask']`` (the env's availability mask,
        already padded to ``max_techs`` in set mode — env.py:1807-1817)
        and falls back to ``tech_mask`` (valid-slot bits) when the env
        was configured without ``expose_action_mask``.  There is NO
        all-ones fallback: masking bugs must surface as errors, not as
        silently unmasked policies.
        """
        raw = obs.get("action_mask")
        if raw is None:
            raw = obs.get("tech_mask")
        if raw is None:
            msg = (
                "obs carries neither 'action_mask' nor 'tech_mask' — "
                "refusing the all-ones fallback"
            )
            raise KeyError(msg)
        mask = np.asarray(raw).reshape(-1).astype(bool)
        out = np.zeros(self.max_techs, dtype=bool)
        n = min(mask.shape[0], self.max_techs)
        out[:n] = mask[:n]
        if not out.any():
            msg = "action mask is all-zero — no valid action to take"
            raise ValueError(msg)
        return out

    # ------------------------------------------------------------------
    # Layout-drift guard
    # ------------------------------------------------------------------
    def _check_stream_kinds(
        self,
        stream: str,
        kinds: torch.Tensor,
        ids: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> None:
        """Compare one stream's ``cont_kinds`` against its frozen layout.

        ``kinds`` / ``ids`` are ``(..., L)``; only real positions take
        part — padded slots (``mask == 0``) and structural PAD tokens
        carry ``CATEGORICAL`` by construction (``_set_obs::_pack_slot``)
        and would raise spurious mismatches.
        """
        expect = getattr(self, f"_kinds_{stream}")
        n = int(expect.shape[0])
        k = kinds.reshape(-1, kinds.shape[-1])
        i = ids.reshape(-1, ids.shape[-1])
        if k.shape != i.shape or k.shape[1] < n:
            msg = (
                "KataEnv._set_obs and mlp_encoder layouts have diverged: "
                f"'{stream}' stream carries cont_kinds of shape "
                f"{tuple(kinds.shape)} against token ids of shape "
                f"{tuple(ids.shape)} — the frozen layout needs {n} "
                "positions per slot"
            )
            raise ValueError(msg)
        live = i[:, :n] != PAD_ID
        if mask is not None:
            live = live & (mask.reshape(-1) > 0).unsqueeze(-1)
        bad = live & (k[:, :n] != expect)
        if not bool(bad.any()):
            return
        pos = int(bad.any(dim=0).nonzero()[0].item())
        row = int(bad[:, pos].nonzero()[0].item())
        sp = _STREAM_LAYOUTS[stream][pos]
        msg = (
            "KataEnv._set_obs and mlp_encoder layouts have diverged: "
            f"'{stream}' stream position {pos} ('{sp.key}') carries "
            f"cont_kind {_kind_repr(int(k[row, pos].item()))} but the "
            f"frozen layout expects {_kind_repr(sp.kind)} — re-sync the "
            "*_SLOT_LAYOUT tables in agents/networks/mlp_encoder.py with "
            "the emission order of KataEnv._set_obs"
        )
        raise ValueError(msg)

    def _check_cont_kinds_once(
        self,
        obs: Mapping[str, Any],
        streams: Sequence[tuple[str, torch.Tensor, torch.Tensor | None]],
    ) -> None:
        """Validate the obs' own kinds channels against the layouts, once.

        The kind dispatch in :meth:`_stream_feats` follows the frozen
        layouts, not the observation — an inserted / reordered
        ``_set_obs`` emission would shift every downstream column
        without any error.  One vectorised comparison per stream on the
        first forward that carries ``*_cont_kinds`` catches that at
        ~zero steady-state cost; observations without the channel (hand-
        built fakes, trimmed replay buffers) simply leave the flag unset
        so a later full observation still gets checked.
        """
        checked = 0
        for stream, ids, mask in streams:
            raw = obs.get(f"{stream}_cont_kinds")
            if raw is None:
                continue
            self._check_stream_kinds(
                stream, self._tensor(raw, torch.long), ids, mask
            )
            checked += 1
        if checked == len(streams):
            self._kinds_checked = True

    # ------------------------------------------------------------------
    # Flattening
    # ------------------------------------------------------------------
    def _tensor(self, x: Any, dtype: torch.dtype) -> torch.Tensor:
        return torch.as_tensor(
            np.asarray(x), dtype=dtype, device=self._dev_probe.device
        )

    @staticmethod
    def _get(obs: Mapping[str, Any], key: str, alias: str) -> Any:
        v = obs.get(key)
        if v is None:
            v = obs.get(alias)
        if v is None:
            msg = f"set observation is missing '{key}'"
            raise KeyError(msg)
        return v

    def _stream_feats(
        self,
        stream: str,
        ids: torch.Tensor,
        vals: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Encode one stream: ``(B, N, L)`` ids/values → ``(B, N*w)``."""
        chunks: list[torch.Tensor] = []
        for pos, sp, width in self._specs[stream]:
            if sp.kind == ContKind.CATEGORICAL:
                lut = getattr(self, f"_lut_{stream}_{pos}")
                raw = ids[..., pos]
                local = lut[raw.clamp(0, self._n_vocab - 1)]
                # Ids outside the vocab (shouldn't happen, but never
                # alias a real bin) route to the OTHER bin.
                oob = (raw < 0) | (raw >= self._n_vocab)
                local = torch.where(oob, torch.full_like(local, width - 1), local)
                chunks.append(F.one_hot(local, num_classes=width).to(torch.float32))
            else:
                v = vals[..., pos]
                if sp.kind != ContKind.RATIO_PLE:
                    # COUNT / TIME / FOURIER (sim-time) scalars span
                    # orders of magnitude — squash; RATIO is already
                    # bounded, keep it raw.
                    v = symlog(v)
                chunks.append(v.unsqueeze(-1))
        feats = torch.cat(chunks, dim=-1)
        if mask is not None:
            mbit = mask.to(torch.float32).clamp(0.0, 1.0).unsqueeze(-1)
            # Zero padded-slot features so stale values in padded rows
            # can never leak; keep the mask bit itself as a feature.
            feats = torch.cat([mbit, feats * mbit], dim=-1)
        return feats.reshape(feats.shape[0], -1)

    def forward(self, obs: Mapping[str, Any]) -> torch.Tensor:
        """Flatten one obs dict (or a batched one) to ``(out_dim,)`` /
        ``(B, out_dim)`` float32.

        Batched inputs are detected by an extra leading dim on
        ``tech_token_ids`` (3-D instead of 2-D).  The kind dispatch
        follows the frozen layouts rather than ``*_cont_kinds``; the
        kinds channel is instead cross-checked against them once (both
        paths), so a schema drift raises rather than silently shifting
        every column.
        """
        tech_ids = self._tensor(
            self._get(obs, "tech_token_ids", "tech_tokens"), torch.long
        )
        tech_vals = self._tensor(obs["tech_cont_values"], torch.float32)
        tech_mask = self._tensor(obs["tech_mask"], torch.float32)
        mach_ids = self._tensor(
            self._get(obs, "machine_token_ids", "machine_tokens"), torch.long
        )
        mach_vals = self._tensor(obs["machine_cont_values"], torch.float32)
        mach_mask = self._tensor(obs["machine_mask"], torch.float32)
        env_ids = self._tensor(
            self._get(obs, "env_token_ids", "env_tokens"), torch.long
        )
        env_vals = self._tensor(obs["env_cont_values"], torch.float32)

        batched = tech_ids.dim() == 3
        if not batched:
            tech_ids, tech_vals, tech_mask = (
                tech_ids[None], tech_vals[None], tech_mask[None]
            )
            mach_ids, mach_vals, mach_mask = (
                mach_ids[None], mach_vals[None], mach_mask[None]
            )
            env_ids, env_vals = env_ids[None], env_vals[None]

        if not self._kinds_checked:
            self._check_cont_kinds_once(
                obs,
                (
                    ("tech", tech_ids, tech_mask),
                    ("machine", mach_ids, mach_mask),
                    ("env", env_ids, None),
                ),
            )

        parts = [
            self._stream_feats("tech", tech_ids, tech_vals, tech_mask),
            self._stream_feats("machine", mach_ids, mach_vals, mach_mask),
            # Env stream: single always-valid slot, no mask bit.
            self._stream_feats(
                "env", env_ids.unsqueeze(1), env_vals.unsqueeze(1), None
            ),
        ]
        out = torch.cat(parts, dim=-1)
        return out if batched else out.squeeze(0)

    # Explicit alias — reads better at call sites that hold a plain
    # flattener rather than a composed module.
    def flatten(self, obs: Mapping[str, Any]) -> torch.Tensor:
        return self(obs)


# ---------------------------------------------------------------------------
# MLP heads
# ---------------------------------------------------------------------------


class MLPTrunk(nn.Module):
    """Linear-LayerNorm-ReLU stack shared by the baseline heads."""

    def __init__(
        self, in_dim: int, hidden_sizes: Sequence[int] = (512, 512)
    ) -> None:
        super().__init__()
        if not hidden_sizes:
            msg = "hidden_sizes must contain at least one layer width"
            raise ValueError(msg)
        layers: list[nn.Module] = []
        d = int(in_dim)
        for h in hidden_sizes:
            layers += [nn.Linear(d, int(h)), nn.LayerNorm(int(h)), nn.ReLU()]
            d = int(h)
        self.net = nn.Sequential(*layers)
        self.out_dim = d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MLPActorCritic(nn.Module):
    """Shared-trunk policy logits + value scalar (PPO-MLP baseline)."""

    def __init__(
        self,
        in_dim: int,
        n_actions: int,
        hidden_sizes: Sequence[int] = (512, 512),
    ) -> None:
        super().__init__()
        self.trunk = MLPTrunk(in_dim, hidden_sizes)
        self.policy_head = nn.Linear(self.trunk.out_dim, int(n_actions))
        self.value_head = nn.Linear(self.trunk.out_dim, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``(..., in_dim)`` → logits ``(..., n_actions)``, value ``(...)``."""
        h = self.trunk(x)
        return self.policy_head(h), self.value_head(h).squeeze(-1)


class MLPPolicy(nn.Module):
    """Logits-only head (REINFORCE-MLP baseline)."""

    def __init__(
        self,
        in_dim: int,
        n_actions: int,
        hidden_sizes: Sequence[int] = (512, 512),
    ) -> None:
        super().__init__()
        self.trunk = MLPTrunk(in_dim, hidden_sizes)
        self.policy_head = nn.Linear(self.trunk.out_dim, int(n_actions))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.policy_head(self.trunk(x))


class MLPQNetwork(nn.Module):
    """Per-action Q-value head (DQN-MLP baseline)."""

    def __init__(
        self,
        in_dim: int,
        n_actions: int,
        hidden_sizes: Sequence[int] = (512, 512),
    ) -> None:
        super().__init__()
        self.trunk = MLPTrunk(in_dim, hidden_sizes)
        self.q_head = nn.Linear(self.trunk.out_dim, int(n_actions))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.q_head(self.trunk(x))


__all__ = [
    "ENV_SLOT_LAYOUT",
    "MACHINE_SLOT_LAYOUT",
    "MLPActorCritic",
    "MLPPolicy",
    "MLPQNetwork",
    "MLPTrunk",
    "SetObsFlattener",
    "SlotPosition",
    "TECH_SLOT_LAYOUT",
    "symlog",
]
