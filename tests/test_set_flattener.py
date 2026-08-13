"""Tests for the SetObsFlattener and the plain-MLP baseline heads.

Constructs everything directly (tiny dims, cpu) against the canonical
frozen vocab ``run_configs/vocab/set_vocab.json`` — no env, no runner
registry.  Fake observations are assembled position-by-position from
the frozen slot layouts so the tests double as a pin on the
``KataEnv._set_obs`` emission order.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from agents.networks.continuous_features import ContKind
from agents.networks.mlp_encoder import (
    ENV_SLOT_LAYOUT,
    MACHINE_SLOT_LAYOUT,
    MLPActorCritic,
    MLPPolicy,
    MLPQNetwork,
    MLPTrunk,
    SetObsFlattener,
    TECH_SLOT_LAYOUT,
)

VOCAB_PATH = (
    Path(__file__).resolve().parents[1] / "run_configs" / "vocab" / "set_vocab.json"
)
VOCAB = SetObsFlattener.load_vocab(VOCAB_PATH)
UNK_ID = 1

# Token spelling per continuous kind (kata.env._SetEmitter).
_CONT_FMT = {
    ContKind.RATIO_PLE: "<RATIO:{}>",
    ContKind.COUNT_PLE: "<COUNT:{}>",
    ContKind.TIME2VEC: "<TIME:{}>",
    ContKind.FOURIER: "<FOUR:{}>",
}

# Neutral per-slot values used unless a test overrides them.
_TECH_DEFAULTS = {"TEMPLATE": "default", "BUSY": "F", "DISRUPT": "F"}
_MACH_DEFAULTS = {
    "M_TYPE": "CNC",
    "BROKEN": "F",
    "PROC": "F",
    "IS_CURRENT": "F",
    "CUR_COMP": "NONE",
}
_ENV_DEFAULTS = {
    "HAS_T": "T",
    "T_M_TYPE": "CNC",
    "T_C_TYPE": "motor",
    "N1_M_TYPE": "NONE",
    "N1_C_TYPE": "NONE",
    "N2_M_TYPE": "NONE",
    "N2_C_TYPE": "NONE",
}


def _fill_slot(ids_row, vals_row, kinds_row, layout, defaults, overrides=None):
    """Write one slot's (token id, cont value, kind) triples in layout order.

    Mirrors ``KataEnv._set_obs``: positions past the layout keep the
    PAD id / 0.0 value / CATEGORICAL kind they were zero-initialised
    with (``_pack_slot``).
    """
    values = dict(defaults)
    values.update(overrides or {})
    for pos, sp in enumerate(layout):
        if sp.kind == ContKind.CATEGORICAL:
            tok = f"{sp.key}={values.get(sp.key)}"
            ids_row[pos] = VOCAB.get(tok, UNK_ID)
        else:
            ids_row[pos] = VOCAB[_CONT_FMT[sp.kind].format(sp.key)]
            vals_row[pos] = float(values.get(sp.key, 0.0))
        kinds_row[pos] = int(sp.kind)


def _make_obs(
    n_techs=2,
    n_machines=2,
    *,
    max_t=4,
    max_m=3,
    lt=16,
    lm=12,
    le=16,
    tech_overrides=None,
    mach_overrides=None,
    env_overrides=None,
):
    """Build a fake ``set`` observation with the env's shapes and dtypes.

    ``*_overrides`` are ``{slot_index: {KEY: value}}`` (env stream:
    plain ``{KEY: value}``); categorical keys take value strings,
    continuous keys floats.
    """
    obs = {
        "tech_token_ids": np.zeros((max_t, lt), dtype=np.int64),
        "tech_cont_values": np.zeros((max_t, lt), dtype=np.float32),
        "tech_cont_kinds": np.zeros((max_t, lt), dtype=np.int8),
        "tech_mask": np.zeros(max_t, dtype=np.int8),
        "machine_token_ids": np.zeros((max_m, lm), dtype=np.int64),
        "machine_cont_values": np.zeros((max_m, lm), dtype=np.float32),
        "machine_cont_kinds": np.zeros((max_m, lm), dtype=np.int8),
        "machine_mask": np.zeros(max_m, dtype=np.int8),
        "env_token_ids": np.zeros(le, dtype=np.int64),
        "env_cont_values": np.zeros(le, dtype=np.float32),
        "env_cont_kinds": np.zeros(le, dtype=np.int8),
    }
    for i in range(n_techs):
        _fill_slot(
            obs["tech_token_ids"][i],
            obs["tech_cont_values"][i],
            obs["tech_cont_kinds"][i],
            TECH_SLOT_LAYOUT,
            _TECH_DEFAULTS,
            (tech_overrides or {}).get(i),
        )
        obs["tech_mask"][i] = 1
    for i in range(n_machines):
        _fill_slot(
            obs["machine_token_ids"][i],
            obs["machine_cont_values"][i],
            obs["machine_cont_kinds"][i],
            MACHINE_SLOT_LAYOUT,
            _MACH_DEFAULTS,
            (mach_overrides or {}).get(i),
        )
        obs["machine_mask"][i] = 1
    _fill_slot(
        obs["env_token_ids"],
        obs["env_cont_values"],
        obs["env_cont_kinds"],
        ENV_SLOT_LAYOUT,
        _ENV_DEFAULTS,
        env_overrides,
    )
    return obs


def _tiny_flattener(**kw):
    kw.setdefault("max_techs", 4)
    kw.setdefault("max_machines", 3)
    return SetObsFlattener(VOCAB, **kw)


# ---------------------------------------------------------------------------
# SetObsFlattener
# ---------------------------------------------------------------------------


class TestSetObsFlattener:
    def test_out_dim_pinned_at_env_defaults(self):
        """Full-size flattener against the frozen 152-token vocab.

        tech slot  = 1 mask + 9 TEMPLATE + 3+3 bools + 11 scalars = 27
        mach slot  = 1 mask + 15 M_TYPE + 3*3 bools + 12 CUR_COMP + 6 = 43
        env slot   = 3 + 16 + 12 + 5 scalars + (16+12+1)*2         = 94
        """
        flat = SetObsFlattener(VOCAB_PATH)
        assert flat.out_dim == 30 * 27 + 100 * 43 + 94 == 5204

    def test_flatten_shape_dtype_finite(self):
        flat = _tiny_flattener()
        out = flat(_make_obs())
        assert out.shape == (flat.out_dim,)
        assert out.dtype == torch.float32
        assert torch.isfinite(out).all()

    def test_busy_flip_changes_output(self):
        """The D11 pin: boolean state lives only in the token id."""
        flat = _tiny_flattener()
        out_f = flat(_make_obs(tech_overrides={0: {"BUSY": "F"}}))
        out_t = flat(_make_obs(tech_overrides={0: {"BUSY": "T"}}))
        assert not torch.equal(out_f, out_t)
        sl = flat.position_slice("tech", "BUSY", slot=0)
        # Bin order follows vocab ids (BUSY=T < BUSY=F) + OTHER bin.
        assert flat.categories("tech", "BUSY") == ("BUSY=T", "BUSY=F")
        assert out_t[sl].tolist() == [1.0, 0.0, 0.0]
        assert out_f[sl].tolist() == [0.0, 1.0, 0.0]
        # Everything outside the flipped position is untouched.
        diff = (out_f != out_t).nonzero().flatten().tolist()
        assert diff == list(range(sl.start, sl.stop))[:2]

    def test_legacy_unk_bools_collapse_to_other_bin(self):
        """Legacy TRUE/FALSE spellings are <UNK> in the frozen vocab —
        both land in the OTHER bin, i.e. the flattener reproduces the
        D11 information loss for legacy ids instead of aliasing T or F."""
        flat = _tiny_flattener()
        obs = _make_obs()
        pos = [sp.key for sp in TECH_SLOT_LAYOUT].index("BUSY")
        obs["tech_token_ids"][0][pos] = VOCAB.get("BUSY=TRUE", UNK_ID)  # -> UNK
        out = flat(obs)
        assert out[flat.position_slice("tech", "BUSY", 0)].tolist() == [0.0, 0.0, 1.0]

    def test_pinned_one_hot_placement(self):
        """Known token ids land in exactly the expected one-hot bins."""
        flat = _tiny_flattener()
        obs = _make_obs(
            tech_overrides={1: {"TEMPLATE": "junior"}},
            mach_overrides={0: {"M_TYPE": "Lathe", "BROKEN": "T"}},
        )
        out = flat(obs)
        # TEMPLATE=junior on tech slot 1.
        cats = flat.categories("tech", "TEMPLATE")
        expected = [0.0] * (len(cats) + 1)
        expected[cats.index("TEMPLATE=junior")] = 1.0
        assert out[flat.position_slice("tech", "TEMPLATE", slot=1)].tolist() == expected
        # M_TYPE=Lathe on machine slot 0.
        cats = flat.categories("machine", "M_TYPE")
        expected = [0.0] * (len(cats) + 1)
        expected[cats.index("M_TYPE=Lathe")] = 1.0
        assert out[flat.position_slice("machine", "M_TYPE", slot=0)].tolist() == expected
        assert out[flat.position_slice("machine", "BROKEN", slot=0)].tolist() == [
            1.0, 0.0, 0.0,
        ]
        # Env stream: HAS_T=T (no mask bit on the env slot).
        assert out[flat.position_slice("env", "HAS_T")].tolist() == [1.0, 0.0, 0.0]

    def test_unseen_category_value_routes_to_other_bin(self):
        """M_TYPE=NONE is not in the frozen vocab (machine slots always
        have a real type) — it must hit OTHER, never alias a real type."""
        flat = _tiny_flattener()
        out = flat(_make_obs(mach_overrides={0: {"M_TYPE": "NONE"}}))
        sl = flat.position_slice("machine", "M_TYPE", slot=0)
        vec = out[sl]
        assert vec[-1].item() == 1.0
        assert vec[:-1].sum().item() == 0.0

    def test_symlog_on_counts_times_raw_ratios(self):
        flat = _tiny_flattener()
        obs = _make_obs(
            tech_overrides={0: {"ASSIGNS": 100.0, "FATIGUE": 0.37, "ETA": 250.0}},
            env_overrides={"SIM_T": 1.0e6},
        )
        out = flat(obs)
        assert np.isclose(
            out[flat.position_slice("tech", "ASSIGNS", 0)].item(), np.log1p(100.0)
        )
        assert np.isclose(
            out[flat.position_slice("tech", "FATIGUE", 0)].item(), 0.37
        )
        assert np.isclose(
            out[flat.position_slice("tech", "ETA", 0)].item(), np.log1p(250.0)
        )
        assert np.isclose(
            out[flat.position_slice("env", "SIM_T")].item(), np.log1p(1.0e6)
        )

    def test_constant_width_and_padded_slots_zeroed(self):
        """Width must not depend on fleet size (2 vs 30 techs), and
        padded slots contribute nothing — even with garbage values."""
        flat = SetObsFlattener(VOCAB)  # env defaults: 30 / 100
        kw = dict(max_t=30, max_m=100)
        small = _make_obs(n_techs=2, n_machines=5, **kw)
        # Garbage in a masked-out row must not leak into the features.
        small["tech_cont_values"][7, :] = 999.0
        full = _make_obs(n_techs=30, n_machines=5, **kw)
        out_s, out_f = flat(small), flat(full)
        assert out_s.shape == out_f.shape == (flat.out_dim,)
        for slot in (2, 7, 29):
            assert out_s[flat.mask_bit_index("tech", slot)].item() == 0.0
            row = out_s[
                flat.mask_bit_index("tech", slot) + 1
                : flat.mask_bit_index("tech", slot) + flat._slot_width["tech"]
            ]
            assert row.abs().sum().item() == 0.0
        assert out_s[flat.mask_bit_index("tech", 0)].item() == 1.0
        assert out_f[flat.mask_bit_index("tech", 29)].item() == 1.0

    def test_batched_matches_single(self):
        flat = _tiny_flattener()
        a = _make_obs(tech_overrides={0: {"BUSY": "T", "FATIGUE": 0.5}})
        b = _make_obs(n_techs=3, mach_overrides={1: {"BROKEN": "T"}})
        batch = {k: np.stack([a[k], b[k]]) for k in a}
        out = flat(batch)
        assert out.shape == (2, flat.out_dim)
        assert torch.equal(out[0], flat(a))
        assert torch.equal(out[1], flat(b))

    def test_action_mask_prefers_env_mask(self):
        flat = _tiny_flattener()
        obs = _make_obs()
        obs["action_mask"] = np.array([1, 0, 1, 0], dtype=np.int8)
        mask = flat.extract_action_mask(obs)
        assert mask.dtype == bool and mask.shape == (4,)
        assert mask.tolist() == [True, False, True, False]

    def test_action_mask_falls_back_to_tech_mask_never_all_ones(self):
        flat = _tiny_flattener()
        obs = _make_obs(n_techs=2)  # no action_mask key
        mask = flat.extract_action_mask(obs)
        assert mask.tolist() == [True, True, False, False]

    def test_action_mask_missing_or_empty_raises(self):
        flat = _tiny_flattener()
        with pytest.raises(KeyError):
            flat.extract_action_mask({})
        obs = _make_obs(n_techs=0)
        with pytest.raises(ValueError):
            flat.extract_action_mask(obs)


# ---------------------------------------------------------------------------
# Layout-drift guard (the obs' own *_cont_kinds vs the frozen layouts)
# ---------------------------------------------------------------------------


class TestContKindsGuard:
    def test_env_shaped_obs_passes_and_caches_the_flag(self):
        """A faithful obs validates once, then the guard steps aside."""
        flat = _tiny_flattener()
        assert flat._kinds_checked is False
        flat(_make_obs())
        assert flat._kinds_checked is True

    def test_second_forward_skips_validation(self):
        """Post-check corruption is not re-detected — the check is
        once-per-module, not per-forward (cost pin)."""
        flat = _tiny_flattener()
        flat(_make_obs())  # arms the cached flag
        obs = _make_obs()
        pos = [sp.key for sp in TECH_SLOT_LAYOUT].index("FATIGUE")
        obs["tech_cont_kinds"][0][pos] = ContKind.TIME2VEC
        out = flat(obs)  # must NOT raise
        assert torch.isfinite(out).all()

    def test_swapped_kind_raises_naming_the_position(self):
        """The D-drift pin: a reordered _set_obs emission shows up as a
        kind mismatch instead of silently shifting every column."""
        flat = _tiny_flattener()
        obs = _make_obs()
        pos = [sp.key for sp in TECH_SLOT_LAYOUT].index("FATIGUE")
        obs["tech_cont_kinds"][0][pos] = ContKind.COUNT_PLE  # layout: RATIO_PLE
        with pytest.raises(ValueError) as excinfo:
            flat(obs)
        msg = str(excinfo.value)
        assert "KataEnv._set_obs and mlp_encoder layouts have diverged" in msg
        assert "'tech'" in msg and f"position {pos}" in msg and "FATIGUE" in msg
        # Both kinds are named — the one seen and the one expected.
        assert "COUNT_PLE" in msg and "RATIO_PLE" in msg
        assert flat._kinds_checked is False

    def test_machine_and_env_streams_are_checked_too(self):
        for stream, layout, key, kinds_key, bad in (
            ("machine", MACHINE_SLOT_LAYOUT, "DOWNTIME", "machine_cont_kinds",
             ContKind.CATEGORICAL),
            ("env", ENV_SLOT_LAYOUT, "SIM_T", "env_cont_kinds",
             ContKind.TIME2VEC),
        ):
            flat = _tiny_flattener()
            obs = _make_obs()
            pos = [sp.key for sp in layout].index(key)
            row = obs[kinds_key][0] if kinds_key != "env_cont_kinds" else obs[kinds_key]
            row[pos] = bad
            with pytest.raises(ValueError) as excinfo:
                flat(obs)
            assert f"'{stream}' stream position {pos} ('{key}')" in str(excinfo.value)

    def test_batched_path_validates(self):
        flat = _tiny_flattener()
        a, b = _make_obs(), _make_obs()
        pos = [sp.key for sp in TECH_SLOT_LAYOUT].index("ETA")
        b["tech_cont_kinds"][1][pos] = ContKind.COUNT_PLE
        batch = {k: np.stack([a[k], b[k]]) for k in a}
        with pytest.raises(ValueError, match=r"'tech' stream position \d+ \('ETA'\)"):
            flat(batch)

    def test_padded_slots_and_pad_positions_are_exempt(self):
        """Masked-out slots and structural PAD positions carry
        CATEGORICAL by construction — garbage there must not raise."""
        flat = _tiny_flattener()
        obs = _make_obs(n_techs=2, n_machines=1)
        obs["tech_cont_kinds"][3, :] = ContKind.FOURIER  # masked-out slot
        obs["machine_cont_kinds"][2, :] = ContKind.TIME2VEC
        # Trailing PAD positions of a *real* slot (layout is shorter
        # than the slot length).
        obs["tech_cont_kinds"][0, len(TECH_SLOT_LAYOUT):] = ContKind.COUNT_PLE
        obs["env_cont_kinds"][len(ENV_SLOT_LAYOUT):] = ContKind.RATIO_PLE
        out = flat(obs)
        assert torch.isfinite(out).all()
        assert flat._kinds_checked is True

    def test_obs_without_kinds_channel_is_accepted_and_stays_unchecked(self):
        """Hand-built / trimmed observations (no kinds channel) still
        flatten, and leave the flag unset so a full obs is checked."""
        flat = _tiny_flattener()
        obs = {k: v for k, v in _make_obs().items() if not k.endswith("_cont_kinds")}
        flat(obs)
        assert flat._kinds_checked is False
        with pytest.raises(ValueError):
            bad = _make_obs()
            bad["env_cont_kinds"][0] = ContKind.COUNT_PLE  # HAS_T is categorical
            flat(bad)

    def test_short_kinds_channel_raises(self):
        """A truncated kinds row cannot be aligned — that is drift too."""
        flat = _tiny_flattener()
        obs = _make_obs()
        obs["env_cont_kinds"] = obs["env_cont_kinds"][: len(ENV_SLOT_LAYOUT) - 1]
        with pytest.raises(ValueError, match="diverged"):
            flat(obs)


# ---------------------------------------------------------------------------
# MLP heads
# ---------------------------------------------------------------------------


class TestMLPHeads:
    def test_trunk_structure_and_out_dim(self):
        trunk = MLPTrunk(8, hidden_sizes=(32, 16))
        # Linear-LayerNorm-ReLU per hidden layer.
        kinds = [type(m) for m in trunk.net]
        assert kinds == [
            torch.nn.Linear, torch.nn.LayerNorm, torch.nn.ReLU,
        ] * 2
        assert trunk.out_dim == 16
        assert trunk(torch.randn(5, 8)).shape == (5, 16)

    def test_actor_critic_shapes(self):
        torch.manual_seed(0)
        net = MLPActorCritic(8, n_actions=3, hidden_sizes=(16, 16))
        logits, value = net(torch.randn(5, 8))
        assert logits.shape == (5, 3)
        assert value.shape == (5,)
        # Unbatched input works too (heads act on the last dim only).
        logits, value = net(torch.randn(8))
        assert logits.shape == (3,)
        assert value.shape == ()

    def test_policy_and_q_shapes(self):
        assert MLPPolicy(8, 3, hidden_sizes=(16,))(torch.randn(4, 8)).shape == (4, 3)
        assert MLPQNetwork(8, 3, hidden_sizes=(16,))(torch.randn(4, 8)).shape == (4, 3)

    def test_gradients_flow_end_to_end(self):
        """Flattener output feeds the trunk; loss reaches every param."""
        flat = _tiny_flattener()
        net = MLPActorCritic(flat.out_dim, n_actions=4, hidden_sizes=(16, 16))
        logits, value = net(flat(_make_obs()))
        (logits.sum() + value.sum()).backward()
        assert all(
            p.grad is not None and torch.isfinite(p.grad).all()
            for p in net.parameters()
        )
