"""Tests for the critic-free GRPO baseline (:class:`GRPOMLPAgent`).

Everything is constructed directly — tiny slot caps, tiny MLP, cpu — on
the canonical frozen vocab ``run_configs/vocab/set_vocab.json``; no env,
no runner registry, no hydra.  Fake ``set`` observations are assembled
position-by-position from the frozen slot layouts, the same way
``tests/test_set_flattener.py`` does.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from agents.grpo.grpo_mlp import GRPOMLPAgent
from agents.networks.continuous_features import ContKind
from agents.networks.mlp_encoder import (
    ENV_SLOT_LAYOUT,
    MACHINE_SLOT_LAYOUT,
    SetObsFlattener,
    TECH_SLOT_LAYOUT,
)

VOCAB_PATH = (
    Path(__file__).resolve().parents[1] / "run_configs" / "vocab" / "set_vocab.json"
)
VOCAB = SetObsFlattener.load_vocab(VOCAB_PATH)
UNK_ID = 1

MAX_T = 4
MAX_M = 3

# Token spelling per continuous kind (kata.env._SetEmitter).
_CONT_FMT = {
    ContKind.RATIO_PLE: "<RATIO:{}>",
    ContKind.COUNT_PLE: "<COUNT:{}>",
    ContKind.TIME2VEC: "<TIME:{}>",
    ContKind.FOURIER: "<FOUR:{}>",
}

_TECH_DEFAULTS = {"TEMPLATE": "junior", "BUSY": "F", "DISRUPT": "F"}
_MACH_DEFAULTS = {
    "M_TYPE": "CNC",
    "BROKEN": "F",
    "PROC": "F",
    "IS_CURRENT": "F",
    "CUR_COMP": "motor",
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


# ---------------------------------------------------------------------------
# Fake observations
# ---------------------------------------------------------------------------


def _fill_slot(ids_row, vals_row, kinds_row, layout, defaults, rng=None):
    """Write one slot's (token id, cont value, kind) triples in layout order.

    Mirrors ``KataEnv._set_obs``: the kinds channel is emitted alongside
    the ids, and the flattener's layout-drift guard checks it.
    """
    for pos, sp in enumerate(layout):
        if sp.kind == ContKind.CATEGORICAL:
            ids_row[pos] = VOCAB.get(f"{sp.key}={defaults.get(sp.key)}", UNK_ID)
        else:
            ids_row[pos] = VOCAB[_CONT_FMT[sp.kind].format(sp.key)]
            vals_row[pos] = float(rng.uniform(0.0, 5.0)) if rng is not None else 0.5
        kinds_row[pos] = int(sp.kind)


def _make_obs(n_techs=3, n_machines=2, *, allowed=None, rng=None):
    """Build a fake ``set`` observation with the env's shapes and dtypes.

    ``allowed`` selects the technician slots flagged available in
    ``action_mask`` (default: every present technician).
    """
    obs = {
        "tech_token_ids": np.zeros((MAX_T, 16), dtype=np.int64),
        "tech_cont_values": np.zeros((MAX_T, 16), dtype=np.float32),
        "tech_cont_kinds": np.zeros((MAX_T, 16), dtype=np.int8),
        "tech_mask": np.zeros(MAX_T, dtype=np.int8),
        "machine_token_ids": np.zeros((MAX_M, 12), dtype=np.int64),
        "machine_cont_values": np.zeros((MAX_M, 12), dtype=np.float32),
        "machine_cont_kinds": np.zeros((MAX_M, 12), dtype=np.int8),
        "machine_mask": np.zeros(MAX_M, dtype=np.int8),
        "env_token_ids": np.zeros(16, dtype=np.int64),
        "env_cont_values": np.zeros(16, dtype=np.float32),
        "env_cont_kinds": np.zeros(16, dtype=np.int8),
    }
    for i in range(n_techs):
        _fill_slot(
            obs["tech_token_ids"][i],
            obs["tech_cont_values"][i],
            obs["tech_cont_kinds"][i],
            TECH_SLOT_LAYOUT,
            _TECH_DEFAULTS,
            rng,
        )
        obs["tech_mask"][i] = 1
    for i in range(n_machines):
        _fill_slot(
            obs["machine_token_ids"][i],
            obs["machine_cont_values"][i],
            obs["machine_cont_kinds"][i],
            MACHINE_SLOT_LAYOUT,
            _MACH_DEFAULTS,
            rng,
        )
        obs["machine_mask"][i] = 1
    _fill_slot(
        obs["env_token_ids"], obs["env_cont_values"], obs["env_cont_kinds"],
        ENV_SLOT_LAYOUT, _ENV_DEFAULTS, rng,
    )
    mask = np.zeros(MAX_T, dtype=bool)
    for i in (range(n_techs) if allowed is None else allowed):
        mask[i] = True
    obs["action_mask"] = mask
    return obs


# ---------------------------------------------------------------------------
# Agent / rollout helpers
# ---------------------------------------------------------------------------


def _make_agent(**overrides) -> GRPOMLPAgent:
    defaults = dict(
        n_actions=MAX_T,
        vocab=VOCAB,
        hidden_sizes=(16, 16),
        max_techs=MAX_T,
        max_machines=MAX_M,
        env_length=16,
        tech_slot_length=16,
        machine_slot_length=12,
        group_size=8,
        n_epochs=2,
        minibatch_size=64,
        lr=1e-3,
        total_updates=8,
        warmup_updates=1,
        device="cpu",
        seed=0,
    )
    defaults.update(overrides)
    return GRPOMLPAgent(**defaults)


def _run_episode(
    agent: GRPOMLPAgent,
    *,
    n_steps: int = 4,
    reward: float = 1.0,
    n_techs: int = 3,
    allowed=None,
    rng=None,
) -> float:
    """Drive one complete episode through the agent (serial-loop order)."""
    agent.on_episode_start()
    total = 0.0
    for t in range(n_steps):
        obs = _make_obs(n_techs=n_techs, allowed=allowed, rng=rng)
        action = agent.select_action(obs)
        next_obs = _make_obs(n_techs=n_techs, allowed=allowed, rng=rng)
        agent.observe_transition(
            obs, action, reward, next_obs, t == n_steps - 1, False, {}
        )
        total += reward
    agent.on_episode_end(total)
    return total


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_action_space_must_match_the_tech_cap(self):
        with pytest.raises(ValueError, match="n_actions == max_techs"):
            _make_agent(n_actions=3)

    def test_singleton_group_is_rejected(self):
        with pytest.raises(ValueError, match="group_size must be >= 2"):
            _make_agent(group_size=1)

    def test_flat_width_matches_the_flattener(self):
        agent = _make_agent()
        flat = agent.flattener(_make_obs())
        assert flat.shape == (agent.obs_dim,)
        assert agent.net.trunk.net[0].in_features == agent.obs_dim


# ---------------------------------------------------------------------------
# Group accumulation
# ---------------------------------------------------------------------------


class TestGroupAccumulation:
    def test_update_is_a_no_op_until_the_group_is_full(self):
        """7 episodes buffer silently; the 8th triggers the group update."""
        agent = _make_agent(group_size=8)
        rng = np.random.default_rng(0)
        for i in range(7):
            _run_episode(agent, reward=float(i), rng=rng)
            assert agent.update() == {}
            assert len(agent._group) == i + 1
        _run_episode(agent, reward=7.0, rng=rng)
        metrics = agent.update()
        assert metrics, "the 8th episode must complete the group"
        for key in (
            "loss", "pg_loss", "entropy", "approx_kl", "clip_fraction",
            "lr", "group_size", "group_steps", "group_return_mean",
            "group_return_std",
        ):
            assert key in metrics
            assert np.isfinite(metrics[key])
        assert metrics["group_size"] == 8.0
        # Group consumed, buffers empty.
        assert agent._group == []
        assert agent._cur_obs == []

    def test_update_without_any_transition_returns_empty(self):
        agent = _make_agent()
        assert agent.update() == {}

    def test_surplus_episodes_carry_over_to_the_next_group(self):
        """A skipped update() must not merge 9 episodes into one group."""
        agent = _make_agent(group_size=4)
        rng = np.random.default_rng(1)
        for _ in range(5):
            _run_episode(agent, rng=rng)
        assert len(agent._group) == 5
        metrics = agent.update()
        assert metrics["group_size"] == 4.0
        assert len(agent._group) == 1

    def test_lr_schedule_steps_once_per_group(self):
        agent = _make_agent(group_size=4)
        rng = np.random.default_rng(2)
        for i in range(8):
            _run_episode(agent, reward=float(i), rng=rng)
            agent.update()
        assert agent.lr_scheduler.last_epoch == 2


# ---------------------------------------------------------------------------
# Advantages
# ---------------------------------------------------------------------------


class TestAdvantages:
    def test_group_advantage_is_the_outcome_zscore(self):
        agent = _make_agent()
        returns = np.array([1.0, 2.0, 3.0, 10.0])
        expected = (returns - returns.mean()) / returns.std()
        got = agent._group_advantages(returns)
        assert np.allclose(got, expected, atol=1e-5)
        assert abs(float(got.mean())) < 1e-5

    def test_zero_variance_group_gives_zero_advantage(self):
        agent = _make_agent()
        got = agent._group_advantages(np.full(6, 4.2))
        assert np.allclose(got, 0.0)

    def test_advantage_is_broadcast_over_every_step_of_its_episode(self):
        """Outcome supervision: one advantage per episode, not per step."""
        agent = _make_agent(group_size=4, n_epochs=1)
        rng = np.random.default_rng(3)
        lengths = [2, 3, 4, 5]
        rewards = [1.0, 2.0, 3.0, 4.0]
        for n_steps, r in zip(lengths, rewards, strict=True):
            _run_episode(agent, n_steps=n_steps, reward=r, rng=rng)
        agent.update()

        returns = np.array(
            [n * r for n, r in zip(lengths, rewards, strict=True)], dtype=np.float64
        )
        assert np.allclose(agent._last_group_returns, returns)
        expected = (returns - returns.mean()) / returns.std()
        adv = agent._last_advantages
        assert adv.shape == (sum(lengths),)
        start = 0
        for n_steps, exp in zip(lengths, expected, strict=True):
            seg = adv[start : start + n_steps]
            # Constant within the episode, equal to the group z-score.
            assert np.allclose(seg, seg[0])
            assert seg[0] == pytest.approx(exp, abs=1e-5)
            start += n_steps


# ---------------------------------------------------------------------------
# The clipped ratio (the legacy agent's headline defect)
# ---------------------------------------------------------------------------


class TestRatio:
    def test_ratio_moves_across_epochs(self):
        """Old log-probs are frozen BEFORE the epoch loop.

        Epoch 0 evaluates the collecting policy on the single minibatch,
        so its ratio is exactly 1; every later epoch sees weights that
        the previous epoch moved.  The legacy agent recomputed
        ``old_log_probs`` from the live network inside the same forward
        (grpo.py:310-321), which pins every epoch's deviation at 0.
        """
        agent = _make_agent(
            group_size=4, n_epochs=3, minibatch_size=4096, lr=0.05
        )
        rng = np.random.default_rng(4)
        for i in range(4):
            _run_episode(agent, n_steps=6, reward=float(i), rng=rng)
        metrics = agent.update()

        devs = agent._epoch_ratio_dev
        assert len(devs) == 3
        assert devs[0] == pytest.approx(0.0, abs=1e-6)
        assert devs[1] > 1e-4
        assert devs[2] > 1e-4
        assert metrics["ratio_dev"] > 0.0

    def test_target_kl_stops_the_epoch_loop_early(self):
        agent = _make_agent(
            group_size=4, n_epochs=3, minibatch_size=4096, lr=0.05,
            target_kl=1e-9,
        )
        rng = np.random.default_rng(12)
        for i in range(4):
            _run_episode(agent, n_steps=6, reward=float(i), rng=rng)
        metrics = agent.update()
        assert metrics["early_stop"] == 1.0
        assert len(agent._epoch_ratio_dev) < 3

    def test_zero_variance_group_leaves_the_surrogate_flat(self):
        """Equal outcomes → zero advantages → only the entropy bonus."""
        agent = _make_agent(group_size=4, n_epochs=1, minibatch_size=4096)
        rng = np.random.default_rng(5)
        for _ in range(4):
            _run_episode(agent, n_steps=3, reward=1.0, rng=rng)
        metrics = agent.update()
        assert metrics["group_return_std"] == pytest.approx(0.0)
        assert metrics["advantage_abs_mean"] == pytest.approx(0.0)
        assert metrics["pg_loss"] == pytest.approx(0.0, abs=1e-6)
        assert np.isfinite(metrics["loss"])


# ---------------------------------------------------------------------------
# Action masking
# ---------------------------------------------------------------------------


class TestMasking:
    def test_masked_slots_are_never_selected(self):
        agent = _make_agent()
        allowed = [1, 3]
        obs = _make_obs(n_techs=4, allowed=allowed)
        drawn = {agent.select_action(obs) for _ in range(64)}
        assert drawn <= set(allowed)
        assert agent.select_action(obs, deterministic=True) in allowed

    def test_mask_falls_back_to_the_valid_slot_bits(self):
        agent = _make_agent()
        obs = _make_obs(n_techs=2)
        obs.pop("action_mask")
        drawn = {agent.select_action(obs) for _ in range(32)}
        assert drawn <= {0, 1}

    def test_missing_mask_raises_instead_of_going_unmasked(self):
        agent = _make_agent()
        obs = _make_obs(n_techs=2)
        obs.pop("action_mask")
        obs.pop("tech_mask")
        with pytest.raises(KeyError):
            agent.select_action(obs)

    def test_update_respects_the_collected_masks(self):
        """A group collected under a 2-slot mask stays finite and masked."""
        agent = _make_agent(group_size=4, n_epochs=2)
        rng = np.random.default_rng(6)
        for i in range(4):
            _run_episode(
                agent, n_steps=3, reward=float(i), n_techs=4,
                allowed=[1, 3], rng=rng,
            )
        metrics = agent.update()
        assert np.isfinite(metrics["loss"])
        assert np.isfinite(metrics["entropy"])
        # Entropy of a 2-way masked distribution cannot exceed log 2.
        assert metrics["entropy"] <= np.log(2) + 1e-5
        obs = _make_obs(n_techs=4, allowed=[1, 3])
        assert agent.select_action(obs, deterministic=True) in (1, 3)


# ---------------------------------------------------------------------------
# Acting
# ---------------------------------------------------------------------------


class TestActing:
    def test_select_action_is_side_effect_free(self):
        agent = _make_agent()
        obs = _make_obs()
        for _ in range(5):
            agent.select_action(obs)
            agent.select_action(obs, deterministic=True)
        assert agent._cur_obs == []
        assert agent._group == []

    def test_deterministic_action_is_stable(self):
        agent = _make_agent()
        obs = _make_obs()
        a1 = agent.select_action(obs, deterministic=True)
        a2 = agent.select_action(obs, deterministic=True)
        assert a1 == a2


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------


class TestCheckpointing:
    def test_save_load_round_trip_preserves_the_policy(self):
        agent = _make_agent(group_size=4)
        rng = np.random.default_rng(7)
        for i in range(4):
            _run_episode(agent, reward=float(i), rng=rng)
        agent.update()  # move the weights away from init

        probe = [_make_obs(n_techs=k, rng=np.random.default_rng(8)) for k in (1, 3, 4)]
        before = [agent.select_action(o, deterministic=True) for o in probe]

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "grpo_mlp_best.pt"
            agent.save(path)
            fresh = _make_agent(group_size=4, seed=123)
            head = "policy_head.weight"
            assert not torch.equal(
                fresh.net.state_dict()[head], agent.net.state_dict()[head]
            ), "the round trip must be proving something"
            fresh.load(path)
            after = [fresh.select_action(o, deterministic=True) for o in probe]

        assert after == before
        for p, q in zip(
            agent.net.state_dict().values(),
            fresh.net.state_dict().values(),
            strict=True,
        ):
            assert torch.equal(p, q)

    def test_partial_groups_are_dropped_by_save_and_load(self):
        """Documented choice: buffered episodes never survive a checkpoint.

        They were collected under the pre-checkpoint policy; letting them
        into a post-restore group would z-score outcomes from two
        different behaviour policies against each other.
        """
        agent = _make_agent(group_size=8)
        rng = np.random.default_rng(9)
        for _ in range(3):
            _run_episode(agent, rng=rng)
        assert len(agent._group) == 3

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "grpo_mlp_ep00003.pt"
            agent.save(path)
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            assert "group" not in ckpt and "episodes" not in ckpt

            fresh = _make_agent(group_size=8, seed=1)
            for _ in range(2):  # a partial group on the *loading* side too
                _run_episode(fresh, rng=rng)
            fresh.load(path)

        assert fresh._group == []
        assert fresh._cur_obs == []

    def test_checkpoint_carries_the_vocab(self):
        agent = _make_agent()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "grpo_mlp_final.pt"
            agent.save(path)
            assert GRPOMLPAgent.peek_vocab(path) == VOCAB

    def test_incompatible_checkpoint_fails_loudly(self):
        agent = _make_agent(hidden_sizes=(16, 16))
        other = _make_agent(hidden_sizes=(24, 24))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "grpo_mlp_final.pt"
            other.save(path)
            with pytest.raises(RuntimeError, match="does not match"):
                agent.load(path)


# ---------------------------------------------------------------------------
# Vocabulary handover (the runner calls attach_vocab right after build)
# ---------------------------------------------------------------------------


class TestVocab:
    def test_attaching_the_same_vocab_is_a_no_op(self):
        agent = _make_agent()
        obs = _make_obs()
        before = agent.flattener(obs).clone()
        agent.attach_vocab(dict(VOCAB))
        assert torch.equal(agent.flattener(obs), before)

    def test_attaching_a_reindexed_vocab_rebuilds_the_encoding(self):
        """Vocab drift re-indexes keys — the one-hot tables must follow.

        Moving ``TEMPLATE=junior`` to a fresh id leaves the id the fake
        observation carries pointing at no TEMPLATE token at all, so the
        rebuilt flattener must route it to the OTHER bin instead of
        keeping the stale one-hot.
        """
        agent = _make_agent()
        obs = _make_obs()
        before = agent.flattener(obs).clone()
        reindexed = dict(VOCAB)
        reindexed["TEMPLATE=junior"] = max(VOCAB.values()) + 1
        agent.attach_vocab(reindexed)
        after = agent.flattener(obs)
        assert not torch.equal(after, before)
        sl = agent.flattener.position_slice("tech", "TEMPLATE", 0)
        assert after[sl][-1] == 1.0  # the trailing OTHER bin
        assert agent._vocab["TEMPLATE=junior"] == max(VOCAB.values()) + 1

    def test_width_changing_vocab_is_refused(self):
        agent = _make_agent()
        shrunk = {k: v for k, v in VOCAB.items() if k != "TEMPLATE=trainee"}
        with pytest.raises(RuntimeError, match="flat observation width"):
            agent.attach_vocab(shrunk)


# ---------------------------------------------------------------------------
# The shipped JSON config
# ---------------------------------------------------------------------------


class TestShippedConfig:
    CONFIG = (
        Path(__file__).resolve().parents[1]
        / "run_configs" / "agents" / "grpo_mlp.json"
    )

    def _params(self) -> dict:
        import json

        data = json.loads(self.CONFIG.read_text())
        assert data["agent_type"] == "grpo_mlp"
        return data["params"]

    def test_config_pins_no_env_derived_sizes(self):
        """No ``vocab_size`` pin (the rainbow_dqn.json:2 lesson) and no
        slot caps: the runner injects those from the env config."""
        params = self._params()
        for key in (
            "vocab_size", "n_actions", "max_techs", "max_machines",
            "env_length", "tech_slot_length",
        ):
            assert key not in params, f"{key} must come from the env config"

    def test_config_params_construct_the_agent(self):
        params = dict(self._params())
        params["device"] = "cpu"
        # Exactly what the runner injects for a set-obs agent.
        agent = GRPOMLPAgent(
            n_actions=30, max_techs=30, max_machines=100, env_length=16,
            tech_slot_length=16, sim_time_scale=200_000.0, **params,
        )
        assert agent.obs_dim == 5204
        assert agent.group_size == 8
        assert agent.gamma == 1.0  # raw undiscounted outcome supervision

    def test_default_vocab_resolves_from_any_working_directory(self, tmp_path,
                                                               monkeypatch):
        monkeypatch.chdir(tmp_path)
        agent = _make_agent(vocab=None)
        assert agent._vocab == VOCAB


# ---------------------------------------------------------------------------
# Scale transfer (within the slot caps)
# ---------------------------------------------------------------------------


class TestScaleTransfer:
    def test_one_policy_serves_every_fleet_size_up_to_the_cap(self):
        agent = _make_agent()
        widths = set()
        for n_techs in (1, 2, 3, 4):
            obs = _make_obs(n_techs=n_techs, n_machines=min(n_techs, MAX_M))
            widths.add(int(agent.flattener(obs).shape[0]))
            action = agent.select_action(obs)
            assert 0 <= action < n_techs
        assert widths == {agent.obs_dim}, "flat width must not vary with scale"

    def test_group_trained_small_transfers_to_a_larger_fleet(self):
        agent = _make_agent(group_size=4)
        rng = np.random.default_rng(10)
        for i in range(4):
            _run_episode(agent, n_techs=1, reward=float(i), rng=rng)
        agent.update()

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "grpo_mlp_best.pt"
            agent.save(path)
            fresh = _make_agent(group_size=4, seed=99)
            fresh.load(path)

        big = _make_obs(n_techs=4, n_machines=3, rng=np.random.default_rng(11))
        assert fresh.select_action(big, deterministic=True) == agent.select_action(
            big, deterministic=True
        )
        assert 0 <= fresh.select_action(big) < MAX_T
