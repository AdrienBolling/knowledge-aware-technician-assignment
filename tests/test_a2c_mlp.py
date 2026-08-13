"""Tests for the A2C-MLP traditional baseline.

The agent is constructed DIRECTLY (tiny dims, cpu, canonical frozen
vocab) — no runner registry, no hydra.  Fake ``set`` observations are
assembled from the frozen slot layouts, mirroring
``tests/test_set_flattener.py``.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from agents.a2c.a2c_mlp import A2CMLPAgent
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

_CONT_FMT = {
    ContKind.RATIO_PLE: "<RATIO:{}>",
    ContKind.COUNT_PLE: "<COUNT:{}>",
    ContKind.TIME2VEC: "<TIME:{}>",
    ContKind.FOURIER: "<FOUR:{}>",
}
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

MAX_T = 6
MAX_M = 3


def _fill_slot(ids_row, vals_row, kinds_row, layout, defaults, overrides=None, *,
               vocab=None):
    """Write one slot's (token id, cont value, kind) triples in layout order.

    Mirrors ``KataEnv._set_obs``: positions past the layout keep the PAD
    id / 0.0 value / CATEGORICAL kind they were zero-initialised with.
    """
    vocab = VOCAB if vocab is None else vocab
    unk = vocab.get("<UNK>", UNK_ID)
    values = dict(defaults)
    values.update(overrides or {})
    for pos, sp in enumerate(layout):
        if sp.kind == ContKind.CATEGORICAL:
            ids_row[pos] = vocab.get(f"{sp.key}={values.get(sp.key)}", unk)
        else:
            ids_row[pos] = vocab[_CONT_FMT[sp.kind].format(sp.key)]
            vals_row[pos] = float(values.get(sp.key, 0.0))
        kinds_row[pos] = int(sp.kind)


def _make_obs(n_techs=3, n_machines=2, *, max_t=MAX_T, max_m=MAX_M, action_mask=None,
              rng=None, vocab=None, tech_overrides=None):
    """Build a fake ``set`` observation with the env's shapes and dtypes.

    ``vocab`` selects the token ids the observation is written with — the
    env's tokenizer decides them, so a re-indexed vocabulary produces a
    re-indexed observation (see the attach_vocab tests).
    """
    lt, lm, le = 16, 12, 16
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
        over = dict(tech_overrides or {})
        if rng is not None:
            over["FATIGUE"] = float(rng.random())
        _fill_slot(
            obs["tech_token_ids"][i], obs["tech_cont_values"][i],
            obs["tech_cont_kinds"][i],
            TECH_SLOT_LAYOUT, _TECH_DEFAULTS, over or None, vocab=vocab,
        )
        obs["tech_mask"][i] = 1
    for i in range(n_machines):
        _fill_slot(
            obs["machine_token_ids"][i], obs["machine_cont_values"][i],
            obs["machine_cont_kinds"][i],
            MACHINE_SLOT_LAYOUT, _MACH_DEFAULTS, vocab=vocab,
        )
        obs["machine_mask"][i] = 1
    _fill_slot(
        obs["env_token_ids"], obs["env_cont_values"], obs["env_cont_kinds"],
        ENV_SLOT_LAYOUT, _ENV_DEFAULTS, vocab=vocab,
    )
    if action_mask is not None:
        obs["action_mask"] = np.asarray(action_mask, dtype=np.int8)
    return obs


def _make_agent(**overrides):
    defaults = dict(
        n_actions=MAX_T,
        vocab=VOCAB,
        max_techs=MAX_T,
        max_machines=MAX_M,
        hidden_sizes=(16, 16),
        rollout_steps=8,
        total_updates=6,
        warmup_updates=1,
        device="cpu",
        seed=0,
    )
    defaults.update(overrides)
    return A2CMLPAgent(**defaults)


def _run_rollout(agent, steps=8, *, env_id=0, seed=0, done_at=None, mask=None):
    """Feed the agent a short synthetic rollout through the public API."""
    rng = np.random.default_rng(seed)
    for t in range(steps):
        obs = _make_obs(action_mask=mask, rng=rng)
        next_obs = _make_obs(action_mask=mask, rng=rng)
        action = agent.select_action(obs, env_id=env_id)
        done = done_at is not None and t == done_at
        agent.observe_transition(
            obs, action, float(rng.normal()), next_obs, done, False,
            {"sim_time": 10.0 * (t + 1)}, env_id=env_id,
        )


# ---------------------------------------------------------------------------
# Acting
# ---------------------------------------------------------------------------


class TestActing:
    def test_select_action_returns_valid_index(self):
        agent = _make_agent()
        a = agent.select_action(_make_obs(n_techs=3))
        assert isinstance(a, int)
        assert 0 <= a < agent.n_actions

    def test_mask_respected_over_many_samples(self):
        """Stochastic sampling must never pick a padded / unavailable slot.

        With 3 real technicians out of 6 slots, the tech_mask fallback
        leaves actions 3-5 invalid.
        """
        agent = _make_agent()
        obs = _make_obs(n_techs=3)  # no action_mask -> tech_mask fallback
        actions = {agent.select_action(obs) for _ in range(200)}
        assert actions and max(actions) < 3

    def test_explicit_action_mask_beats_tech_mask(self):
        agent = _make_agent()
        obs = _make_obs(n_techs=5, action_mask=[0, 1, 0, 1, 0, 0])
        for _ in range(100):
            assert agent.select_action(obs) in (1, 3)
        for _ in range(5):
            assert agent.select_action(obs, deterministic=True) in (1, 3)

    def test_deterministic_is_stable_and_is_masked_argmax(self):
        agent = _make_agent()
        obs = _make_obs(n_techs=4, action_mask=[0, 1, 1, 1, 0, 0])
        a1 = agent.select_action(obs, deterministic=True)
        a2 = agent.select_action(obs, deterministic=True)
        assert a1 == a2
        # Same decision the raw network would make once masked.
        x = agent.flattener(obs).unsqueeze(0)
        with torch.no_grad():
            logits, _ = agent.net(x)
        mask = torch.from_numpy(agent._action_mask(obs)).unsqueeze(0)
        expected = int(
            logits.masked_fill(~mask, float("-inf")).argmax(dim=-1).item()
        )
        assert a1 == expected

    def test_select_action_is_side_effect_free(self):
        """Acting caches nothing: an inline eval cannot corrupt a stream."""
        agent = _make_agent()
        obs = _make_obs()
        before = (
            dict(agent._streams), dict(agent._last), dict(agent._last_sim_time),
        )
        for _ in range(10):
            agent.select_action(obs)
            agent.select_action(obs, deterministic=True)
        assert (dict(agent._streams), dict(agent._last),
                dict(agent._last_sim_time)) == before

    def test_never_falls_back_to_all_ones_mask(self):
        agent = _make_agent()
        with pytest.raises(ValueError):
            agent.select_action(_make_obs(n_techs=0))
        with pytest.raises(KeyError):
            agent.select_action({})


# ---------------------------------------------------------------------------
# Scale transfer
# ---------------------------------------------------------------------------


class TestScaleTransfer:
    def test_same_agent_acts_at_two_fleet_sizes(self):
        """One checkpoint, a 2-tech and a 30-tech factory.

        The flattener's width is set by the slot CAPS, not the real
        fleet, so no parameter depends on the scenario size.
        """
        agent = A2CMLPAgent(
            n_actions=30, vocab=VOCAB, max_techs=30, max_machines=10,
            hidden_sizes=(16, 16), device="cpu", seed=0,
        )
        small = _make_obs(n_techs=2, n_machines=2, max_t=30, max_m=10)
        big = _make_obs(n_techs=30, n_machines=8, max_t=30, max_m=10)
        n_params = agent.num_parameters()
        for _ in range(50):
            assert agent.select_action(small) < 2
        a_big = {agent.select_action(big) for _ in range(50)}
        assert max(a_big) < 30 and len(a_big) > 1
        # Acting at either scale changes nothing about the network.
        assert agent.num_parameters() == n_params

    def test_n_actions_must_match_max_techs(self):
        with pytest.raises(ValueError, match="n_actions == max_techs"):
            A2CMLPAgent(n_actions=4, vocab=VOCAB, max_techs=6, device="cpu")


# ---------------------------------------------------------------------------
# GAE
# ---------------------------------------------------------------------------


def _gae(agent, rewards, values, dones, last_value, dts=None):
    return agent._compute_gae(
        np.asarray(rewards, dtype=np.float32),
        np.asarray(values, dtype=np.float32),
        np.asarray(dones, dtype=bool),
        float(last_value),
        None if dts is None else np.asarray(dts, dtype=np.float64),
    )


class TestGAE:
    def test_hand_computed_three_step_example(self):
        """Three transitions, no episode end, hand-rolled by the book.

        gamma = 0.5, lambda = 0.5, V = [1, 2, 3], r = [1, 1, 1],
        bootstrap V(s_3) = 4.

            d2 = 1 + .5*4 - 3 = 0.0     A2 = 0.0
            d1 = 1 + .5*3 - 2 = 0.5     A1 = 0.5  + .25*0.0  = 0.5
            d0 = 1 + .5*2 - 1 = 1.0     A0 = 1.0  + .25*0.5  = 1.125
        """
        agent = _make_agent(gamma=0.5, gae_lambda=0.5)
        adv, ret = _gae(agent, [1.0, 1.0, 1.0], [1.0, 2.0, 3.0],
                        [False, False, False], last_value=4.0)
        assert adv == pytest.approx([1.125, 0.5, 0.0], abs=1e-6)
        assert ret == pytest.approx([2.125, 2.5, 3.0], abs=1e-6)

    def test_dones_t_masks_its_own_bootstrap(self):
        """The corrected ``dones[t]`` semantics (defect D6).

        Same numbers, but transition 1 ends its episode: its bootstrap
        AND the lambda-chain into transition 0 must be severed.

            d1 = 1 - 2 = -1.0           A1 = -1.0
            d0 = 1 + .5*2 - 1 = 1.0     A0 = 1.0 + .25*(-1.0) = 0.75
        """
        agent = _make_agent(gamma=0.5, gae_lambda=0.5)
        adv, _ = _gae(agent, [1.0, 1.0, 1.0], [1.0, 2.0, 3.0],
                      [False, True, False], last_value=4.0)
        assert adv[1] == pytest.approx(-1.0, abs=1e-6)
        assert adv[0] == pytest.approx(0.75, abs=1e-6)
        # Transition 2 opens a new episode: untouched by the boundary.
        assert adv[2] == pytest.approx(0.0, abs=1e-6)

    def test_terminal_transition_ignores_last_value(self):
        agent = _make_agent(gamma=0.99, gae_lambda=0.95)
        adv, ret = _gae(agent, [2.0], [0.5], [True], last_value=100.0)
        assert adv[0] == pytest.approx(1.5)
        assert ret[0] == pytest.approx(2.0)

    def test_semi_mdp_discounts_by_elapsed_sim_time(self):
        agent = _make_agent(gamma=0.9999, gae_lambda=0.0,
                            time_based_discount=True)
        for dt in (1.0, 22.0, 200.0):
            adv, _ = _gae(agent, [0.0], [0.0], [False], 1.0, dts=[dt])
            assert adv[0] == pytest.approx(0.9999**dt, rel=1e-6)

    def test_dts_ignored_when_flag_off(self):
        agent = _make_agent(gamma=0.5, gae_lambda=0.5)
        a1, _ = _gae(agent, [1.0, 1.0], [0.0, 0.0], [False, False], 0.0,
                     dts=[500.0, 500.0])
        a2, _ = _gae(agent, [1.0, 1.0], [0.0, 0.0], [False, False], 0.0)
        assert a1 == pytest.approx(a2)


# ---------------------------------------------------------------------------
# Update
# ---------------------------------------------------------------------------


class TestUpdate:
    def test_update_runs_and_emits_metrics(self):
        agent = _make_agent()
        _run_rollout(agent, steps=agent.rollout_steps)
        metrics = agent.update()
        for key in ("loss", "pg_loss", "vf_loss", "entropy", "approx_kl",
                    "grad_norm", "lr", "rollout_size"):
            assert key in metrics, key
            assert np.isfinite(metrics[key]), key
        assert metrics["rollout_size"] == float(agent.rollout_steps)
        # Strictly on-policy: the rollout is consumed exactly once.
        assert not agent._streams
        assert not agent._last
        assert agent.update() == {}

    def test_update_changes_parameters(self):
        agent = _make_agent()
        before = [p.detach().clone() for p in agent.net.parameters()]
        _run_rollout(agent, steps=agent.rollout_steps)
        agent.update()
        after = list(agent.net.parameters())
        assert any(not torch.equal(b, a) for b, a in zip(before, after))

    def test_lr_schedule_advances_once_per_update(self):
        agent = _make_agent(total_updates=10, warmup_updates=2, lr=1e-3)
        lr0 = agent.optimizer.param_groups[0]["lr"]
        _run_rollout(agent, steps=agent.rollout_steps)
        agent.update()
        lr1 = agent.optimizer.param_groups[0]["lr"]
        assert lr1 != lr0
        assert agent.lr_scheduler.last_epoch == 1

    def test_multi_stream_rollout_is_gae_isolated(self):
        """Per-stream GAE: two vec streams in one buffer, one of them
        ending on a terminal transition (no bootstrap)."""
        agent = _make_agent()
        _run_rollout(agent, steps=4, env_id=0, seed=1)
        _run_rollout(agent, steps=4, env_id=1, seed=2, done_at=3)
        metrics = agent.update()
        assert metrics["rollout_size"] == 8.0

    def test_normalize_rewards_rescales_buffered_rewards(self):
        agent = _make_agent(normalize_rewards=True)
        rng = np.random.default_rng(0)
        for t in range(20):
            obs, nxt = _make_obs(), _make_obs()
            a = agent.select_action(obs)
            agent.observe_transition(
                obs, a, 1.0, nxt, False, False, {"sim_time": 10.0 * (t + 1)},
            )
        buffered = agent._streams[0]["reward"]
        assert max(abs(r - 1.0) for r in buffered) > 1e-3
        assert agent._return_rms.count > 1

    def test_approx_kl_measures_the_gradient_step(self):
        """``approx_kl`` must report the pre -> post policy movement.

        Both log-prob forwards used to happen BEFORE ``optimizer.step()``
        (same weights, same masks), so the logged value was identically
        0.0 for every update ever — telemetry that could never fire.
        """
        agent = _make_agent(lr=1e-2)
        _run_rollout(agent, steps=agent.rollout_steps)
        metrics = agent.update()
        assert np.isfinite(metrics["approx_kl"])
        assert metrics["approx_kl"] > 0.0

    def test_approx_kl_is_zero_when_the_step_moves_nothing(self):
        """lr = 0: the weights do not move, so there is nothing to report.

        Pins that the metric tracks the STEP and not forward noise.
        """
        agent = _make_agent(lr=0.0)
        before = [p.detach().clone() for p in agent.net.parameters()]
        _run_rollout(agent, steps=agent.rollout_steps)
        metrics = agent.update()
        assert all(
            torch.equal(b, a) for b, a in zip(before, agent.net.parameters())
        )
        assert metrics["approx_kl"] == pytest.approx(0.0, abs=1e-12)

    def test_dt_bookkeeping_across_episode_boundary(self):
        agent = _make_agent(time_based_discount=True)
        obs = _make_obs()
        for t, (sim_t, done) in enumerate(
            [(107.0, False), (129.5, False), (329.5, True), (13.0, False)]
        ):
            agent.observe_transition(
                obs, 0, 0.0, obs, done, False, {"sim_time": sim_t}
            )
            _ = t
        assert agent._streams[0]["dt"] == [0.0, 22.5, 200.0, 0.0]


# ---------------------------------------------------------------------------
# Vectorised-collection contract (runner.py::_train_loop_vec)
# ---------------------------------------------------------------------------


class TestVecInterface:
    def test_required_hooks_exist(self):
        agent = _make_agent()
        assert callable(agent.select_actions)
        assert callable(agent.reset_stream)
        assert isinstance(agent.rollout_steps, int)
        agent.reset_stream(3)  # arbitrary stream id, must not raise

    def test_select_actions_batches_and_respects_per_env_masks(self):
        agent = _make_agent()
        obs_list = [
            _make_obs(n_techs=2, action_mask=[1, 0, 0, 0, 0, 0]),
            _make_obs(n_techs=6, action_mask=[0, 0, 0, 0, 0, 1]),
            _make_obs(n_techs=3),
        ]
        for _ in range(20):
            actions = agent.select_actions(obs_list, env_ids=[0, 1, 2])
            assert len(actions) == 3
            assert actions[0] == 0
            assert actions[1] == 5
            assert actions[2] < 3

    def test_select_actions_matches_single_env_path_deterministically(self):
        agent = _make_agent()
        obs_list = [_make_obs(n_techs=4), _make_obs(n_techs=2)]
        batched = agent.select_actions(obs_list, deterministic=True)
        single = [
            agent.select_action(o, deterministic=True, env_id=i)
            for i, o in enumerate(obs_list)
        ]
        assert batched == single

    def test_reset_stream_keeps_the_rollout_but_clears_episode_state(self):
        """Autoreset semantics: reset_stream fires mid-round and must not
        drop transitions the round-boundary update still needs."""
        agent = _make_agent()
        _run_rollout(agent, steps=3, env_id=1)
        assert len(agent._streams[1]["obs"]) == 3
        agent.reset_stream(1)
        assert len(agent._streams[1]["obs"]) == 3
        assert agent._last_sim_time[1] is None

    def test_snapshot_restore_protects_stream_zero(self):
        agent = _make_agent()
        _run_rollout(agent, steps=3, env_id=0)
        snap = agent.snapshot_stream_state()
        anchor = agent._last_sim_time[0]
        agent.on_episode_start()          # what an inline eval does
        assert agent._last_sim_time[0] is None
        agent.restore_stream_state(snap)
        assert agent._last_sim_time[0] == anchor


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------


class TestCheckpointing:
    def test_save_load_round_trip(self):
        agent = _make_agent()
        agent.attach_vocab(VOCAB)
        _run_rollout(agent, steps=agent.rollout_steps)
        agent.update()

        obs = _make_obs(n_techs=4)
        agent.net.eval()
        with torch.no_grad():
            ref_logits, ref_value = agent.net(agent.flattener(obs).unsqueeze(0))

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "a2c.pt"
            agent.save(path)
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            assert ckpt["vocab"] == VOCAB
            assert ckpt["improvements"]["hidden_sizes"] == [16, 16]
            assert ckpt["improvements"]["in_dim"] == agent.flattener.out_dim
            assert "optimizer" in ckpt and "lr_scheduler" in ckpt
            assert A2CMLPAgent.peek_vocab(path) == VOCAB

            fresh = _make_agent(seed=123)
            with torch.no_grad():
                pre_logits, _ = fresh.net(fresh.flattener(obs).unsqueeze(0))
            assert not torch.allclose(pre_logits, ref_logits)

            fresh.load(path)
            fresh.net.eval()
            with torch.no_grad():
                new_logits, new_value = fresh.net(
                    fresh.flattener(obs).unsqueeze(0)
                )
            assert torch.allclose(ref_logits, new_logits, atol=1e-6)
            assert torch.allclose(ref_value, new_value, atol=1e-6)
            # Optimiser + schedule state travel with the weights.
            assert fresh.lr_scheduler.last_epoch == agent.lr_scheduler.last_epoch
            assert fresh.optimizer.state_dict()["state"]
            assert fresh._vocab == VOCAB

    def test_load_rearms_an_exhausted_lr_schedule(self):
        """A resume past the schedule end must not train at the floor
        (defect D1) — and must keep the CALLER's lr, not the ckpt's."""
        agent = _make_agent(total_updates=4, warmup_updates=1, lr=3e-4)
        for _ in range(6):
            agent.lr_scheduler.step()
        assert agent.lr_scheduler.last_epoch >= agent.total_updates

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "a2c.pt"
            agent.save(path)
            resumed = _make_agent(total_updates=4, warmup_updates=1, lr=1.5e-4)
            resumed.load(path)

        assert resumed.lr_scheduler.last_epoch == resumed.warmup_updates
        expected = 1.5e-4 * resumed._lr_lambda(resumed.warmup_updates)
        assert all(
            abs(g["lr"] - expected) < 1e-12 for g in resumed.optimizer.param_groups
        )
        assert all(abs(b - 1.5e-4) < 1e-12 for b in resumed.lr_scheduler.base_lrs)

    def test_incompatible_vocab_is_rejected(self):
        """Vocab drift re-indexes keys — a leading-row remap would
        scramble the one-hot columns, so attach/load must refuse."""
        agent = _make_agent()
        drifted = dict(VOCAB)
        # Swap two BUSY ids: same tokens, different one-hot order.
        drifted["BUSY=T"], drifted["BUSY=F"] = VOCAB["BUSY=F"], VOCAB["BUSY=T"]
        with pytest.raises(RuntimeError, match="one-hot layout differs"):
            agent.attach_vocab(drifted)

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "a2c.pt"
            agent.save(path)
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            ckpt["vocab"] = drifted
            torch.save(ckpt, path)
            with pytest.raises(RuntimeError, match="one-hot layout differs"):
                _make_agent().load(path)

    def test_attach_vocab_rebuilds_the_flattener_when_ids_move(self):
        """Same one-hot layout, shifted token ids — the D11-class trap.

        The signature check passes (the bin ORDER is untouched), but the
        flattener's lookup tables are indexed by ABSOLUTE token id, so
        keeping the constructor's tables would route every categorical to
        the OTHER bin.  The tables have to be rebuilt.
        """
        shifted = {tok: tid + 7 for tok, tid in VOCAB.items()}
        agent = _make_agent()                       # tables built on VOCAB
        stale = agent.flattener
        obs = _make_obs(n_techs=3, vocab=shifted)   # ids the tokenizer emits

        agent.attach_vocab(shifted)
        assert agent.flattener is not stale
        assert agent._vocab == shifted
        assert agent.vocab_size == len(shifted)
        assert agent.flattener.out_dim == stale.out_dim  # network stays valid

        # Ground truth: an agent constructed directly with the shifted vocab.
        ref = _make_agent(vocab=shifted)
        assert torch.allclose(agent.flattener(obs), ref.flattener(obs))
        # ... and the defect itself: the un-rebuilt tables disagree.
        assert not torch.allclose(stale(obs), ref.flattener(obs))

        # The categorical still lands in its own bin, not the OTHER bin.
        busy = _make_obs(n_techs=3, vocab=shifted, tech_overrides={"BUSY": "T"})
        bins = agent.flattener.categories("tech", "BUSY")
        flat = agent.flattener(busy)[agent.flattener.position_slice("tech", "BUSY")]
        assert flat.tolist() == [
            1.0 if tok == "BUSY=T" else 0.0 for tok in bins
        ] + [0.0]           # trailing OTHER bin

    def test_checkpoint_with_moved_ids_rebuilds_too(self):
        """A checkpoint's ids are what its weights were trained under."""
        agent = _make_agent()
        shifted = {tok: tid + 7 for tok, tid in VOCAB.items()}
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "a2c.pt"
            agent.save(path)
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            ckpt["vocab"] = shifted
            torch.save(ckpt, path)

            fresh = _make_agent()
            fresh.load(path)
        obs = _make_obs(n_techs=3, vocab=shifted)
        ref = _make_agent(vocab=shifted)
        assert fresh._vocab == shifted
        assert torch.allclose(fresh.flattener(obs), ref.flattener(obs))

    def test_architecture_mismatch_raises_clearly(self):
        agent = _make_agent(hidden_sizes=(16, 16))
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "a2c.pt"
            agent.save(path)
            other = _make_agent(hidden_sizes=(8, 8))
            with pytest.raises(RuntimeError, match="does not match"):
                other.load(path)


# ---------------------------------------------------------------------------
# Construction guards
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_defaults_to_the_canonical_vocab(self):
        agent = A2CMLPAgent(
            n_actions=MAX_T, max_techs=MAX_T, max_machines=MAX_M,
            hidden_sizes=(8,), device="cpu",
        )
        assert agent._vocab == VOCAB
        assert agent.flattener.out_dim > 0

    def test_launcher_only_flags_raise_instead_of_being_ignored(self):
        """train_hydra writes use_popart / rnn_type into every agent's
        params — this baseline implements neither."""
        with pytest.raises(ValueError, match="PopArt"):
            _make_agent(use_popart=True)
        with pytest.raises(ValueError, match="feed-forward"):
            _make_agent(rnn_type="gru")

    def test_accepts_runner_injected_params(self):
        """The set-agent injection block passes these unconditionally."""
        agent = A2CMLPAgent(
            n_actions=MAX_T,
            vocab=VOCAB,
            max_techs=MAX_T,
            max_machines=MAX_M,
            env_length=16,
            tech_slot_length=16,
            sim_time_scale=5_000_000.0,
            vocab_size=len(VOCAB),
            hidden_sizes=(8,),
            device="cpu",
        )
        assert agent.select_action(_make_obs()) < MAX_T

    def test_mixin_contract_is_satisfied(self):
        agent = _make_agent()
        for attr in ("net", "optimizer", "lr_scheduler", "total_updates",
                     "warmup_updates", "_lr_lambda", "_ctor_lr"):
            assert hasattr(agent, attr), attr
        # _eval_mode_if restores the training flag it found.
        agent.net.train()
        with agent._eval_mode_if(True):
            assert not agent.net.training
        assert agent.net.training


def test_config_json_matches_the_constructor():
    """Every param in the shipped agent JSON is a real ctor keyword."""
    import inspect
    import json

    cfg = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "run_configs" / "agents" / "a2c_mlp.json"
        ).read_text()
    )
    assert cfg["agent_type"] == "a2c_mlp"
    sig = inspect.signature(A2CMLPAgent.__init__).parameters
    unknown = sorted(set(cfg["params"]) - set(sig))
    assert not unknown, unknown
    # The runner injects the caps/slot lengths from the env config
    # (setdefault → a JSON value would silently win over the env).
    for injected in ("max_techs", "max_machines", "env_length",
                     "tech_slot_length", "n_actions", "vocab_size"):
        assert injected not in cfg["params"]
    params = dict(cfg["params"], device="cpu")
    agent = A2CMLPAgent(
        n_actions=MAX_T, max_techs=MAX_T, max_machines=MAX_M,
        vocab=VOCAB, **params,
    )
    assert isinstance(agent, A2CMLPAgent)
    assert agent.hidden_sizes == (512, 512)


def test_stub_gae_matches_a_reference_implementation():
    """Randomised cross-check against textbook GAE, with interior dones."""
    rng = np.random.default_rng(7)
    n = 64
    rewards, values = rng.normal(size=n), rng.normal(size=n)
    dones = np.zeros(n, dtype=bool)
    dones[[9, 30, 47]] = True
    stub = SimpleNamespace(gamma=0.99, gae_lambda=0.95, time_based_discount=False)
    adv, ret = A2CMLPAgent._compute_gae(
        stub, rewards.astype(np.float32), values.astype(np.float32), dones, 1.7
    )
    ref = np.zeros(n)
    gae = 0.0
    for t in reversed(range(n)):
        nnt = 0.0 if dones[t] else 1.0
        nv = 1.7 if t == n - 1 else values[t + 1]
        delta = rewards[t] + 0.99 * nv * nnt - values[t]
        gae = delta + 0.99 * 0.95 * nnt * gae
        ref[t] = gae
    np.testing.assert_allclose(adv, ref, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(ret, ref + values, rtol=1e-5, atol=1e-5)
