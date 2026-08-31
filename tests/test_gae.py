"""Regression tests for ``SetTransformerAgent._compute_gae``.

Pins the episode-boundary semantics of GAE: ``dones[t]`` marks
transition t as episode-ending, so it must mask transition t's own
bootstrap and lambda-chain.  A historical off-by-one consumed
``dones[t+1]`` instead, which (a) bootstrapped terminal transitions
with the value of the NEXT episode's first observation, (b) chained
advantages across episode boundaries, and (c) severed the
terminal-reward credit at the second-to-last transition of every
episode.  These tests fail on that variant.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from agents.ppo.ppo_set_transformer import SetTransformerAgent
from agents.ppo.ppo_transformer import PPOTransformerAgent

GAMMA = 0.99
LAM = 0.95


def _gae(rewards, values, dones, last_value, gamma=GAMMA, lam=LAM):
    """Call the (unbound) production implementation on a stub self."""
    stub = SimpleNamespace(
        gamma=gamma, gae_lambda=lam, time_based_discount=False
    )
    return SetTransformerAgent._compute_gae(
        stub,
        np.asarray(rewards, dtype=np.float32),
        np.asarray(values, dtype=np.float32),
        np.asarray(dones, dtype=bool),
        float(last_value),
    )


def _gae_reference(rewards, values, dones, last_value, gamma=GAMMA, lam=LAM):
    """Textbook GAE (Schulman et al. 2016), written independently."""
    n = len(rewards)
    adv = np.zeros(n, dtype=np.float64)
    gae = 0.0
    for t in reversed(range(n)):
        nnt = 0.0 if dones[t] else 1.0
        nv = last_value if t == n - 1 else values[t + 1]
        delta = rewards[t] + gamma * nv * nnt - values[t]
        gae = delta + gamma * lam * nnt * gae
        adv[t] = gae
    return adv, adv + np.asarray(values, dtype=np.float64)


def test_matches_reference_with_interior_dones():
    rng = np.random.default_rng(7)
    n = 64
    rewards = rng.normal(size=n)
    values = rng.normal(size=n)
    dones = np.zeros(n, dtype=bool)
    dones[[9, 30, 47]] = True  # three interior episode boundaries
    adv, ret = _gae(rewards, values, dones, last_value=1.7)
    ref_adv, ref_ret = _gae_reference(rewards, values, dones, last_value=1.7)
    np.testing.assert_allclose(adv, ref_adv, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(ret, ref_ret, rtol=1e-5, atol=1e-5)


def test_terminal_transition_has_no_bootstrap():
    # Single transition ending the episode: advantage must be r - V,
    # regardless of last_value (there is no s_{t+1} to bootstrap from).
    adv, ret = _gae([2.0], [0.5], [True], last_value=100.0)
    assert adv[0] == pytest.approx(2.0 - 0.5)
    assert ret[0] == pytest.approx(2.0)


def test_episode_boundary_isolates_episodes():
    # A buffer holding two complete episodes must yield, for the first
    # episode, exactly the advantages of that episode computed alone.
    # This is the property the off-by-one violated (cross-episode leak).
    rng = np.random.default_rng(3)
    r1, v1 = rng.normal(size=5), rng.normal(size=5)
    r2, v2 = rng.normal(size=4), rng.normal(size=4)
    d1 = [False] * 4 + [True]
    d2 = [False] * 3 + [True]

    adv_joint, _ = _gae(
        np.concatenate([r1, r2]),
        np.concatenate([v1, v2]),
        np.asarray(d1 + d2),
        last_value=0.0,
    )
    adv_solo, _ = _gae(r1, v1, d1, last_value=123.0)  # last_value must be moot
    np.testing.assert_allclose(adv_joint[:5], adv_solo, rtol=1e-5, atol=1e-5)


def test_second_to_last_transition_keeps_bootstrap():
    # Within one episode ending at the buffer tail, transition n-2 must
    # bootstrap from V(s_{n-1}) and chain to the terminal advantage.
    # (The off-by-one zeroed both, making terminal rewards invisible to
    # every step but the last.)
    rewards = [0.0, 0.0, 10.0]  # terminal reward only
    values = [0.0, 0.0, 0.0]
    dones = [False, False, True]
    adv, _ = _gae(rewards, values, dones, last_value=0.0)
    # delta_2 = 10; adv_1 = gamma*lam*10; adv_0 = (gamma*lam)^2 * 10
    assert adv[2] == pytest.approx(10.0)
    assert adv[1] == pytest.approx(GAMMA * LAM * 10.0, rel=1e-5)
    assert adv[0] == pytest.approx((GAMMA * LAM) ** 2 * 10.0, rel=1e-5)


def test_truncation_bootstraps_only_the_tail():
    # Rollout cut mid-episode (no done at the tail): the last transition
    # bootstraps from last_value; nothing is masked.
    rewards = [1.0, 1.0]
    values = [0.0, 0.0]
    dones = [False, False]
    adv, _ = _gae(rewards, values, dones, last_value=5.0)
    delta1 = 1.0 + GAMMA * 5.0
    delta0 = 1.0 + GAMMA * 0.0
    assert adv[1] == pytest.approx(delta1, rel=1e-5)
    assert adv[0] == pytest.approx(delta0 + GAMMA * LAM * delta1, rel=1e-5)


# ---------------------------------------------------------------------------
# Semi-MDP time-based discounting: gamma is per sim-time-unit and each
# transition is discounted by gamma**dt.
# ---------------------------------------------------------------------------


def _gae_smdp(rewards, values, dones, last_value, dts, gamma=0.9999, lam=LAM):
    stub = SimpleNamespace(
        gamma=gamma, gae_lambda=lam, time_based_discount=True
    )
    return SetTransformerAgent._compute_gae(
        stub,
        np.asarray(rewards, dtype=np.float32),
        np.asarray(values, dtype=np.float32),
        np.asarray(dones, dtype=bool),
        float(last_value),
        np.asarray(dts, dtype=np.float64),
    )


def test_smdp_with_unit_dts_equals_per_decision_gae():
    rng = np.random.default_rng(3)
    n = 48
    rewards, values = rng.normal(size=n), rng.normal(size=n)
    dones = np.zeros(n, dtype=bool)
    dones[20] = True
    adv_t, ret_t = _gae_smdp(rewards, values, dones, 0.4, np.ones(n),
                             gamma=GAMMA)
    adv_d, ret_d = _gae(rewards, values, dones, 0.4, gamma=GAMMA)
    np.testing.assert_allclose(adv_t, adv_d, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(ret_t, ret_d, rtol=1e-5, atol=1e-5)


def test_smdp_discount_depends_on_elapsed_time_not_step_count():
    # One transition, reward only in the bootstrap: A_0 = g*V' - V.
    # The discount must be gamma**dt.
    for dt in (1.0, 22.0, 200.0):
        adv, _ = _gae_smdp([0.0], [0.0], [False], 1.0, [dt], lam=0.0)
        assert adv[0] == pytest.approx(0.9999 ** dt, rel=1e-6)


def test_smdp_respects_episode_boundaries():
    # Terminal transition must not bootstrap regardless of its dt.
    adv, _ = _gae_smdp([1.0], [0.0], [True], 55.0, [1e6], lam=0.0)
    assert adv[0] == pytest.approx(1.0)


def test_disabled_flag_ignores_dts():
    stub = SimpleNamespace(
        gamma=GAMMA, gae_lambda=LAM, time_based_discount=False
    )
    rng = np.random.default_rng(11)
    n = 16
    rewards, values = rng.normal(size=n), rng.normal(size=n)
    dones = np.zeros(n, dtype=bool)
    adv, _ = SetTransformerAgent._compute_gae(
        stub,
        rewards.astype(np.float32), values.astype(np.float32), dones, 0.0,
        np.full(n, 500.0),
    )
    ref, _ = _gae(rewards, values, dones, 0.0)
    np.testing.assert_allclose(adv, ref, rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------------------
# Parent class (PPOTransformerAgent — the §7.4 anchor agent) must share the
# same episode-boundary semantics.  Its GAE is a separate implementation;
# the 2026-07-20 fix originally landed only in the subclass (defect D6).
# ---------------------------------------------------------------------------


def _gae_parent(rewards, values, dones, last_value, gamma=GAMMA, lam=LAM):
    stub = SimpleNamespace(gamma=gamma, gae_lambda=lam)
    return PPOTransformerAgent._compute_gae(
        stub,
        np.asarray(rewards, dtype=np.float32),
        np.asarray(values, dtype=np.float32),
        np.asarray(dones, dtype=bool),
        float(last_value),
    )


def test_parent_matches_reference_with_interior_dones():
    rng = np.random.default_rng(17)
    n = 64
    rewards = rng.normal(size=n)
    values = rng.normal(size=n)
    dones = np.zeros(n, dtype=bool)
    dones[[9, 30, 47]] = True
    adv, ret = _gae_parent(rewards, values, dones, last_value=1.7)
    ref_adv, ref_ret = _gae_reference(rewards, values, dones, last_value=1.7)
    np.testing.assert_allclose(adv, ref_adv, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(ret, ref_ret, rtol=1e-5, atol=1e-5)


def test_parent_second_to_last_transition_keeps_bootstrap():
    # The exact symptom of the off-by-one: with a terminal-only reward,
    # transitions before the last must chain to the terminal advantage.
    rewards = [0.0, 0.0, 10.0]
    values = [0.0, 0.0, 0.0]
    dones = [False, False, True]
    adv, _ = _gae_parent(rewards, values, dones, last_value=0.0)
    assert adv[2] == pytest.approx(10.0)
    assert adv[1] == pytest.approx(GAMMA * LAM * 10.0, rel=1e-5)
    assert adv[0] == pytest.approx((GAMMA * LAM) ** 2 * 10.0, rel=1e-5)


def test_parent_episode_boundary_isolates_episodes():
    rng = np.random.default_rng(5)
    r1, v1 = rng.normal(size=5), rng.normal(size=5)
    r2, v2 = rng.normal(size=4), rng.normal(size=4)
    d1 = [False] * 4 + [True]
    d2 = [False] * 3 + [True]
    adv_joint, _ = _gae_parent(
        np.concatenate([r1, r2]),
        np.concatenate([v1, v2]),
        np.asarray(d1 + d2),
        last_value=0.0,
    )
    adv_solo, _ = _gae_parent(r1, v1, d1, last_value=123.0)
    np.testing.assert_allclose(adv_joint[:5], adv_solo, rtol=1e-5, atol=1e-5)


def test_observe_transition_records_sim_time_deltas():
    """dt bookkeeping: first transition of an episode falls back to 0,
    later ones record the sim-time delta; done resets the stamp."""
    agent = SetTransformerAgent.__new__(SetTransformerAgent)
    agent.n_actions = 4
    agent.normalize_rewards = False
    agent.rnn_type = "none"
    agent.time_based_discount = True
    from collections import defaultdict
    agent._streams = defaultdict(
        lambda: {
            "obs": [], "action": [], "reward": [], "done": [],
            "logprob": [], "value": [], "mask": [], "hidden": [], "dt": [],
        }
    )
    agent._pending, agent._last, agent._rnn_state = {}, {}, {}
    agent._return_running = defaultdict(float)
    agent._last_sim_time = {}
    agent._extract_obs = lambda o: {}
    agent._extract_action_mask = lambda o: np.ones(4, dtype=bool)
    agent._hidden_to_numpy = lambda h: None

    def step(sim_time, done=False):
        agent.observe_transition(
            {}, 0, 0.0, {}, done, False, {"sim_time": sim_time}, env_id=0
        )

    step(107.0)          # first of episode: no previous stamp
    step(129.5)          # dt = 22.5
    step(329.5, done=True)  # dt = 200, then stamp cleared
    step(13.0)           # new episode: fallback again
    assert agent._streams[0]["dt"] == [0.0, 22.5, 200.0, 0.0]


def test_rearm_uses_constructor_lr_not_checkpoint_base(tmp_path):
    """A resumed run with a deliberately changed LR must keep it: the
    re-arm must not resurrect the checkpoint's base_lrs (which
    lr_scheduler.load_state_dict restores)."""
    import torch

    def make(lr, total):
        return PPOTransformerAgent(
            n_actions=3, vocab_size=8, d_model=16, n_heads=2, n_layers=1,
            max_seq_len=8, lr=lr, total_updates=total, warmup_updates=2,
            device="cpu",
        )

    a = make(3e-4, 10)
    for _ in range(15):
        a.lr_scheduler.step()  # exhaust the schedule
    p = tmp_path / "ck.pt"
    a.save(p)

    b = make(1.5e-4, 10)  # fine-tune at half LR
    b.load(p)
    lrs = [g["lr"] for g in b.optimizer.param_groups]
    expected = 1.5e-4 * b._lr_lambda(b.warmup_updates)
    assert all(abs(x - expected) < 1e-9 for x in lrs), lrs
    assert all(abs(x - 1.5e-4) < 1e-9 for x in b.lr_scheduler.base_lrs)


def test_rearm_fires_on_same_size_extension(tmp_path):
    """The hc_v6_ext defect (2026-08-31): a checkpoint whose schedule
    position sits just BELOW the new run's total_updates (same-size
    extension) must still get a fresh schedule — the old exhausted-only
    gate resumed the cosine at its tail and the whole extension trained
    at floor LR."""
    import torch

    def make(lr, total):
        return PPOTransformerAgent(
            n_actions=3, vocab_size=8, d_model=16, n_heads=2, n_layers=1,
            max_seq_len=8, lr=lr, total_updates=total, warmup_updates=2,
            device="cpu",
        )

    a = make(3e-4, 10)
    for _ in range(9):
        a.lr_scheduler.step()  # just below the new run's budget
    p = tmp_path / "ck.pt"
    a.save(p)

    b = make(1e-4, 10)
    b.load(p)
    assert b.lr_scheduler.last_epoch == b.warmup_updates
    expected = 1e-4 * b._lr_lambda(b.warmup_updates)
    lrs = [g["lr"] for g in b.optimizer.param_groups]
    assert all(abs(x - expected) < 1e-9 for x in lrs), lrs
