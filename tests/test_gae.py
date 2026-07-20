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

GAMMA = 0.99
LAM = 0.95


def _gae(rewards, values, dones, last_value, gamma=GAMMA, lam=LAM):
    """Call the (unbound) production implementation on a stub self."""
    stub = SimpleNamespace(gamma=gamma, gae_lambda=lam)
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
