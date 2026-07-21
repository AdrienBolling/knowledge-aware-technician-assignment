"""Equivalence tests: event-driven failure sampling vs per-dt polling.

The event driver samples time-to-failure by inverse-CDF at the envelope
hazard and thins by machine state (exact for state-modulated hazards),
so the two drivers must agree in distribution (up to the dt
discretisation of the polling variant).
"""

from __future__ import annotations

import random

import numpy as np

from kata.features.breakdown.simple_breakdown import (
    SimpleBreakdownProcess,
    WeibullBreakdownProcess,
)


def _polled_failure_age(bp, max_steps=100_000):
    for _ in range(max_steps):
        p = bp.step_and_get_proba()
        if p > 0 and random.uniform(0, 1) <= p:
            return bp.age if hasattr(bp, "age") else bp.time_since_repair
    raise AssertionError("no failure within cap")


def test_weibull_event_sampling_matches_polling_distribution():
    random.seed(123)
    n = 3000
    polled = [
        _polled_failure_age(WeibullBreakdownProcess(shape=2.0, scale=300.0))
        for _ in range(n)
    ]
    sampled = [
        WeibullBreakdownProcess(shape=2.0, scale=300.0).sample_envelope_wait(1.0)
        for _ in range(n)
    ]
    mp, ms = float(np.mean(polled)), float(np.mean(sampled))
    # Weibull mean = scale * Gamma(1 + 1/shape) = 300 * 0.8862 = 265.9
    assert abs(mp - ms) / mp < 0.05, (mp, ms)
    q_p, q_s = np.percentile(polled, [25, 50, 75]), np.percentile(sampled, [25, 50, 75])
    assert np.all(np.abs(q_p - q_s) / q_p < 0.08), (q_p, q_s)


def test_simple_event_sampling_matches_geometric_mean():
    random.seed(7)
    n = 4000
    p_fail = 0.01
    polled = [
        _polled_failure_age(SimpleBreakdownProcess(failure_prob_working=p_fail))
        for _ in range(n)
    ]
    sampled = [
        SimpleBreakdownProcess(failure_prob_working=p_fail).sample_envelope_wait(1.0)
        for _ in range(n)
    ]
    mp, ms = float(np.mean(polled)), float(np.mean(sampled))
    assert abs(mp - ms) / mp < 0.06, (mp, ms)


def test_idle_thinning_fraction_matches_hazard_ratio():
    bp = WeibullBreakdownProcess(shape=2.0, scale=300.0)
    assert bp.accept_fraction(True, 1.0) == 1.0
    assert bp.accept_fraction(False, 1.0) == 0.1
    sp = SimpleBreakdownProcess(failure_prob_working=0.01, failure_prob_idle=0.001)
    assert sp.accept_fraction(True, 1.0) == 1.0
    assert 0.09 < sp.accept_fraction(False, 1.0) < 0.11
    # hazard-free process yields no candidates
    assert SimpleBreakdownProcess(0.0, 0.0).sample_envelope_wait(1.0) is None


def test_event_sampling_respects_kijima_residual_age():
    random.seed(11)
    n = 2500
    fresh = np.mean(
        [WeibullBreakdownProcess(shape=2.0, scale=300.0).sample_envelope_wait(1.0)
         for _ in range(n)]
    )
    aged_waits = []
    for _ in range(n):
        bp = WeibullBreakdownProcess(shape=2.0, scale=300.0, restoration_alpha=0.5)
        bp.advance_age(200.0)
        bp.repair()          # Kijima: residual age 100 survives
        assert bp.age == 100.0
        aged_waits.append(bp.sample_envelope_wait(1.0))
    # Increasing hazard (shape > 1): residual age shortens expected wait.
    assert np.mean(aged_waits) < 0.9 * fresh


def test_aged_high_shape_component_cannot_storm():
    """An aged, high-shape (exploding-hazard) component must never sample
    sub-dt candidate waits: the polling driver could fire at most one
    event per dt, and without this cap the event calendar degenerates
    into millions of micro-events late in long episodes."""
    random.seed(5)
    bp = WeibullBreakdownProcess(shape=3.0, scale=100.0, dt=1)
    bp.advance_age(300_000.0)
    waits = [bp.sample_envelope_wait(1.0) for _ in range(200)]
    assert min(waits) >= 1.0
