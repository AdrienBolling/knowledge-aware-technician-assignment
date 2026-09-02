"""Per-type disruption counters and cumulative held-time metrics.

The benchmark's ``ill_technician_count`` sums events across all types;
these tests pin the per-type breakdown added 2026-09-02: technician-level
accumulation in ``GymTechnician._take_disruption`` and the fleet-level
``disruptions_<type>`` / ``disruption_time_<type>`` episode metrics.
"""

from __future__ import annotations

import types

import simpy

from kata.entities.technicians.GymTechnician import GymTechnician
from kata.metrics import DisruptionCountByType, DisruptionTimeByType


def _mk_tech(env):
    tech = GymTechnician.__new__(GymTechnician)
    tech.retired = False
    tech._in_disruption = False
    tech.disruption_count = 0
    tech.disruption_counts_by_type = {}
    tech.disruption_time_by_type = {}

    class _Rng:
        def normal(self, mu, sig):
            return mu

        def exponential(self, scale):
            return scale

    tech._rng = _Rng()
    return tech


def test_take_disruption_accumulates_count_and_elapsed_time():
    env = simpy.Environment()
    tech = _mk_tech(env)
    res = simpy.PreemptiveResource(env, capacity=1)
    cfg = types.SimpleNamespace(preemptive=False, duration_mu=120.0, duration_sig=0.0)

    def proc():
        yield from tech._take_disruption(env, res, "exhaustion", cfg)
        yield from tech._take_disruption(env, res, "exhaustion", cfg)
        yield from tech._take_disruption(env, res, "vacation", cfg)

    env.process(proc())
    env.run()

    assert tech.disruption_count == 3
    assert tech.disruption_counts_by_type == {"exhaustion": 2, "vacation": 1}
    assert tech.disruption_time_by_type["exhaustion"] == 240.0
    assert tech.disruption_time_by_type["vacation"] == 120.0


def test_fleet_metrics_sum_over_techs_including_retired():
    env = simpy.Environment()
    a, b = _mk_tech(env), _mk_tech(env)
    a.disruption_counts_by_type = {"injury": 2}
    a.disruption_time_by_type = {"injury": 480.0}
    b.disruption_counts_by_type = {"injury": 1, "vacation": 4}
    b.disruption_time_by_type = {"injury": 240.0, "vacation": 1920.0}
    b.retired = True  # history must still count

    fake_env = types.SimpleNamespace(
        dispatcher=types.SimpleNamespace(techs=[a, b])
    )
    assert DisruptionCountByType("injury").compute(fake_env) == 3.0
    assert DisruptionTimeByType("injury").compute(fake_env) == 720.0
    assert DisruptionCountByType("vacation").compute(fake_env) == 4.0
    assert DisruptionTimeByType("vacation").compute(fake_env) == 1920.0
    # unconfigured type reads as zero, not KeyError
    assert DisruptionCountByType("solar_flare").compute(fake_env) == 0.0


def test_metric_names_registered():
    from kata.metrics import EPISODE_METRICS

    names = {m.name for m in EPISODE_METRICS}
    for d in ("injury", "exhaustion", "vacation"):
        assert f"disruptions_{d}" in names
        assert f"disruption_time_{d}" in names
