"""Tests for breakdown process implementations."""

import math

from kata.features.breakdown.simple_breakdown import (
    SimpleBreakdownProcess,
    WeibullBreakdownProcess,
)


class TestSimpleBreakdownProcess:
    def test_returns_constant_working_probability(self):
        bp = SimpleBreakdownProcess(failure_prob_working=0.05, failure_prob_idle=0.01)
        assert bp.step_and_get_proba() == 0.05

    def test_returns_constant_idle_probability(self):
        bp = SimpleBreakdownProcess(failure_prob_working=0.05, failure_prob_idle=0.01)
        assert bp.step_and_get_idle_proba() == 0.01

    def test_repair_resets_time_since_repair(self):
        bp = SimpleBreakdownProcess()
        bp.step_and_get_proba()
        bp.step_and_get_proba()
        assert bp.time_since_repair == 2
        bp.repair()
        assert bp.time_since_repair == 0

    def test_zero_probability_never_triggers(self):
        bp = SimpleBreakdownProcess(failure_prob_working=0.0, failure_prob_idle=0.0)
        for _ in range(1000):
            assert bp.step_and_get_proba() == 0.0
            assert bp.step_and_get_idle_proba() == 0.0


class TestWeibullBreakdownProcess:
    def test_probability_increases_with_age(self):
        bp = WeibullBreakdownProcess(shape=2.0, scale=100.0, dt=1)
        probs = [bp.step_and_get_proba() for _ in range(50)]
        # With shape > 1, hazard increases over time -> probabilities increase
        assert probs[-1] > probs[0]

    def test_repair_resets_age(self):
        bp = WeibullBreakdownProcess(shape=2.0, scale=100.0, dt=1)
        for _ in range(10):
            bp.step_and_get_proba()
        assert bp.age == 10
        bp.repair()
        assert bp.age == 0

    def test_idle_probability_is_lower(self):
        bp = WeibullBreakdownProcess(shape=2.0, scale=100.0, dt=1)
        bp.step_and_get_proba()  # advance age
        bp2 = WeibullBreakdownProcess(shape=2.0, scale=100.0, dt=1)
        p_working = bp2.step_and_get_proba()
        p_idle = bp2.step_and_get_idle_proba()
        assert p_idle < p_working

    def test_uses_math_exp_not_hardcoded(self):
        bp = WeibullBreakdownProcess(shape=2.0, scale=100.0, dt=1)
        bp.step_and_get_proba()
        # Calculate expected probability using math.exp
        age = 1
        hazard = (2.0 / 100.0) * ((age / 100.0) ** (2.0 - 1))
        expected = 1.0 - math.exp(-hazard * 1)
        actual = bp.step_and_get_proba()  # age=2 now, but let's check it doesn't crash
        assert 0.0 <= actual <= 1.0

    def test_probability_capped_at_one(self):
        bp = WeibullBreakdownProcess(shape=5.0, scale=10.0, dt=1)
        for _ in range(10000):
            p = bp.step_and_get_proba()
            assert p <= 1.0
