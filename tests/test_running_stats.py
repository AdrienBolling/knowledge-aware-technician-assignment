"""Tests for the RunningMeanStd helper."""

import numpy as np

from agents.networks.running_stats import RunningMeanStd


class TestRunningMeanStd:
    def test_defaults(self):
        rms = RunningMeanStd()
        assert rms.mean == 0.0
        assert rms.var == 1.0
        assert rms.count > 0

    def test_converges_to_population_stats(self):
        """Feeding samples drawn from N(mu, sigma) should converge to (mu, sigma**2)."""
        rng = np.random.default_rng(0)
        rms = RunningMeanStd()
        data = rng.normal(loc=5.0, scale=2.0, size=50_000)
        for x in data:
            rms.update(x)
        assert abs(rms.mean - 5.0) < 0.05
        assert abs(np.sqrt(rms.var) - 2.0) < 0.05

    def test_batch_update_matches_per_sample(self):
        """Batched and per-sample updates of identical data agree."""
        rng = np.random.default_rng(1)
        data = rng.normal(loc=-1.0, scale=0.5, size=200)

        rms_serial = RunningMeanStd()
        for x in data:
            rms_serial.update(x)

        rms_batch = RunningMeanStd()
        rms_batch.update(data)

        assert abs(rms_serial.mean - rms_batch.mean) < 1e-6
        assert abs(rms_serial.var - rms_batch.var) < 1e-6

    def test_state_dict_round_trip(self):
        rms = RunningMeanStd()
        rms.update(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
        state = rms.state_dict()
        rms2 = RunningMeanStd()
        rms2.load_state_dict(state)
        assert rms2.mean == rms.mean
        assert rms2.var == rms.var
        assert rms2.count == rms.count
