"""Tests for MCAEncoder."""

import numpy as np

from kata.entities.encoder.mca_encoder import MCAEncoder


class _FakeMachine:
    def __init__(self, mtype: str = "CNC"):
        self.mtype = mtype
        self.machine_id = 1


class _FakeRequest:
    def __init__(
        self,
        machine_type: str = "CNC",
        component_type: str | None = "motor",
        repair_time: float = 20.0,
    ):
        self.machine = _FakeMachine(machine_type)
        self.created_at = 0
        self._comp_type = component_type
        self._repair_time = repair_time

    def get_repair_time(self) -> float:
        return self._repair_time

    def get_failed_component_info(self):
        if self._comp_type is None:
            return None
        return {
            "component_type": self._comp_type,
            "component_id": f"{self._comp_type}_0",
            "repair_time": self._repair_time,
        }


class TestMCAEncoderUnfitted:
    def test_not_fitted_initially(self):
        enc = MCAEncoder()
        assert not enc.fitted

    def test_fallback_encode_works(self):
        enc = MCAEncoder(grid_shape=(10, 10))
        req = _FakeRequest()
        result = enc.encode(req)
        assert result.shape == (2,)
        assert result.dtype == np.intp
        assert all(0 <= c < 10 for c in result)

    def test_fallback_deterministic(self):
        enc = MCAEncoder(grid_shape=(10, 10))
        req1 = _FakeRequest(machine_type="A", component_type="x")
        req2 = _FakeRequest(machine_type="A", component_type="x")
        np.testing.assert_array_equal(enc.encode(req1), enc.encode(req2))


class TestMCAEncoderFitted:
    def _make_diverse_requests(self, n: int = 50) -> list[_FakeRequest]:
        types = ["CNC", "lathe", "drill", "grinder"]
        comps = ["motor", "spindle", "bearing", "belt", None]
        times = [3.0, 10.0, 25.0, 45.0, 100.0]
        requests = []
        for i in range(n):
            requests.append(
                _FakeRequest(
                    machine_type=types[i % len(types)],
                    component_type=comps[i % len(comps)],
                    repair_time=times[i % len(times)],
                )
            )
        return requests

    def test_fit_succeeds(self):
        enc = MCAEncoder(grid_shape=(10, 10), n_components=2)
        requests = self._make_diverse_requests()
        enc.fit(requests)
        assert enc.fitted

    def test_encode_after_fit(self):
        enc = MCAEncoder(grid_shape=(8, 8), n_components=2)
        requests = self._make_diverse_requests()
        enc.fit(requests)

        result = enc.encode(requests[0])
        assert result.shape == (2,)
        assert all(0 <= c < 8 for c in result)

    def test_different_requests_can_differ(self):
        enc = MCAEncoder(grid_shape=(10, 10), n_components=2)
        requests = self._make_diverse_requests(100)
        enc.fit(requests)

        coords = [tuple(enc.encode(r)) for r in requests]
        # Not all coordinates should be identical
        assert len(set(coords)) > 1

    def test_fit_with_too_few_requests(self):
        enc = MCAEncoder()
        enc.fit([_FakeRequest()])
        assert not enc.fitted  # too few to fit

    def test_3d_grid_shape(self):
        enc = MCAEncoder(grid_shape=(5, 5, 5), n_components=3)
        requests = self._make_diverse_requests()
        enc.fit(requests)
        result = enc.encode(requests[0])
        assert result.shape == (3,)
        assert all(0 <= c < 5 for c in result)


class TestMCAEncoderDiagnostics:
    def _diverse(self, n=80) -> list[_FakeRequest]:
        types = ["CNC", "lathe", "drill", "grinder", "press"]
        comps = ["motor", "spindle", "bearing", "belt", "sensor"]
        out = []
        for i in range(n):
            out.append(
                _FakeRequest(
                    machine_type=types[i % len(types)],
                    component_type=comps[(i * 3) % len(comps)],
                )
            )
        return out

    def test_diagnostics_exposed_after_fit(self):
        enc = MCAEncoder(grid_shape=(10, 10), n_components=2)
        enc.fit(self._diverse())
        d = enc.fit_diagnostics
        assert d["n_requests"] == 80
        assert d["n_unique_feature_tuples"] >= 5
        assert "eigenvalues" in d and len(d["eigenvalues"]) >= 1
        assert 0.0 <= d["cumulative_inertia"] <= 1.0 + 1e-9
        assert "explained_inertia" in d

    def test_well_trained_true_for_clean_data(self):
        enc = MCAEncoder(grid_shape=(10, 10), n_components=2)
        enc.fit(self._diverse())
        # Two independent categorical features with multiple levels each
        # → MCA captures essentially all of the (categorical) variance.
        assert enc.is_well_trained()
        assert enc.fit_diagnostics["cumulative_inertia"] > 0.5

    def test_summary_string_safe(self):
        enc = MCAEncoder()
        # Not fitted yet
        assert "not fitted" in enc.summary()
        enc.fit(self._diverse())
        assert "well_trained" in enc.summary()

    def test_singleton_dataset_does_not_fit(self):
        enc = MCAEncoder()
        # All requests share the same feature tuple
        reqs = [_FakeRequest(machine_type="A", component_type="motor") for _ in range(10)]
        enc.fit(reqs)
        assert not enc.fitted
        # Falls back to hash encoding deterministically
        a = enc.encode(reqs[0])
        b = enc.encode(reqs[1])
        np.testing.assert_array_equal(a, b)
