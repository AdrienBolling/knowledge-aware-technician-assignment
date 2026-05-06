"""Tests for MCAEncoder."""

import numpy as np

from kata.entities.encoder.mca_encoder import MCAEncoder, _bucket_repair_time


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


class TestBucketRepairTime:
    def test_buckets(self):
        assert _bucket_repair_time(3) == "very_short"
        assert _bucket_repair_time(10) == "short"
        assert _bucket_repair_time(25) == "medium"
        assert _bucket_repair_time(45) == "long"
        assert _bucket_repair_time(100) == "very_long"


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
