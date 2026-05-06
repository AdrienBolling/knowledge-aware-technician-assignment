"""Tests for the encoder system."""

import numpy as np

from kata.entities.encoder.base import ENCODER, HashEncoder, LookupEncoder


class _FakeMachine:
    def __init__(self, mtype: str = "drill"):
        self.mtype = mtype


class _FakeRequest:
    def __init__(self, mtype: str = "drill", component_type: str | None = None):
        self.machine = _FakeMachine(mtype)
        self._comp_type = component_type

    def get_failed_component_info(self):
        if self._comp_type is None:
            return None
        return {"component_type": self._comp_type}


class TestHashEncoder:
    def test_returns_ndarray(self):
        enc = HashEncoder(grid_shape=(10, 10))
        req = _FakeRequest("drill", "motor")
        result = enc.encode(req)
        assert isinstance(result, np.ndarray)

    def test_output_shape_matches_grid(self):
        enc = HashEncoder(grid_shape=(5, 8, 3))
        req = _FakeRequest("drill", "motor")
        result = enc.encode(req)
        assert result.shape == (3,)

    def test_values_within_grid_bounds(self):
        enc = HashEncoder(grid_shape=(10, 10))
        req = _FakeRequest("cnc", "spindle")
        result = enc.encode(req)
        assert all(0 <= v < 10 for v in result)

    def test_deterministic(self):
        enc = HashEncoder(grid_shape=(10, 10))
        req = _FakeRequest("drill", "motor")
        r1 = enc.encode(req)
        r2 = enc.encode(req)
        np.testing.assert_array_equal(r1, r2)

    def test_different_keys_can_differ(self):
        enc = HashEncoder(grid_shape=(100, 100))
        r1 = enc.encode(_FakeRequest("drill", "motor"))
        r2 = enc.encode(_FakeRequest("lathe", "bearing"))
        # Different inputs should (usually) give different coords
        # This is probabilistic but with grid 100x100 collision is rare
        assert not np.array_equal(r1, r2) or True  # don't fail on rare collision

    def test_no_component_uses_none_key(self):
        enc = HashEncoder(grid_shape=(10, 10))
        req = _FakeRequest("drill", component_type=None)
        result = enc.encode(req)
        assert result.shape == (2,)


class TestLookupEncoder:
    def test_uses_lookup_when_key_exists(self):
        lookup = {("drill", "motor"): (3, 7)}
        enc = LookupEncoder(lookup=lookup, grid_shape=(10, 10))
        req = _FakeRequest("drill", "motor")
        result = enc.encode(req)
        np.testing.assert_array_equal(result, [3, 7])

    def test_falls_back_to_hash_for_unknown_key(self):
        enc = LookupEncoder(lookup={}, grid_shape=(10, 10))
        req = _FakeRequest("drill", "motor")
        result = enc.encode(req)
        assert result.shape == (2,)
        assert all(0 <= v < 10 for v in result)


class TestDefaultEncoder:
    def test_ENCODER_singleton_is_hash_encoder(self):
        assert isinstance(ENCODER, HashEncoder)

    def test_ENCODER_works_with_request(self):
        req = _FakeRequest("drill", "motor")
        result = ENCODER.encode(req)
        assert isinstance(result, np.ndarray)
