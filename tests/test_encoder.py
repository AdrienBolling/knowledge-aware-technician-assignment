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

    def test_output_dtype_is_float(self):
        enc = HashEncoder(grid_shape=(10, 10))
        req = _FakeRequest("cnc", "spindle")
        result = enc.encode(req)
        # Embeddings are floats (not grid indices) — the grid does the
        # final coordinate mapping internally.
        assert np.issubdtype(result.dtype, np.floating)

    def test_values_within_embedding_bounds(self):
        enc = HashEncoder(grid_shape=(10, 10))
        req = _FakeRequest("cnc", "spindle")
        result = enc.encode(req)
        # Default bounds are [0, 100] per axis
        assert all(0.0 <= v < 100.0 for v in result)

    def test_custom_embedding_bounds(self):
        bounds = np.array([[-10.0, 10.0], [0.0, 1.0]])
        enc = HashEncoder(grid_shape=(10, 10), embedding_bounds=bounds)
        req = _FakeRequest("cnc", "spindle")
        result = enc.encode(req)
        assert -10.0 <= result[0] < 10.0
        assert 0.0 <= result[1] < 1.0

    def test_deterministic(self):
        enc = HashEncoder(grid_shape=(10, 10))
        req = _FakeRequest("drill", "motor")
        r1 = enc.encode(req)
        r2 = enc.encode(req)
        np.testing.assert_array_equal(r1, r2)

    def test_deterministic_across_instances(self):
        """BLAKE2-based hashing is stable across encoder instances."""
        enc1 = HashEncoder(grid_shape=(10, 10))
        enc2 = HashEncoder(grid_shape=(10, 10))
        req = _FakeRequest("drill", "motor")
        np.testing.assert_array_equal(enc1.encode(req), enc2.encode(req))

    def test_distinct_keys_land_in_distinct_grid_cells(self):
        """Across a handful of (mtype, ctype) pairs, the embeddings
        should map to several different grid cells — not all collapse
        to one corner like the old behaviour did.
        """
        enc = HashEncoder(grid_shape=(10, 10))
        keys = [
            ("CNC", "spindle"),
            ("Conveyor", "mechanical"),
            ("Assembly", "motor"),
            ("Welder", "torch"),
            ("Inspection", "sensor"),
            ("Assembly", "bearing"),
            ("CNC", "pump"),
        ]
        # Mimic the grid's coord mapping: scale by (bounds_hi - bounds_lo) → bin
        bounds = enc.embedding_bounds
        cells = set()
        for mt, ct in keys:
            emb = enc.encode(_FakeRequest(mt, ct))
            scaled = (emb - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0])
            cell = tuple(int(np.clip(s * 10, 0, 9)) for s in scaled)
            cells.add(cell)
        # Expect at least 5 of 7 distinct cells — with 100 cells and
        # uniform hash distribution, collisions are rare.
        assert len(cells) >= 5, f"too many hash collisions: {cells}"

    def test_no_component_uses_none_key(self):
        enc = HashEncoder(grid_shape=(10, 10))
        req = _FakeRequest("drill", component_type=None)
        result = enc.encode(req)
        assert result.shape == (2,)

    def test_embedding_bounds_shape_mismatch_raises(self):
        import pytest

        with pytest.raises(ValueError, match="dimensions"):
            HashEncoder(
                grid_shape=(10, 10),
                embedding_bounds=np.array([[0.0, 1.0]]),  # only 1 dim
            )


class TestLookupEncoder:
    def test_uses_lookup_when_key_exists(self):
        lookup = {("drill", "motor"): (35.0, 75.0)}
        enc = LookupEncoder(lookup=lookup, grid_shape=(10, 10))
        req = _FakeRequest("drill", "motor")
        result = enc.encode(req)
        np.testing.assert_array_equal(result, [35.0, 75.0])

    def test_falls_back_to_hash_for_unknown_key(self):
        enc = LookupEncoder(lookup={}, grid_shape=(10, 10))
        req = _FakeRequest("drill", "motor")
        result = enc.encode(req)
        assert result.shape == (2,)
        # Falls back to the bounded HashEncoder, so values land in [0, 100)
        assert all(0.0 <= v < 100.0 for v in result)


class TestDefaultEncoder:
    def test_ENCODER_singleton_is_hash_encoder(self):
        assert isinstance(ENCODER, HashEncoder)

    def test_ENCODER_works_with_request(self):
        req = _FakeRequest("drill", "motor")
        result = ENCODER.encode(req)
        assert isinstance(result, np.ndarray)
        assert np.issubdtype(result.dtype, np.floating)
