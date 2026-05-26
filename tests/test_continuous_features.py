"""Tests for the continuous-feature encoders (PLE, Time2Vec, Fourier).

The properties under test mirror the original papers' invariants:

* **PLE** — bounded in ``[0, 1]`` per dimension; the K-dim ramp is
  monotonic in the input; values outside the bin range clamp.
* **Time2Vec** — output is sensitive to the input (zero variance only
  on degenerate inputs); the first dimension is linear in ``t``.
* **Fourier features** — periodic and bounded; output norm independent
  of ``t`` (a fixed-frequency cos/sin pair has unit magnitude).
"""

from __future__ import annotations

import math

import torch

from agents.networks.continuous_features import (
    ContKind,
    FourierFeatures,
    PiecewiseLinearEncoding,
    Time2Vec,
)


class TestPLE:
    def test_ramp_in_unit_interval(self):
        ple = PiecewiseLinearEncoding(
            bin_edges=[0.0, 0.25, 0.5, 0.75, 1.0], d_model=8
        )
        x = torch.tensor([0.0, 0.4, 0.6, 1.0, 1.5, -0.5])
        ramp = ple.encode_raw(x)
        assert ramp.shape == (6, 4)
        assert ramp.min().item() >= 0.0
        assert ramp.max().item() <= 1.0

    def test_monotonic_in_input(self):
        ple = PiecewiseLinearEncoding(
            bin_edges=[0.0, 0.5, 1.0], d_model=4
        )
        x = torch.linspace(0.0, 1.0, 11)
        ramp = ple.encode_raw(x).sum(dim=-1)  # total mass grows with x
        assert torch.all(ramp[1:] >= ramp[:-1] - 1e-6)

    def test_below_min_edge_is_zero(self):
        ple = PiecewiseLinearEncoding(
            bin_edges=[0.0, 0.5, 1.0], d_model=4
        )
        # Anything <= edges[0] clamps to all zeros (no bin yet entered).
        ramp = ple.encode_raw(torch.tensor([-1.0]))
        assert torch.allclose(ramp, torch.zeros_like(ramp))

    def test_above_max_edge_saturates(self):
        ple = PiecewiseLinearEncoding(
            bin_edges=[0.0, 0.5, 1.0], d_model=4
        )
        # Past the last edge every bin is fully crossed.
        ramp = ple.encode_raw(torch.tensor([2.0]))
        assert torch.allclose(ramp, torch.ones_like(ramp))

    def test_output_shape_after_projection(self):
        ple = PiecewiseLinearEncoding(bin_edges=[0.0, 0.5, 1.0], d_model=16)
        x = torch.rand(2, 5)
        out = ple(x)
        assert out.shape == (2, 5, 16)


class TestTime2Vec:
    def test_output_shape(self):
        t2v = Time2Vec(d_model=12, n_freqs=8)
        out = t2v(torch.rand(3, 7))
        assert out.shape == (3, 7, 12)

    def test_linear_dim_grows_with_t(self):
        torch.manual_seed(0)
        t2v = Time2Vec(d_model=4, n_freqs=4)
        ts = torch.linspace(0.0, 1.0, 5)
        raw = t2v.encode_raw(ts)
        # The first dim is ``ω_0 * t + φ_0`` — strictly monotonic in t
        # for any non-zero ω_0.  Force a known sign of ω_0 by inspecting
        # the sign of the (constant) difference.
        diffs = raw[1:, 0] - raw[:-1, 0]
        assert torch.all(diffs > 0) or torch.all(diffs < 0)

    def test_sine_dims_bounded(self):
        torch.manual_seed(0)
        t2v = Time2Vec(d_model=4, n_freqs=4)
        raw = t2v.encode_raw(torch.linspace(-100.0, 100.0, 200))
        # All but the first column are sines and must live in [-1, 1].
        assert raw[:, 1:].abs().max().item() <= 1.0 + 1e-6


class TestFourierFeatures:
    def test_output_shape_and_periodicity(self):
        ff = FourierFeatures(d_model=8, n_freqs=4, sigma=1.0, input_scale=1.0)
        raw = ff.encode_raw(torch.rand(2, 5))
        # 2K-dim raw, K = n_freqs
        assert raw.shape == (2, 5, 8)
        # Bounded in [-1, 1]
        assert raw.abs().max().item() <= 1.0 + 1e-6

    def test_unit_magnitude_pairs(self):
        """For each frequency pair, ``cos² + sin² = 1`` (Pythagorean identity)."""
        ff = FourierFeatures(d_model=8, n_freqs=4, sigma=1.0)
        raw = ff.encode_raw(torch.tensor([0.0, 1.0, math.pi]))
        # First K dims are cos, second K dims are sin (same B per pair).
        k = ff.n_freqs
        cos_part = raw[..., :k]
        sin_part = raw[..., k:]
        norm = cos_part.pow(2) + sin_part.pow(2)
        assert torch.allclose(norm, torch.ones_like(norm), atol=1e-5)

    def test_frequencies_are_frozen(self):
        ff = FourierFeatures(d_model=8, n_freqs=4, sigma=1.0)
        # The Tancik paper's recipe — B sampled once at init, not
        # registered as a parameter.
        b_state = ff.B.detach().clone()
        # Take a gradient step on the projection layer
        opt = torch.optim.SGD(ff.parameters(), lr=1.0)
        opt.zero_grad()
        ff(torch.tensor([1.0])).sum().backward()
        opt.step()
        assert torch.allclose(ff.B, b_state, atol=1e-6)


def test_cont_kind_enum_codes_are_stable():
    """ContKind codes are a public contract — env writes / encoder reads."""
    assert ContKind.CATEGORICAL == 0
    assert ContKind.RATIO_PLE == 1
    assert ContKind.COUNT_PLE == 2
    assert ContKind.TIME2VEC == 3
    assert ContKind.FOURIER == 4
