#!/usr/bin/env python
#
# Copyright (c) 2026 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the GNU Public Licence, v3 or any higher version
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for the streaming multi-tau correlator."""

import numpy as np
import pytest

from maicos.lib.correlator import MultiTauCorrelator


class TestLagSchedule:
    """Tests for the construction of the lag grid."""

    def test_default_schedule(self):
        c = MultiTauCorrelator(num_levels=4, channels_per_level=8)
        expected = [
            0, 1, 2, 3, 4, 5, 6, 7,            # level 0
            4 * 2, 5 * 2, 6 * 2, 7 * 2,         # level 1
            4 * 4, 5 * 4, 6 * 4, 7 * 4,         # level 2
            4 * 8, 5 * 8, 6 * 8, 7 * 8,         # level 3
        ]
        np.testing.assert_array_equal(c.lags, expected)

    def test_lag_count(self):
        # Total = m + (p-1) * m / 2
        c = MultiTauCorrelator(num_levels=5, channels_per_level=16)
        assert c.n_lags == 16 + 4 * 8
        assert c.lags.size == c.n_lags

    def test_monotonic(self):
        c = MultiTauCorrelator(num_levels=10, channels_per_level=16)
        assert np.all(np.diff(c.lags) > 0)

    def test_single_level(self):
        c = MultiTauCorrelator(num_levels=1, channels_per_level=8)
        np.testing.assert_array_equal(c.lags, np.arange(8))


class TestValidation:
    """Tests for constructor argument validation."""

    def test_odd_channels(self):
        with pytest.raises(ValueError, match="even"):
            MultiTauCorrelator(channels_per_level=7)

    def test_too_few_channels(self):
        with pytest.raises(ValueError, match="at least 2"):
            MultiTauCorrelator(channels_per_level=0)

    def test_zero_levels(self):
        with pytest.raises(ValueError, match="at least 1"):
            MultiTauCorrelator(num_levels=0)

    def test_wrong_shape(self):
        c = MultiTauCorrelator(shape=(3,))
        with pytest.raises(ValueError, match="shape"):
            c.add(np.zeros(4))


class TestCorrectness:
    """Numerical correctness against brute-force references."""

    def test_constant_signal(self):
        """Constant signal x = c gives C(tau) = c^2 for every populated lag."""
        c = MultiTauCorrelator(num_levels=4, channels_per_level=4)
        for _ in range(64):
            c.add(2.0)
        corr = c.correlation
        counts = c.counts
        # All lags that received any pair must equal 4.0.
        np.testing.assert_allclose(corr[counts > 0], 4.0)

    def test_level0_matches_brute_force(self):
        """Level-0 lags match a brute-force autocorrelation exactly."""
        rng = np.random.default_rng(1)
        n = 200
        m = 8
        x = rng.normal(size=n)
        c = MultiTauCorrelator(num_levels=1, channels_per_level=m)
        for v in x:
            c.add(v)

        expected = np.empty(m)
        for j in range(m):
            expected[j] = np.sum(x[: n - j] * x[j:]) / (n - j)
        np.testing.assert_allclose(c.correlation[:m], expected, rtol=1e-12)

    def test_counts_at_level0(self):
        """count[0, j] = n_inserted - j for j < n_inserted."""
        c = MultiTauCorrelator(num_levels=1, channels_per_level=8)
        n = 20
        for _ in range(n):
            c.add(1.0)
        expected_counts = np.array([n - j for j in range(8)])
        np.testing.assert_array_equal(c.counts, expected_counts)

    def test_white_noise(self):
        """White noise: C(0) ~ var, C(tau > 0) ~ 0."""
        rng = np.random.default_rng(0)
        n = 20000
        x = rng.normal(size=n)
        c = MultiTauCorrelator(num_levels=6, channels_per_level=8)
        for v in x:
            c.add(v)
        corr = c.correlation
        var = x.var()
        # C(0) = <x^2> approx var for zero-mean.
        assert abs(corr[0] - var) < 0.05 * var
        # Mid-range lag (still level 0) should be close to zero.
        assert abs(corr[5]) < 0.05 * var

    def test_ar1_decay_level0(self):
        """AR(1) process: C(j)/C(0) ~ alpha^j at level 0."""
        rng = np.random.default_rng(4)
        alpha = 0.6
        n = 200000
        eps = rng.normal(size=n)
        x = np.empty(n)
        x[0] = eps[0]
        for i in range(1, n):
            x[i] = alpha * x[i - 1] + np.sqrt(1.0 - alpha**2) * eps[i]
        c = MultiTauCorrelator(num_levels=1, channels_per_level=8)
        for v in x:
            c.add(v)
        corr = c.correlation
        normed = corr / corr[0]
        expected = alpha ** np.arange(8)
        np.testing.assert_allclose(normed, expected, atol=0.02)

    def test_coarsened_constant_at_higher_levels(self):
        """Coarsening of a constant signal preserves the value at every level."""
        c = MultiTauCorrelator(num_levels=5, channels_per_level=8)
        # Need enough frames to populate the upper levels: 2**(p-1) * m = 16*8 = 128.
        for _ in range(256):
            c.add(3.0)
        counts = c.counts
        corr = c.correlation
        # Every level should be populated.
        assert np.all(counts > 0)
        np.testing.assert_allclose(corr, 9.0)


class TestArrayObservables:
    """Tests for array-valued signals."""

    def test_vector_shape(self):
        rng = np.random.default_rng(2)
        n = 200
        N = 3
        m = 8
        x = rng.normal(size=(n, N))
        c = MultiTauCorrelator(shape=(N,), num_levels=1, channels_per_level=m)
        for i in range(n):
            c.add(x[i])
        corr = c.correlation
        assert corr.shape == (m, N)
        for j in range(m):
            for d in range(N):
                ref = np.sum(x[: n - j, d] * x[j:, d]) / (n - j)
                np.testing.assert_allclose(corr[j, d], ref, rtol=1e-12)

    def test_2d_shape(self):
        """Signal of shape (N, d) is correlated element-wise."""
        rng = np.random.default_rng(5)
        n = 150
        N, d = 4, 3
        m = 8
        x = rng.normal(size=(n, N, d))
        c = MultiTauCorrelator(shape=(N, d), num_levels=1, channels_per_level=m)
        for i in range(n):
            c.add(x[i])
        corr = c.correlation
        assert corr.shape == (m, N, d)
        for j in range(m):
            ref = np.einsum("tij,tij->ij", x[: n - j], x[j:]) / (n - j)
            np.testing.assert_allclose(corr[j], ref, rtol=1e-12)


class TestComplex:
    """Tests for complex-valued signals."""

    def test_complex_autocorrelation(self):
        """Complex signal: uses <conj(A(0)) * A(tau)>."""
        rng = np.random.default_rng(3)
        n = 100
        m = 4
        x = rng.normal(size=n) + 1j * rng.normal(size=n)
        c = MultiTauCorrelator(
            num_levels=1, channels_per_level=m, dtype=np.complex128
        )
        for v in x:
            c.add(v)
        for j in range(m):
            ref = np.sum(np.conj(x[: n - j]) * x[j:]) / (n - j)
            np.testing.assert_allclose(c.correlation[j], ref, rtol=1e-12)

    def test_complex_zero_lag_is_real(self):
        """C(0) for a complex signal is real and equals <|A|^2>."""
        rng = np.random.default_rng(6)
        n = 500
        x = rng.normal(size=n) + 1j * rng.normal(size=n)
        c = MultiTauCorrelator(num_levels=1, channels_per_level=4, dtype=np.complex128)
        for v in x:
            c.add(v)
        c0 = c.correlation[0]
        np.testing.assert_allclose(c0.imag, 0.0, atol=1e-12)
        np.testing.assert_allclose(c0.real, np.mean(np.abs(x) ** 2), rtol=1e-12)


class TestEmpty:
    """Tests for short-stream behaviour."""

    def test_no_frames(self):
        c = MultiTauCorrelator(num_levels=3, channels_per_level=4)
        counts = c.counts
        assert np.all(counts == 0)
        # Correlation has NaNs where count == 0.
        assert np.all(np.isnan(c.correlation))

    def test_short_signal_empty_higher_levels(self):
        """With too few frames, only level 0 partially populates."""
        c = MultiTauCorrelator(num_levels=5, channels_per_level=8)
        for _ in range(3):
            c.add(1.0)
        counts = c.counts
        # Level 0, lags 0..2 have data; 3..7 do not.
        assert np.all(counts[:3] > 0)
        assert np.all(counts[3:8] == 0)
        # Higher levels stay empty.
        assert np.all(counts[8:] == 0)

    def test_n_frames_property(self):
        c = MultiTauCorrelator(num_levels=2, channels_per_level=4)
        assert c.n_frames == 0
        for _ in range(7):
            c.add(0.5)
        assert c.n_frames == 7


class TestMemoryFootprint:
    """Sanity test for the memory-independence claim."""

    def test_buffer_size_independent_of_stream_length(self):
        """Buffer size depends only on (p, m, shape), not on n_inserted."""
        c = MultiTauCorrelator(shape=(10,), num_levels=4, channels_per_level=8)
        n_bytes_before = c._buffer.nbytes + c._accum.nbytes
        for _ in range(5000):
            c.add(np.ones(10))
        n_bytes_after = c._buffer.nbytes + c._accum.nbytes
        assert n_bytes_before == n_bytes_after
