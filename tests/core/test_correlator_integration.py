#!/usr/bin/env python
#
# Copyright (c) 2026 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the GNU Public Licence, v3 or any higher version
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for the multi-tau correlator hooks on :class:`AnalysisBase`."""

import sys
from pathlib import Path

import MDAnalysis as mda
import numpy as np
import pytest
from MDAnalysisTests.datafiles import TPR, XTC

from maicos.core import AnalysisBase
from maicos.lib.correlator import MultiTauCorrelator

sys.path.append(str(Path(__file__).parents[1]))


def _base_kwargs(**overrides):
    """Standard minimal kwargs for the :class:`AnalysisBase` constructor."""
    kwargs = dict(
        unwrap=False,
        pack=True,
        refgroup=None,
        jitter=0.0,
        concfreq=0,
        wrap_compound="atoms",
    )
    kwargs.update(overrides)
    return kwargs


class VolumeCorr(AnalysisBase):
    """Streams the box volume (a scalar) into the multi-tau correlator."""

    def __init__(self, atomgroup, **kwargs):
        super().__init__(atomgroup=atomgroup, **_base_kwargs(**kwargs))

    def _single_frame(self):
        self._corr.volume = float(self._ts.volume)
        return float(self._ts.volume)


class PositionsCorr(AnalysisBase):
    """Streams atom positions, shape (n_atoms, 3)."""

    def __init__(self, atomgroup, **kwargs):
        super().__init__(atomgroup=atomgroup, **_base_kwargs(**kwargs))

    def _single_frame(self):
        p = self.atomgroup.positions.copy()
        self._corr.position = p
        return float(np.mean(p * p))


class TwoKeyCorr(AnalysisBase):
    """Two independent signals correlated in the same pass."""

    def __init__(self, atomgroup, **kwargs):
        super().__init__(atomgroup=atomgroup, **_base_kwargs(**kwargs))

    def _single_frame(self):
        self._corr.volume = float(self._ts.volume)
        self._corr.first_pos = float(self.atomgroup.positions[0, 0])
        return 0.0


class NoCorr(AnalysisBase):
    """Reference analysis that never populates ``_corr``."""

    def __init__(self, atomgroup):
        super().__init__(atomgroup=atomgroup, **_base_kwargs())

    def _single_frame(self):
        return 0.0


class PhaseCorr(AnalysisBase):
    """Streams a complex per-atom phase factor."""

    def __init__(self, atomgroup, q_vec, **kwargs):
        super().__init__(atomgroup=atomgroup, **_base_kwargs(**kwargs))
        self._q = np.asarray(q_vec, dtype=np.float64)

    def _single_frame(self):
        phase = np.exp(1j * (self.atomgroup.positions @ self._q))
        self._corr.phase = phase
        return 0.0


@pytest.fixture
def universe():
    """Return a small MDAnalysis universe for the correlator tests."""
    return mda.Universe(TPR, XTC)


class TestActivation:
    """Correlator attributes only appear when a module populates ``_corr``."""

    def test_no_corr_leaves_attributes_unset(self, universe):
        """An analysis that never touches ``_corr`` exposes no correlator attrs."""
        ana = NoCorr(universe.atoms).run(stop=10)
        assert not hasattr(ana, "lags")
        assert not hasattr(ana, "lag_counts")
        assert not hasattr(ana, "correlation")
        # The internal store exists but is empty.
        assert ana._correlators == {}

    def test_populated_corr_creates_attributes(self, universe):
        """Setting any ``_corr`` key adds the lag/count/correlation attributes."""
        ana = VolumeCorr(universe.atoms).run(stop=10)
        assert hasattr(ana, "lags")
        assert hasattr(ana, "lag_counts")
        assert hasattr(ana, "correlation")
        assert "volume" in ana.correlation


class TestLifecycle:
    """The base class allocates the correlator lazily and feeds every frame."""

    def test_correlator_sized_from_first_frame(self, universe):
        """The correlator picks up its shape from the first frame's payload."""
        ana = PositionsCorr(universe.atoms).run(stop=5)
        corr = ana._correlators["position"]
        assert isinstance(corr, MultiTauCorrelator)
        assert corr.shape == universe.atoms.positions.shape
        assert corr.n_frames == 5

    def test_correlator_settings_propagate(self, universe):
        """``correlator_*`` constructor kwargs reach the underlying correlator."""
        ana = VolumeCorr(
            universe.atoms,
            correlator_num_levels=4,
            correlator_channels_per_level=8,
        ).run(stop=20)
        c = ana._correlators["volume"]
        assert c.num_levels == 4
        assert c.channels_per_level == 8

    def test_all_frames_ingested(self, universe):
        """Every frame in the iteration window feeds the correlator."""
        ana = VolumeCorr(universe.atoms).run(stop=7)
        assert ana._correlators["volume"].n_frames == ana.n_frames == 7


class TestResults:
    """Exposed attributes match the underlying correlator output."""

    def test_shapes_match(self, universe):
        """Exposed array shapes line up with ``(n_lags, *signal_shape)``."""
        ana = PositionsCorr(universe.atoms).run(stop=15)
        n_lags = ana.lags.size
        assert ana.lag_counts.shape == (n_lags,)
        assert ana.correlation.position.shape == (
            n_lags,
            *universe.atoms.positions.shape,
        )

    def test_lags_are_in_frame_units(self, universe):
        """Lag values are reported as integer frame offsets, starting at 0."""
        ana = VolumeCorr(universe.atoms).run(stop=20)
        # Level 0 lags 0..min(m-1, n-1) must always be present.
        assert ana.lags[0] == 0
        np.testing.assert_array_equal(ana.lags[:5], np.arange(5))

    def test_valid_lag_mask_drops_empty(self, universe):
        """Lags with zero counts are not exposed in the public arrays."""
        ana = VolumeCorr(
            universe.atoms,
            correlator_num_levels=6,
            correlator_channels_per_level=8,
        ).run(stop=5)
        # All exposed lags must have count > 0.
        assert np.all(ana.lag_counts > 0)
        # The full grid has more lags than we exposed.
        full = ana._correlators["volume"]
        assert ana.lags.size < full.n_lags

    def test_correlation_values_match_brute_force(self, universe):
        """Level-0 entries match a direct autocorrelation of the streamed signal."""
        ana = VolumeCorr(
            universe.atoms,
            correlator_num_levels=1,
            correlator_channels_per_level=8,
        ).run()
        n = ana.n_frames
        signal = np.array([ts.volume for ts in universe.trajectory], dtype=np.float64)
        m = ana._correlators["volume"].channels_per_level
        expected = np.array(
            [np.sum(signal[: n - j] * signal[j:]) / (n - j) for j in range(m)]
        )
        # Only the lags that exist (count > 0) are exposed.
        np.testing.assert_allclose(ana.correlation.volume[:m], expected, rtol=1e-12)


class TestMultipleKeys:
    """Multiple ``_corr`` entries get independent correlators sharing one lag grid."""

    def test_two_signals_independent_storage(self, universe):
        """Distinct ``_corr`` keys produce distinct entries in ``correlation``."""
        ana = TwoKeyCorr(universe.atoms).run(stop=15)
        assert set(ana.correlation.keys()) == {"volume", "first_pos"}
        # Both signals see the same ingestion schedule → identical lag grids.
        assert ana.correlation.volume.shape[0] == ana.correlation.first_pos.shape[0]
        assert ana.correlation.volume.shape[0] == ana.lags.size

    def test_signals_not_mixed(self, universe):
        """Two signals streamed in the same pass do not contaminate each other."""
        ana = TwoKeyCorr(universe.atoms).run(stop=20)
        # The two signals have different magnitudes — they must remain distinct.
        assert ana.correlation.volume[0] != ana.correlation.first_pos[0]


class TestComplexSignal:
    """Complex-valued ``_corr`` entries propagate complex dtype through."""

    def test_complex_dtype_preserved(self, universe):
        """Complex inputs yield complex-valued correlation output."""
        ana = PhaseCorr(universe.atoms, q_vec=[0.5, 0.0, 0.0]).run(stop=10)
        assert np.iscomplexobj(ana.correlation.phase)
        # C(0) for a phase factor is |A|^2 = N (sum over atoms of 1).
        c0 = ana.correlation.phase[0]
        np.testing.assert_allclose(c0.imag, 0.0, atol=1e-10)
        np.testing.assert_allclose(c0.real, 1.0, atol=1e-10)


class TestConcludeOrdering:
    """``self.correlation`` is available to user ``_conclude``."""

    def test_correlation_visible_in_conclude(self, universe):
        """``self.correlation`` is populated before subclass ``_conclude`` runs."""
        captured = {}

        class CaptureCorr(AnalysisBase):
            def __init__(self, atomgroup):
                super().__init__(atomgroup=atomgroup, **_base_kwargs())

            def _single_frame(self):
                self._corr.volume = float(self._ts.volume)
                return 0.0

            def _conclude(self):
                # Must be visible *before* the user _conclude runs.
                captured["correlation"] = self.correlation.volume.copy()
                captured["lags"] = self.lags.copy()

        CaptureCorr(universe.atoms).run(stop=10)
        assert "correlation" in captured
        assert captured["lags"][0] == 0
