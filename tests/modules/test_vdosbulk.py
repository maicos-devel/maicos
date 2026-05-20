#!/usr/bin/env python
#
# Copyright (c) 2026 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the GNU Public Licence, v3 or any higher version
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for the VDOSBulk class."""

import sys
from pathlib import Path

import MDAnalysis as mda
import numpy as np
import pytest
from MDAnalysis.exceptions import NoDataError
from MDAnalysisTests.datafiles import TPR, XTC

from maicos import VDOSBulk

sys.path.append(str(Path(__file__).parents[1]))
from data import WATER_TPR_NPT, WATER_TRR_NPT  # noqa: E402


@pytest.fixture
def universe():
    """Return a water NPT universe with both positions and velocities."""
    return mda.Universe(WATER_TPR_NPT, WATER_TRR_NPT)


@pytest.fixture
def velocity_signal(universe):
    """Per-frame velocities for the full trajectory, shape (n_frames, n_atoms, 3)."""
    return np.array([universe.atoms.velocities.copy() for _ in universe.trajectory])


class TestSmokeRun:
    """Basic shape and finalisation checks."""

    def test_results_populated(self, universe):
        """``results`` carries the four expected arrays after a run."""
        ana = VDOSBulk(universe.atoms).run()
        assert hasattr(ana.results, "times")
        assert hasattr(ana.results, "vacf")
        assert hasattr(ana.results, "frequencies")
        assert hasattr(ana.results, "vdos")

    def test_vacf_normalized(self, universe):
        """The reported VACF is normalised so ``vacf[0] == 1``."""
        ana = VDOSBulk(universe.atoms).run()
        assert ana.results.vacf[0] == pytest.approx(1.0, abs=1e-12)

    def test_first_lag_is_zero(self, universe):
        """The time axis starts at lag zero."""
        ana = VDOSBulk(universe.atoms).run()
        assert ana.results.times[0] == pytest.approx(0.0)

    def test_shapes_consistent(self, universe):
        """VACF and VDOS array shapes match the requested resolutions."""
        ana = VDOSBulk(universe.atoms, n_frequencies=64).run()
        n_lags = ana.results.times.size
        assert ana.results.vacf.shape == (n_lags,)
        assert ana.results.frequencies.shape == (64,)
        assert ana.results.vdos.shape == (64,)

    def test_vdos_real(self, universe):
        """The VDOS is real and finite."""
        ana = VDOSBulk(universe.atoms).run()
        assert np.isrealobj(ana.results.vdos)
        assert np.all(np.isfinite(ana.results.vdos))


class TestCorrectness:
    """Compare against brute-force VACF on the same velocity stream."""

    def test_level0_matches_brute_force(self, universe, velocity_signal):
        """Level-0 VACF entries reproduce a direct sum over time origins."""
        ana = VDOSBulk(
            universe.atoms,
            correlator_num_levels=1,
            correlator_channels_per_level=16,
        ).run()
        m = ana.correlator_channels_per_level
        v = velocity_signal  # (n_frames, n_atoms, 3)
        n = v.shape[0]
        # Reduce over (n_atoms, 3) per frame pair, then average over time origins.
        expected_raw = np.empty(m)
        for j in range(m):
            expected_raw[j] = np.sum(v[: n - j] * v[j:]) / (n - j)
        expected = expected_raw / expected_raw[0]
        # Loose rtol because the brute-force and streaming sums use different
        # reduction orders; absolute agreement is < 1e-8.
        np.testing.assert_allclose(ana.results.vacf[:m], expected, atol=1e-8)

    def test_times_use_trajectory_dt(self, universe):
        """Lag times are reported in physical units of ``dt``."""
        ana = VDOSBulk(
            universe.atoms,
            correlator_num_levels=2,
            correlator_channels_per_level=8,
        ).run()
        # First few lags are 0, 1, 2 frames → 0, dt, 2*dt in physical units.
        dt = float(np.mean(np.diff(ana.times)))
        np.testing.assert_allclose(ana.results.times[:3], dt * np.arange(3))


class TestValidation:
    """Constructor validation and missing-velocity guard."""

    def test_no_velocities_raises(self):
        """Running on a trajectory without velocities raises ``NoDataError``."""
        u = mda.Universe(TPR, XTC)  # XTC carries positions only
        ana = VDOSBulk(u.atoms)
        with pytest.raises(NoDataError, match="velocities"):
            ana.run(stop=1)

    def test_bad_n_frequencies(self, universe):
        """An ``n_frequencies`` below 2 is rejected."""
        with pytest.raises(ValueError, match="n_frequencies"):
            VDOSBulk(universe.atoms, n_frequencies=1)


class TestCorrelatorWiring:
    """The correlator hyperparameters propagate through to the underlying object."""

    def test_settings_propagate(self, universe):
        """``correlator_*`` kwargs are forwarded to the underlying correlator."""
        ana = VDOSBulk(
            universe.atoms,
            correlator_num_levels=3,
            correlator_channels_per_level=8,
        ).run()
        c = ana._correlators["velocity"]
        assert c.num_levels == 3
        assert c.channels_per_level == 8

    def test_higher_levels_populate_on_long_run(self, universe):
        """Long-enough runs expose lags from coarsened levels."""
        # Trajectory has 101 frames; with m=8, level 1 starts at frame 2 and
        # level 2 at frame 4 — both should fill comfortably.
        ana = VDOSBulk(
            universe.atoms,
            correlator_num_levels=4,
            correlator_channels_per_level=8,
        ).run()
        # Beyond level-0's max lag of m-1=7, more lags exist only if higher
        # levels were reached.
        assert np.any(ana.lags > 7)


class TestSave:
    """The save() method writes both VACF and VDOS files."""

    def test_save_writes_both_files(self, universe, tmp_path, monkeypatch):
        """``save()`` emits the VACF and VDOS files using the configured prefix."""
        monkeypatch.chdir(tmp_path)
        VDOSBulk(universe.atoms, output_prefix="myvdos").run().save()
        assert (tmp_path / "myvdos_vacf.dat").exists()
        assert (tmp_path / "myvdos_vdos.dat").exists()

    def test_save_files_parseable(self, universe, tmp_path, monkeypatch):
        """Saved files round-trip through ``np.loadtxt`` and match the results."""
        monkeypatch.chdir(tmp_path)
        ana = VDOSBulk(universe.atoms, n_frequencies=32, output_prefix="vdos").run()
        ana.save()
        vacf = np.loadtxt(tmp_path / "vdos_vacf.dat")
        vdos = np.loadtxt(tmp_path / "vdos_vdos.dat")
        assert vacf.shape[1] == 2
        assert vdos.shape == (32, 2)
        np.testing.assert_allclose(vacf[:, 1], ana.results.vacf, rtol=1e-12)
        np.testing.assert_allclose(vdos[:, 1], ana.results.vdos, rtol=1e-12)
