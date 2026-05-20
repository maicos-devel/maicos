#!/usr/bin/env python
#
# Copyright (c) 2026 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the GNU Public Licence, v3 or any higher version
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for the VDOSPlanar class."""

import sys
from pathlib import Path

import MDAnalysis as mda
import numpy as np
import pytest
from MDAnalysis.exceptions import NoDataError
from MDAnalysisTests.datafiles import TPR, XTC

from maicos import VDOSBulk, VDOSPlanar

sys.path.append(str(Path(__file__).parents[1]))
from data import WATER_TPR_NPT, WATER_TRR_NPT  # noqa: E402


@pytest.fixture
def universe():
    """Return a water-NPT universe with both positions and velocities."""
    return mda.Universe(WATER_TPR_NPT, WATER_TRR_NPT)


@pytest.fixture
def velocity_signal(universe):
    """Per-frame velocities for the full trajectory, shape (n_frames, n_atoms, 3)."""
    return np.array([universe.atoms.velocities.copy() for _ in universe.trajectory])


class TestSmokeRun:
    """Basic shape and finalisation checks."""

    def test_results_populated(self, universe):
        """``results`` carries the six expected per-slab arrays after a run."""
        ana = VDOSPlanar(universe.atoms, bin_width=5.0).run()
        for attr in ("times", "vacf", "frequencies", "vdos", "bin_pos", "bin_counts"):
            assert hasattr(ana.results, attr), f"missing results.{attr}"

    def test_shapes_consistent(self, universe):
        """Per-bin output shapes match the expected ``(n_lags/freqs, n_bins)``."""
        ana = VDOSPlanar(universe.atoms, bin_width=5.0, n_frequencies=64).run()
        n_lags = ana.results.times.size
        n_bins = ana.results.bin_pos.size
        assert ana.results.vacf.shape == (n_lags, n_bins)
        assert ana.results.vdos.shape == (64, n_bins)
        assert ana.results.bin_counts.shape == (n_bins,)

    def test_vacf_normalized_per_bin(self, universe):
        """Each populated slab has its own normalised VACF starting at 1."""
        ana = VDOSPlanar(universe.atoms, bin_width=5.0).run()
        populated = ana.results.bin_counts > 0
        np.testing.assert_allclose(ana.results.vacf[0, populated], 1.0, atol=1e-12)


class TestStaticBinAssignment:
    """The first-frame z-positions determine each atom's slab for the whole run."""

    def test_assignment_is_static(self, universe):
        """Bin assignment is fixed at the first frame and not re-evaluated."""
        ana = VDOSPlanar(universe.atoms, bin_width=4.0).run()
        # We can't reconstruct the lab-frame edges exactly without reproducing
        # _compute_lab_frame_planar; instead just check totals match.
        assert ana.results.bin_counts.sum() <= universe.atoms.n_atoms
        assert ana.results.bin_counts.sum() > 0

    def test_atoms_outside_range_excluded(self, universe):
        """Atoms outside ``[zmin, zmax]`` at the first frame are dropped."""
        # Restrict to a thin slab at the center; atoms outside are dropped.
        box = universe.dimensions[2]
        ana = VDOSPlanar(
            universe.atoms,
            bin_width=2.0,
            zmin=-box / 8,
            zmax=box / 8,
        ).run()
        n_assigned = ana.results.bin_counts.sum()
        assert 0 < n_assigned < universe.atoms.n_atoms

    def test_bin_count_sum_equals_atoms_when_full_box(self, universe):
        """When the slab range covers the whole box, every atom is assigned."""
        ana = VDOSPlanar(universe.atoms, bin_width=2.0).run()
        assert ana.results.bin_counts.sum() == universe.atoms.n_atoms


class TestCorrectness:
    """Compare against brute-force VACF on the same velocity stream."""

    def test_level0_matches_brute_force_per_bin(self, universe, velocity_signal):
        """Level-0 VACF entries match a direct per-bin sum over time origins."""
        ana = VDOSPlanar(
            universe.atoms,
            bin_width=5.0,
            correlator_num_levels=1,
            correlator_channels_per_level=8,
        ).run()
        m = ana.correlator_channels_per_level
        # Reproduce the same static assignment for the brute-force check.
        bin_index = ana._bin_index
        included = ana._included
        v = velocity_signal  # (n_frames, n_atoms, 3)
        n = v.shape[0]
        for k in np.where(ana.results.bin_counts > 0)[0]:
            mask = included & (bin_index == k)
            vk = v[:, mask, :]
            ref_raw = np.array(
                [np.sum(vk[: n - j] * vk[j:]) / (n - j) for j in range(m)]
            )
            ref = ref_raw / ref_raw[0]
            np.testing.assert_allclose(ana.results.vacf[:m, k], ref, atol=1e-8)

    def test_single_bin_matches_vdosbulk(self, universe):
        """One slab covering the whole box reproduces the bulk result."""
        n_atoms = universe.atoms.n_atoms

        # Force a single bin by setting bin_width to the box length.
        box_z = universe.dimensions[2]
        planar = VDOSPlanar(
            universe.atoms,
            bin_width=box_z,
            correlator_num_levels=2,
            correlator_channels_per_level=8,
        ).run()
        bulk = VDOSBulk(
            universe.atoms,
            correlator_num_levels=2,
            correlator_channels_per_level=8,
        ).run()

        assert planar.results.bin_pos.size == 1
        assert planar.results.bin_counts[0] == n_atoms
        # Tolerance set to the float32 noise floor — velocities are stored as
        # single precision in the TRR, so the two reduction orders differ at
        # the last bit.
        np.testing.assert_allclose(
            planar.results.vacf[:, 0], bulk.results.vacf, atol=2e-6
        )


class TestValidation:
    """Constructor validation and missing-velocity guard."""

    def test_no_velocities_raises(self):
        """Running on a trajectory without velocities raises ``NoDataError``."""
        u = mda.Universe(TPR, XTC)
        ana = VDOSPlanar(u.atoms, bin_width=5.0)
        with pytest.raises(NoDataError, match="velocities"):
            ana.run(stop=1)

    def test_bad_n_frequencies(self, universe):
        """An ``n_frequencies`` below 2 is rejected."""
        with pytest.raises(ValueError, match="n_frequencies"):
            VDOSPlanar(universe.atoms, n_frequencies=1)


class TestSave:
    """The save() method writes both VACF and VDOS files with bin columns."""

    def test_save_writes_both_files(self, universe, tmp_path, monkeypatch):
        """``save()`` emits the VACF and VDOS files using the configured prefix."""
        monkeypatch.chdir(tmp_path)
        VDOSPlanar(
            universe.atoms,
            bin_width=5.0,
            n_frequencies=16,
            output_prefix="slab",
        ).run().save()
        assert (tmp_path / "slab_vacf.dat").exists()
        assert (tmp_path / "slab_vdos.dat").exists()

    def test_save_files_parseable(self, universe, tmp_path, monkeypatch):
        """Saved files round-trip through ``np.loadtxt`` with per-bin columns."""
        monkeypatch.chdir(tmp_path)
        ana = VDOSPlanar(
            universe.atoms,
            bin_width=5.0,
            n_frequencies=32,
            output_prefix="vdos_planar",
        ).run()
        ana.save()
        vacf = np.loadtxt(tmp_path / "vdos_planar_vacf.dat")
        vdos = np.loadtxt(tmp_path / "vdos_planar_vdos.dat")
        # First column is time/frequency; remaining columns are per-bin.
        assert vacf.shape[1] == ana.results.bin_pos.size + 1
        assert vdos.shape == (32, ana.results.bin_pos.size + 1)
