#!/usr/bin/env python
#
# Copyright (c) 2025 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the GNU Public Licence, v3 or any higher version
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for the DielectricPlanar class."""

import sys
from pathlib import Path

import MDAnalysis as mda
import numpy as np
import pytest
import sympy as sp
from numpy.testing import assert_allclose, assert_equal

from maicos import DielectricPlanar

sys.path.append(str(Path(__file__).parents[1]))

from data import (  # noqa: E402
    DIPOLE_GRO,
    DIPOLE_ITP,
    WATER_2F_TRR_NPT,
    WATER_TPR_NPT,
    WATER_TRR_NPT,
)
from util import error_prop  # noqa: E402


def dipoles(positions, orientations):
    """Atomgroup consisting of defined dipoles.

    Create MDA universe with dipole molecules inside a 10 Å x 10 Å x 10 Å box cubic box.
    """
    template = mda.Universe(DIPOLE_ITP, DIPOLE_GRO, topology_format="itp")

    dipoles = []
    for position, orientation in zip(positions, orientations, strict=True):
        position = np.array(position)
        orientation = np.array(orientation)
        dipole = template.copy()
        # The dipole from the itp file is oriented in the x-direction
        angle = np.arccos(orientation[0] / np.linalg.norm(orientation))
        direction = np.cross(np.array([1, 0, 0]), np.array(orientation))
        # Don't rotate already correctly oriented dipoles
        if np.all(np.linalg.norm(orientation) * orientation == [1, 0, 0]):
            pass
        else:
            dipole.atoms.rotateby(np.rad2deg(angle), direction)
        dipole.atoms.translate(position)
        dipole.atoms.residues.molnums = [len(dipoles)]
        dipoles.append(dipole.atoms)
    u = mda.Merge(*dipoles)
    u.dimensions = [10, 10, 10, 90, 90, 90]
    return u.atoms


class TestDielectricPlanar:
    """Tests for the DielectricPlanar class.

    Number of times DielectricPlanar broke: ||||

    If you are reading this, most likely you are investigating a bug in the
    DielectricPlanar class. To calculate the local electric permittivity in a system,
    the module needs two quantities: the total dipole moment and the local dipole moment
    density.

    The total dipole moment is the sum of the charges times the positions of the atoms.
    It is checked first in every test case, so make sure this is correct.

    The local dipole moment density is determined by the virtual cutting method. In
    short, charges are tallied in bins according to their positions and the resulting
    histogram is integrated. (see Schlaich et al, Phys. Rev. Lett. 117, 048001 and its
    SI for more info.)

    The correctness of the local dipole moment density is checked by integrating it and
    comparing it to the total dipole moment (which should be equal). Some things that
    broke this method in the past where:
        - Incorrect treatment of the charges at the box edges. Molecules have to be
          unwrapped to keep the system charge neutral overall. Charges that cross over
          the box limits in the lateral direction are capped (i.e. shifted) to the box
          edge so they will land in the first/last bin when doing the charge histograms.
        - For the parallel component, the positions of charges in the normal direction
          are shifted to the center of charge of the molecule they belong to. If
          something goes wrong here and charges are shifted out of the box, they are no
          longer counted in the histogram.
    """

    @pytest.fixture
    def ag(self):
        """Import MDA universe."""
        u = mda.Universe(WATER_TPR_NPT, WATER_TRR_NPT)
        return u.atoms

    @pytest.fixture
    def ag_two_frames(self):
        """Import MDA universe, single frame."""
        u = mda.Universe(WATER_TPR_NPT, WATER_2F_TRR_NPT)
        return u.atoms

    @pytest.mark.parametrize(
        ("orientation", "M_par", "M_perp"),
        [
            ([0, 0, 1], [0, 0], 1),
            ([1, 0, 0], [1, 0], 0),
            ([0, 1, 0], [0, 1], 0),
            ([1, 1, 1], [1 / np.sqrt(3), 1 / np.sqrt(3)], 1 / np.sqrt(3)),
        ],
    )
    def test_single_dipole_orientations(self, orientation, M_par, M_perp):
        """Test the dipole density with a single dipole.

        This test places a single dipole (two unit charges separated by a 1 Å bond) with
        dipole moment 1 eÅ in a box of size 10 Å x 10 Å x 10 Å.

        The dipole is oriented in a given direction and the computed total dipole moment
        of the system is checked against the expected value.

        To check if the virtual cutting method produces the right results, the local
        dipole moment density is integrated over the entire system and checked against
        total dipole moment.
        """
        dipole = dipoles([[5, 5, 5]], [orientation])
        # very fine binning to get the correct value for the dipole
        eps = DielectricPlanar(dipole, bin_width=0.01, vcutwidth=0.01)
        eps.run()
        # Check the total dipole moment of the system
        assert_allclose(eps._obs.M_par, M_par, rtol=0.1)
        assert_allclose(eps._obs.M_perp, M_perp, rtol=0.1)
        # Check the local dipole moment density by integrating it over the
        # volume and comparing with the total dipole moment of the system.
        bin_volume = eps.means.bin_volume[0]
        assert_allclose(np.sum(eps._obs.m_par, axis=0) * bin_volume, M_par, rtol=0.1)
        assert_allclose(np.sum(eps._obs.m_perp, axis=0) * bin_volume, M_perp, rtol=0.1)

    @pytest.mark.parametrize("selection", [1, 2])
    @pytest.mark.parametrize(
        ("orientation", "M_par", "M_perp"),
        [
            ([0, 0, 1], [0, 0], 1),
            ([1, 0, 0], [1, 0], 0),
            ([0, 1, 0], [0, 1], 0),
            ([1, 1, 1], [1 / np.sqrt(3), 1 / np.sqrt(3)], 1 / np.sqrt(3)),
        ],
    )
    def test_multiple_dipole_orientations(self, selection, orientation, M_par, M_perp):
        """Test the dipole moment density with multiple dipoles.

        This test places a grid of 5x5x5 dipoles (two unit charges separated by a 1 Å
        bond) with dipole moment 1 eÅ in a box of size 10 Å x 10 Å x 10 Å.

        The dipoles are oriented in a given direction and the computed total dipole
        moment of the system is checked against the expected value.

        To check if the virtual cutting method produces the right results, the local
        dipole moment density is integrated over the entire system and checked against
        total dipole moment.

        This test is a variation of `test_single_dipole_orientations` to catch problems
        with system scaling of shifting that are not catched by the single dipole test.
        For example if some positions of charges are erroneously shifted out of the
        system.

        """
        xx, yy, zz = np.meshgrid(
            np.arange(0, 10, 2) + 1, np.arange(0, 10, 2) + 1, np.arange(0, 10, 2) + 1
        )

        pos = np.array([xx.flatten(), yy.flatten(), zz.flatten()]).T.tolist()
        n_dipoles = len(pos)
        dipole = dipoles(pos, [orientation] * n_dipoles)

        n = int(len(dipole) / selection) + 1 if selection == 2 else len(dipole)
        # very fine binning to get the correct value for the dipole
        eps = DielectricPlanar(dipole[:n], bin_width=0.01, vcutwidth=0.01)
        eps.run()
        # Check the total dipole moment of the system
        assert_allclose(eps._obs.M_par, np.multiply(M_par, n_dipoles), rtol=0.1)
        assert_allclose(eps._obs.M_perp, np.multiply(M_perp, n_dipoles), rtol=0.1)
        # Check the local dipole moment density by integrating it over the
        # volume and comparing with the total dipole moment of the system.
        bin_volume = eps.means.bin_volume[0]
        assert_allclose(
            np.sum(eps._obs.m_par, axis=0) * bin_volume,
            np.multiply(M_par, n_dipoles / selection),
            rtol=0.1,
        )
        assert_allclose(
            np.sum(eps._obs.m_perp, axis=0) * bin_volume,
            np.multiply(M_perp, n_dipoles / selection),
            rtol=0.1,
        )

    def test_epsilon(self, ag_two_frames):
        """Test that epsilon is constructed correctly from covariances."""
        eps = DielectricPlanar(ag_two_frames).run()

        cov_perp = eps.means.mM_perp - eps.means.m_perp * eps.means.M_perp
        assert_equal(eps.results.eps_perp, -eps._pref * cov_perp)

        cov_par = 0.5 * (eps.means.mM_par - np.dot(eps.means.m_par, eps.means.M_par))

        assert_equal(eps.results.eps_par, eps._pref * cov_par)

    def test_unsorted_ags(self, ag_two_frames):
        """Tests for inputs that don't have ordered atoms (i.e. LAMMPS)."""
        # Randomly shuffle the atomgroup
        rng = np.random.default_rng()
        permute = rng.permutation(len(ag_two_frames))
        ag2 = ag_two_frames[permute]

        eps1 = DielectricPlanar(ag_two_frames)
        eps1.run()

        eps2 = DielectricPlanar(ag2)
        eps2.run()

        assert np.allclose(eps1.results.eps_par, eps2.results.eps_par)
        assert np.allclose(eps1.results.eps_perp, eps2.results.eps_perp)

    def test_output(self, ag_two_frames, monkeypatch, tmp_path):
        """Test output."""
        monkeypatch.chdir(tmp_path)

        eps = DielectricPlanar(ag_two_frames)
        eps.run()
        eps.save()
        res_perp = np.loadtxt(f"{eps.output_prefix}_perp.dat")
        assert_allclose(eps.results.eps_perp, res_perp[:, 1])
        res_par = np.loadtxt(f"{eps.output_prefix}_par.dat")
        assert_allclose(eps.results.eps_par, res_par[:, 1])

    def test_output_name(self, ag_two_frames, monkeypatch, tmp_path):
        """Test output name."""
        monkeypatch.chdir(tmp_path)

        eps = DielectricPlanar(ag_two_frames, output_prefix="foo")
        eps.run()
        eps.save()
        with Path("foo_perp.dat").open():
            pass
        with Path("foo_par.dat").open():
            pass

    def test_sym(self, ag_two_frames):
        """Test for symmetric case."""
        eps_sym = DielectricPlanar(ag_two_frames, sym=True).run()
        eps = DielectricPlanar(ag_two_frames, sym=False).run()

        # Check that the z column is not changed
        assert_equal(eps.results.bin_pos, eps_sym.results.bin_pos)

        # Check that the all eps components are symmetric
        for d in [
            "eps_perp",
            "deps_perp",
            "eps_perp_self",
            "eps_perp_coll",
            "eps_par",
            "deps_par",
            "eps_par_self",
            "eps_par_coll",
        ]:
            A = (eps.results[d] + eps.results[d][::-1]) / 2

            assert_equal(A, eps_sym.results[d])

    @pytest.mark.parametrize(
        ("zmin", "zmax", "result"),
        [(1, None, True), (None, 1, True), (1, 1, True), (None, None, False)],
    )
    def test_range_warning(self, caplog, zmin, zmax, result):
        """Test for range warning."""
        warning = "Setting `zmin` and `zmax` might cut off molecules."
        ag = mda.Universe(DIPOLE_ITP, DIPOLE_GRO, topology_format="itp").atoms
        DielectricPlanar(ag, zmin=zmin, zmax=zmax)
        assert result == (warning in "".join([rec.message for rec in caplog.records]))

    def test_error_calculation(self, ag_two_frames):
        """Test the analytic error propagation."""
        eps = DielectricPlanar(ag_two_frames).run()

        # set values and errors for testing propagation:
        # perpendicular
        eps.means.mM_perp = 0.4
        eps.sems.mM_perp = 0.3

        eps.means.m_perp = 0.7
        eps.sems.m_perp = 1

        eps.means.M_perp = 0.001
        eps.sems.M_perp = 5e-5

        # rerun conclude function with changed values
        eps._conclude()
        deps_perp = eps.results.deps_perp

        m_mM, m_M, m_m = sp.symbols("m_mM m_M m_m")
        eps_perp = 1 - (m_mM - m_m * m_M) * eps._pref

        deps_perp_sympy = error_prop(
            eps_perp,
            [m_mM, m_M, m_m],
            [eps.sems.mM_perp, eps.sems.M_perp, eps.sems.m_perp],
        )(eps.means.mM_perp, eps.means.M_perp, eps.means.m_perp)

        assert_allclose(deps_perp, deps_perp_sympy)

        # same for parallel
        eps.n_bins = 1
        eps.means.mM_par = np.array([0.4])
        eps.sems.mM_par = np.array([0.3])

        eps.means.m_par = np.array([[0.7, 0.1]])
        eps.sems.m_par = np.array([[10.0, 1e-3]])

        eps.means.mm_par = np.array([0.4])
        eps.sems.mm_par = np.array([0.3])

        eps.means.M_par = np.array([[5, 5]])
        eps.sems.M_par = np.array([[0.7, 0.1]])

        eps.means.cmM_par = np.array([0.4])
        eps.means.cM_par = np.array([5, 5])

        # rerun conclude function with changed values
        eps._conclude()
        deps_par = eps.results.deps_par

        m_mM, m_M1, m_m1, m_M2, m_m2 = sp.symbols("m_mM m_M1 m_m1 m_M2 m_m2")
        eps_par = 0.5 * (m_mM - (m_m1 * m_M1 + m_m2 * m_M2)) * eps._pref

        deps_par_sympy = error_prop(
            eps_par,
            [m_mM, m_M1, m_m1, m_M2, m_m2],
            [
                eps.sems.mM_par[0],
                eps.sems.M_par[0, 0],
                eps.sems.m_par[0, 0],
                eps.sems.M_par[0, 1],
                eps.sems.m_par[0, 1],
            ],
        )(
            eps.means.mM_par[0],
            eps.means.M_par[0, 0],
            eps.means.m_par[0, 0],
            eps.means.M_par[0, 1],
            eps.means.m_par[0, 1],
        )

        assert_allclose(deps_par, deps_par_sympy)
