#!/usr/bin/env python3
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
#
# Copyright (c) 2022 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the GNU Public Licence, v3 or any higher version
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for the DielectricPlanar class."""
import os
import sys

import MDAnalysis as mda
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_equal

from maicos import DielectricPlanar


sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from data import (  # noqa: E402
    DIPOLE_GRO,
    DIPOLE_ITP,
    WATER_GRO,
    WATER_TPR,
    WATER_TRR,
    )


def dipoles(positions, orientations):
    """Atomgroup consisting of defined dipoles."""
    """
    Create MDA universe with dipole molecules
    inside a 10 Å x 10 Å x 10 Å box cubic box.
    """

    dipoles = []
    for position, orientation in zip(positions, orientations):
        position = np.array(position)
        orientation = np.array(orientation)
        dipole = mda.Universe(DIPOLE_ITP, DIPOLE_GRO, topology_format='itp')
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


class TestDielectricPlanar(object):
    """Tests for the DielectricPlanar class."""

    """

    Number of times DielectricPlanar broke: ||||

    If you are reading this, most likely you are investigating a bug in the
    DielectricPlanar class. To calculate the local electric permittivity in a
    system, the module needs two quantities: the total dipole moment and the
    local dipole moment density.

    The total dipole moment is the sum of the charges times the positions of the
    atoms. It is checked first in every test case, so make sure this is correct.

    The local dipole moment density is determined by the virtual cutting method.
    In short, charges are tallied in bins according to their positions and the
    resulting histogram is integrated.
    (see Schlaich et al, Phys. Rev. Lett. 117, 048001 and its SI for more info.)

    The correctness of the local dipole moment density is checked by integrating
    it and comparing it to the total dipole moment (which should be equal). Some
    things that broke this method in the past where:
        - Incorrect treatment of the charges at the box edges. Molecules have to
          be unwrapped to keep the system charge neutral overall. Charges that
          cross over the box limits in the lateral direction are capped (i.e.
          shifted) to the box edge so they will land in the first/last bin when
          doing the charge histograms.
        - For the parallel component, the positions of charges in the normal
          direction are shifted to the center of charge of the molecule they
          belong to. If something goes wrong here and charges are shifted out
          of the box, they are no longer counted in the histogram.
    """

    @pytest.fixture()
    def ag(self):
        """Import MDA universe."""
        u = mda.Universe(WATER_TPR, WATER_TRR)
        return u.atoms

    @pytest.fixture()
    def ag_single_frame(self):
        """Import MDA universe, single frame."""
        u = mda.Universe(WATER_TPR, WATER_GRO)
        return u.atoms

    @pytest.mark.parametrize('orientation, M_par, M_perp', (
        ([0, 0, 1], [0, 0], 1),
        ([1, 0, 0], [1, 0], 0),
        ([0, 1, 0], [0, 1], 0),
        ([1, 1, 1], [1 / np.sqrt(3), 1 / np.sqrt(3)], 1 / np.sqrt(3)),)
        )
    def test_single_dipole_orientations(self, orientation, M_par, M_perp):
        """Test the dipole density with a single dipole."""
        """
        This test places a single dipole (two unit charges separated by a 1 Å
        bond) with dipole moment 1 eÅ in a box of size 10 Å x 10 Å x 10 Å.

        The dipole is oriented in a given direction and the computed total
        dipole moment of the system is checked against the expected value.

        To check if the virtual cutting method produces the right results,
        the local dipole moment density is integrated over the entire system
        and checked against total dipole moment.
        """

        dipole = dipoles([[5, 5, 5]], [orientation])
        # very fine binning to get the correct value for the dipole
        eps = DielectricPlanar(dipole, bin_width=0.001, vcutwidth=0.001)
        eps.run()
        # Check the total dipole moment of the system
        assert np.allclose(eps._obs.M_par, M_par, rtol=0.1)
        assert np.allclose(eps._obs.M_perp, M_perp, rtol=0.1)
        # Check the local dipole moment density by integrating it over the
        # volume and comparing with the total dipole moment of the system.
        bin_volume = eps.means.bin_volume[0]
        assert np.allclose(
            np.sum(eps._obs.m_par[:, :, 0], axis=0)
            * bin_volume,
            M_par, rtol=0.1)
        assert np.allclose(
            np.sum(eps._obs.m_perp[:, 0], axis=0)
            * bin_volume,
            M_perp, rtol=0.1)

    @pytest.mark.parametrize('orientation, M_par, M_perp', (
        ([0, 0, 1], [0, 0], 1),
        ([1, 0, 0], [1, 0], 0),
        ([0, 1, 0], [0, 1], 0),
        ([1, 1, 1], [1 / np.sqrt(3), 1 / np.sqrt(3)], 1 / np.sqrt(3)),)
        )
    def test_multiple_dipole_orientations(self, orientation, M_par, M_perp):
        """Test the dipole moment density with multiple dipoles."""
        """
        This test places a grid of 5x5x5 dipoles (two unit charges separated by
        a 1 Å bond) with dipole moment 1 eÅ in a box of size 10 Å x 10 Å x 10 Å.

        The dipoles are oriented in a given direction and the computed total
        dipole moment of the system is checked against the expected value.

        To check if the virtual cutting method produces the right results,
        the local dipole moment density is integrated over the entire system
        and checked against total dipole moment.

        This test is a variation of `test_single_dipole_orientations` to catch
        problems with system scaling of shifting that are not catched by the
        single dipole test. For example if some positions of charges are
        erroneously shifted out of the system.

        See https://gitlab.com/maicos-devel/maicos/-/issues/83 for more infos.
        """
        xx, yy, zz = np.meshgrid(np.arange(0, 10, 2) + 1,
                                 np.arange(0, 10, 2) + 1,
                                 np.arange(0, 10, 2) + 1)

        pos = np.array([xx.flatten(), yy.flatten(), zz.flatten()]).T.tolist()
        n_dipoles = len(pos)
        dipole = dipoles(pos, [orientation] * n_dipoles)

        # very fine binning to get the correct value for the dipole
        eps = DielectricPlanar(dipole, bin_width=0.001, vcutwidth=0.001)
        eps.run()
        # Check the total dipole moment of the system
        assert np.allclose(eps._obs.M_par,
                           np.multiply(M_par, n_dipoles),
                           rtol=0.1)
        assert np.allclose(eps._obs.M_perp,
                           np.multiply(M_perp, n_dipoles),
                           rtol=0.1)
        # Check the local dipole moment density by integrating it over the
        # volume and comparing with the total dipole moment of the system.
        bin_volume = eps.means.bin_volume[0]
        assert np.allclose(np.sum(eps._obs.m_par[:, :, 0], axis=0)
                           * bin_volume,
                           np.multiply(M_par, n_dipoles),
                           rtol=0.1)
        assert np.allclose(np.sum(eps._obs.m_perp[:, 0], axis=0)
                           * bin_volume,
                           np.multiply(M_perp, n_dipoles),
                           rtol=0.1)

    def test_epsilon(self, ag):
        """Test that epsilon is constructed correctly from covariances."""
        eps = DielectricPlanar(ag, xy=True).run()

        cov_perp = eps.means.mM_perp \
            - eps.means.m_perp * eps.means.M_perp
        assert_equal(eps.results.eps_perp,
                     - eps.results.pref * cov_perp)

        cov_par = 0.5 * (eps.means.mM_par[:, 0]
                         - np.dot(eps.means.m_par[:, :, 0],
                                  eps.means.M_par))

        assert_equal(eps.results.eps_par[:, 0],
                     eps.results.pref * cov_par)

    def test_unsorted_ags(self, ag):
        """Tests for inputs that don't have ordered atoms (i.e. LAMMPS)."""
        # Randomly shuffle the atomgroup
        rng = np.random.default_rng()
        permute = rng.permutation(len(ag))
        ag2 = ag[permute]

        eps1 = DielectricPlanar(ag, xy=True)
        eps1.run()
        k1 = eps1.results.eps_par

        eps2 = DielectricPlanar(ag2, xy=True, vac=True)
        eps2.run()
        k2 = eps2.results.eps_par
        assert np.allclose(k1, k2, rtol=1e-1)

    def test_output(self, ag_single_frame, tmpdir):
        """Test output."""
        with tmpdir.as_cwd():
            eps = DielectricPlanar(ag_single_frame)
            eps.run()
            eps.save()
            res_perp = np.loadtxt("{}_perp.dat".format(eps.output_prefix))
            assert_allclose(eps.results.eps_perp[:, 0], res_perp[:, 1])
            res_par = np.loadtxt("{}_par.dat".format(eps.output_prefix))
            assert_allclose(eps.results.eps_par[:, 0], res_par[:, 1])

    def test_output_name(self, ag_single_frame, tmpdir):
        """Test output name."""
        with tmpdir.as_cwd():
            eps = DielectricPlanar(ag_single_frame, output_prefix="foo")
            eps.run()
            eps.save()
            open("foo_perp.dat")
            open("foo_par.dat")

    def test_xy_vac(self, ag):
        """Tests for conditions xy & vac when True."""
        eps1 = DielectricPlanar(ag, xy=True)
        eps1.run()
        k1 = np.mean(eps1.results.eps_perp - 1)
        eps2 = DielectricPlanar(ag, xy=True, vac=True)
        eps2.run()
        k2 = np.mean(eps2.results.eps_perp - 1)
        assert_allclose((k1 / k2), 1.5, rtol=1e-1)

    def test_sym(self, ag_single_frame):
        """Test for symmetric case."""
        eps_sym = DielectricPlanar(
            [ag_single_frame, ag_single_frame[:-30]], sym=True).run()
        eps = DielectricPlanar(
            [ag_single_frame, ag_single_frame[:-30]], sym=False).run()

        # Check that the z column is not changed
        assert_equal(eps.results.bin_pos, eps_sym.results.bin_pos)

        # Check that the all eps components are symmetric
        for d in ["eps_perp", "deps_perp", "eps_perp_self", "eps_perp_coll",
                  "eps_par", "deps_par", "eps_par_self", "eps_par_coll"]:
            A = (eps.results[d] + eps.results[d][::-1]) / 2

            assert_equal(A, eps_sym.results[d])
