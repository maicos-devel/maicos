#!/usr/bin/env python3
"""Tests for the epsilon modules."""
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
#
# Copyright (c) 2022 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the GNU Public Licence, v3 or any higher version
# SPDX-License-Identifier: GPL-3.0-or-later

import MDAnalysis as mda
import numpy as np
import pytest
from datafiles import DIPOLE_GRO, DIPOLE_ITP, WATER_GRO, WATER_TPR, WATER_TRR
from numpy.testing import assert_allclose, assert_equal

from maicos import DielectricSpectrum, EpsilonCylinder, EpsilonPlanar


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


class TestEpsilonPlanar(object):
    """Tests for the EpsilonPlanar class."""

    """

    Number of times EpsilonPlanar broke: ||||

    If you are reading this, most likely you are investigating a bug in the
    EpsilonPlanar class. To calculate the local electric permittivity in a
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
        eps = EpsilonPlanar(dipole, binwidth=0.001, vcutwidth=0.001)
        eps.run()
        # Check the total dipole moment of the system
        assert np.allclose(eps.results.frame.M_par, M_par, rtol=0.1)
        assert np.allclose(eps.results.frame.M_perp, M_perp, rtol=0.1)
        # Check the local dipole moment density by integrating it over the
        # volume and comparing with the total dipole moment of the system.
        assert np.allclose(
            np.sum(eps.results.frame.m_par[:, :, 0], axis=0)
            * eps.results.frame.V_bin,
            M_par, rtol=0.1)
        assert np.allclose(
            np.sum(eps.results.frame.m_perp[:, 0], axis=0)
            * eps.results.frame.V_bin,
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
        eps = EpsilonPlanar(dipole, binwidth=0.001, vcutwidth=0.001)
        eps.run()
        # Check the total dipole moment of the system
        assert np.allclose(eps.results.frame.M_par,
                           np.multiply(M_par, n_dipoles),
                           rtol=0.1)
        assert np.allclose(eps.results.frame.M_perp,
                           np.multiply(M_perp, n_dipoles),
                           rtol=0.1)
        # Check the local dipole moment density by integrating it over the
        # volume and comparing with the total dipole moment of the system.
        assert np.allclose(np.sum(eps.results.frame.m_par[:, :, 0], axis=0)
                           * eps.results.frame.V_bin,
                           np.multiply(M_par, n_dipoles),
                           rtol=0.1)
        assert np.allclose(np.sum(eps.results.frame.m_perp[:, 0], axis=0)
                           * eps.results.frame.V_bin,
                           np.multiply(M_perp, n_dipoles),
                           rtol=0.1)

    def test_epsilon(self, ag):
        """Test that epsilon is constructed correctly from covariances."""
        eps = EpsilonPlanar(ag, xy=True).run()

        cov_perp = eps.results.means.mM_perp \
            - eps.results.means.m_perp * eps.results.means.M_perp
        assert_equal(eps.results.eps_perp,
                     1 - eps.results.pref * cov_perp)

        cov_par = 0.5 * (eps.results.means.mM_par[:, 0]
                         - np.dot(eps.results.means.m_par[:, :, 0],
                                  eps.results.means.M_par))

        assert_equal(eps.results.eps_par[:, 0],
                     1 + eps.results.pref * cov_par)

    def test_output(self, ag_single_frame, tmpdir):
        """Test output."""
        with tmpdir.as_cwd():
            eps = EpsilonPlanar(ag_single_frame)
            eps.run()
            eps.save()
            res_perp = np.loadtxt("{}_perp.dat".format(eps.output_prefix))
            assert_allclose(eps.results.eps_perp[:, 0], res_perp[:, 1])
            res_par = np.loadtxt("{}_par.dat".format(eps.output_prefix))
            assert_allclose(eps.results.eps_par[:, 0], res_par[:, 1])

    def test_output_name(self, ag_single_frame, tmpdir):
        """Test output name."""
        with tmpdir.as_cwd():
            eps = EpsilonPlanar(ag_single_frame, output_prefix="foo")
            eps.run()
            eps.save()
            open("foo_perp.dat")
            open("foo_par.dat")

    def test_xy_vac(self, ag):
        """Tests for conditions xy & vac when True."""
        eps1 = EpsilonPlanar(ag, xy=True)
        eps1.run()
        k1 = np.mean(eps1.results.eps_perp - 1)
        eps2 = EpsilonPlanar(ag, xy=True, vac=True)
        eps2.run()
        k2 = np.mean(eps2.results.eps_perp - 1)
        assert_allclose((k1 / k2), 1.5, rtol=1e-1)

    def test_sym(self, ag_single_frame):
        """Test for symmetric case."""
        eps_sym = EpsilonPlanar(
            [ag_single_frame, ag_single_frame[:-30]], sym=True).run()
        eps = EpsilonPlanar(
            [ag_single_frame, ag_single_frame[:-30]], sym=False).run()

        # Check that the z column is not changed
        assert_equal(eps.results.z, eps_sym.results.z)

        # Check that the all eps components are symmetric
        for d in ["eps_perp", "deps_perp", "eps_perp_self", "eps_perp_coll",
                  "eps_par", "deps_par", "eps_par_self", "eps_par_coll"]:
            A = (eps.results[d] + eps.results[d][::-1]) / 2

            assert_equal(A, eps_sym.results[d])


class TestEpsilonCylinder(object):
    """Tests for the EpsilonCylinder class."""

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

    def test_radius(self, ag):
        """Tests radius set."""
        eps = EpsilonCylinder(ag, unwrap=False, radius=50)
        eps.run(start=0, stop=1)
        assert eps.radius == 50

    def test_one_frame(self, ag):
        """Test analysis running for one frame.

        Test if the division by the number of frames is correct.
        """
        eps = EpsilonCylinder(ag).run(stop=1)
        assert not np.isnan(eps.results.eps_rad).any()
        assert not np.isnan(eps.results.eps_ax).any()

    def test_radius_box(self, ag):
        """Tests radius taken from box."""
        eps = EpsilonCylinder(ag, unwrap=False)
        eps.run(start=0, stop=1)
        assert eps.radius == ag.universe.dimensions[:2].min() / 2

    def test_broken_molecules(self, ag):
        """Tests broken molecules."""
        eps = EpsilonCylinder(ag, unwrap=False).run()
        assert_allclose(eps.results['eps_ax'].mean(), 1179.0, rtol=1e-1)
        assert_allclose(eps.results['eps_rad'].mean(), -10, rtol=1e-1)

    def test_repaired_molecules(self, ag):
        """Tests repaired molecules."""
        eps = EpsilonCylinder(ag, unwrap=True).run()
        assert_allclose(eps.results['eps_ax'].mean(), 1179.6, rtol=1e-1)
        assert_allclose(eps.results['eps_rad'].mean(), -10, rtol=1e-1)

    def test_output(self, ag_single_frame, tmpdir):
        """Tests output."""
        with tmpdir.as_cwd():
            eps = EpsilonCylinder(ag_single_frame)
            eps.run()
            eps.save()
            res_ax = np.loadtxt("{}_ax.dat".format(eps.output_prefix))
            assert_allclose(eps.results["eps_ax"], res_ax[:, 1], rtol=1e-1)
            res_rad = np.loadtxt("{}_rad.dat".format(eps.output_prefix))
            assert_allclose(eps.results["eps_rad"], res_rad[:, 1], rtol=1e-2)

    def test_output_name(self, ag_single_frame, tmpdir):
        """Tests output name."""
        with tmpdir.as_cwd():
            eps = EpsilonCylinder(ag_single_frame, output_prefix="foo")
            eps.run()
            eps.save()
            open("foo_ax.dat")
            open("foo_rad.dat")

    def test_verbose(self, ag_single_frame):
        """Tests verbose."""
        EpsilonCylinder(ag_single_frame, verbose=True).run()

    def test_length(self, ag):
        """Test refactoring length."""
        eps = EpsilonCylinder(ag, length=100)
        eps.run()
        assert_equal(eps.length, 100)

    def test_variable_binwidth(self, ag):
        """Test variable binwidth."""
        eps = EpsilonCylinder(ag, variable_dr=True)
        eps.run()
        assert_allclose(np.std(eps.dr), 0.44, rtol=1e-1)

    def test_singleline(self, ag):
        """Test for single line 1D case."""
        eps = EpsilonCylinder(ag, single=True)
        eps.run()
        assert_allclose(np.mean(eps.results.eps_ax), 1282, rtol=1e-1)


class TestDielectricSpectrum(object):
    """Tests for the DielectricSpectrum class."""

    @pytest.fixture()
    def ag(self):
        """Import MDA universe."""
        u = mda.Universe(WATER_TPR, WATER_TRR)
        return u.atoms

    def test_output_name(self, ag, tmpdir):
        """Test output name."""
        with tmpdir.as_cwd():
            ds = DielectricSpectrum(ag)
            ds.run()
            ds.save()
            open("susc.dat")
            open("P_tseries.npy")
            open("tseries.npy")
            open("V.txt")

    def test_output_name_prefix(self, ag, tmpdir):
        """Test output name with custom prefix."""
        with tmpdir.as_cwd():
            ds = DielectricSpectrum(ag, output_prefix="foo")
            ds.run()
            ds.save()
            open("foo_susc.dat")
            open("foo_P_tseries.npy")
            open("foo_tseries.npy")
            open("foo_V.txt")

    def test_output_name_binned(self, ag, tmpdir):
        """Test output name of binned data."""
        """
        The parameters are not meant to be sensible,
        but just to force the binned output.
        """
        with tmpdir.as_cwd():
            ds = DielectricSpectrum(ag, bins=5, binafter=0, segs=5)
            ds.run()
            ds.save()
            open("susc.dat")
            open("susc_binned.dat")
            open("P_tseries.npy")
            open("tseries.npy")
            open("V.txt")

    def test_output(self, ag, tmpdir):
        """Test output values by comparing with magic numbers."""
        with tmpdir.as_cwd():
            ds = DielectricSpectrum(ag)
            ds.run()

            V = 1559814.4
            nu = [0., 0.2, 0.5, 0.7, 1.]
            susc = [27.5 + 0.j, 2.9 + 22.3j, -5.0 + 3.6j,
                    -0.5 + 10.7j, -16.8 + 3.5j]
            dsusc = [3.4 + 0.j, 0.4 + 2.9j, 1.0 + 0.5j, 0.3 + 1.5j, 2.0 + 0.6j]

            assert_allclose(ds.V, V, rtol=1e-1)
            assert_allclose(ds.results.nu, nu, rtol=1)
            assert_allclose(ds.results.susc, susc, rtol=1e-1)
            assert_allclose(ds.results.dsusc, dsusc, rtol=1e-1)

    def test_binning(self, ag, tmpdir):
        """Test binning & seglen case."""
        with tmpdir.as_cwd():
            ds = DielectricSpectrum(ag, nobin=False, segs=2, bins=49)
            ds.run()
            assert_allclose(np.mean(ds.results.nu_binned), 0.57, rtol=1e-2)
            ds.save()
