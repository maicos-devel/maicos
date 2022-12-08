#!/usr/bin/env python3
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
#
# Copyright (c) 2022 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the GNU Public Licence, v3 or any higher version
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for the DielectricCylinder class."""
import os
import sys

import MDAnalysis as mda
import numpy as np
import pytest
from numpy.testing import assert_allclose

from maicos import DielectricCylinder


sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from data import (  # noqa: E402
    DIPOLE_GRO,
    DIPOLE_ITP,
    WATER_GRO,
    WATER_TPR,
    WATER_TRR,
    )


class TestDielectricCylinder(object):
    """Tests for the DielectricCylinder class."""

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

    def test_radial_dipole_orientations(self):
        """Check radial dipole moment density.

        create 4 dipoles radially pointing outwards and check if the
        volume integral over the dipole radial dipole moment density
        equals the total radial dipole moment of the system.
        """
        dipole1 = mda.Universe(DIPOLE_ITP, DIPOLE_GRO, topology_format='itp')
        dipole1.atoms.translate([1, 0, 0])

        dipole2 = mda.Universe(DIPOLE_ITP, DIPOLE_GRO, topology_format='itp')
        dipole2.atoms.rotateby(90, [0, 0, 1])
        dipole2.atoms.translate([0, 1, 0])

        dipole3 = mda.Universe(DIPOLE_ITP, DIPOLE_GRO, topology_format='itp')
        dipole3.atoms.rotateby(180, [0, 0, 1])
        dipole3.atoms.translate([-1, 0, 0])

        dipole4 = mda.Universe(DIPOLE_ITP, DIPOLE_GRO, topology_format='itp')
        dipole4.atoms.rotateby(270, [0, 0, 1])
        dipole4.atoms.translate([0, -1, 0])

        dipole = mda.Merge(*[dipole1.atoms, dipole2.atoms,
                             dipole3.atoms, dipole4.atoms])
        dipole.dimensions = [10, 10, 10, 90, 90, 90]
        dipole.atoms.translate(- dipole.atoms.center_of_mass()
                               + dipole.dimensions[:3] / 2)

        # very fine binning to get the correct value for the dipole
        eps = DielectricCylinder(dipole.atoms, bin_width=0.001, vcutwidth=0.001)
        eps.run()
        # Check the total dipole moment of the system
        assert_allclose(np.sum(eps._obs.bin_volume * eps._obs.m_rad),
                        4, rtol=0.1)
        assert_allclose(eps._obs.M_rad,
                        np.sum(eps._obs.m_rad * eps._obs.bin_width), rtol=0.1)
        assert_allclose(np.sum(eps._obs.m_ax), 0, rtol=0.1)

    def test_axial_dipole_orientations(self):
        """Check radial dipole moment density.

        create 4 dipoles pointing in the axial direction and check if the
        volume integral over the axial dipole moment density
        equals the total axial dipole moment of the system.
        """
        dipole1 = mda.Universe(DIPOLE_ITP, DIPOLE_GRO, topology_format='itp')
        dipole1.atoms.rotateby(-90, [0, 1, 0])
        dipole1.atoms.translate([1, 0, 1])

        dipole2 = mda.Universe(DIPOLE_ITP, DIPOLE_GRO, topology_format='itp')
        dipole2.atoms.rotateby(-90, [0, 1, 0])
        dipole2.atoms.translate([0, 1, 2])

        dipole3 = mda.Universe(DIPOLE_ITP, DIPOLE_GRO, topology_format='itp')
        dipole3.atoms.rotateby(-90, [0, 1, 0])
        dipole3.atoms.translate([-1, 0, 3])

        dipole4 = mda.Universe(DIPOLE_ITP, DIPOLE_GRO, topology_format='itp')
        dipole4.atoms.rotateby(-90, [0, 1, 0])
        dipole4.atoms.translate([0, -1, 4])

        dipole = mda.Merge(*[dipole1.atoms, dipole2.atoms,
                             dipole3.atoms, dipole4.atoms])
        dipole.dimensions = [10, 10, 10, 90, 90, 90]
        dipole.atoms.translate(- dipole.atoms.center_of_mass()
                               + dipole.dimensions[:3] / 2)

        # very fine binning to get the correct value for the dipole
        eps = DielectricCylinder(dipole.atoms, bin_width=1, vcutwidth=0.001)
        eps.run()
        # Check the total dipole moment of the system
        assert_allclose(np.sum(eps._obs.bin_volume * eps._obs.m_ax), 4,
                        rtol=0.1)
        assert_allclose(eps._obs.M_ax, 4, rtol=0.1)
        assert_allclose(np.sum(eps._obs.m_rad), 0, rtol=0.1)

    def test_repaired_molecules(self, ag):
        """Tests repaired molecules."""
        eps = DielectricCylinder(ag, unwrap=True).run()
        assert_allclose(eps.results['eps_ax'].mean(), 19.8, rtol=1e-1)
        assert_allclose(eps.results['eps_rad'].mean(), -2.2, rtol=1e-1)

    def test_output(self, ag_single_frame, tmpdir):
        """Tests output."""
        with tmpdir.as_cwd():
            eps = DielectricCylinder(ag_single_frame)
            eps.run()
            eps.save()
            res_ax = np.loadtxt("{}_ax.dat".format(eps.output_prefix))
            assert_allclose(eps.results["eps_ax"], res_ax[:, 1], rtol=1e-1)
            res_rad = np.loadtxt("{}_rad.dat".format(eps.output_prefix))
            assert_allclose(eps.results["eps_rad"], res_rad[:, 1], rtol=1e-2)

    def test_output_name(self, ag_single_frame, tmpdir):
        """Tests output name."""
        with tmpdir.as_cwd():
            eps = DielectricCylinder(ag_single_frame, output_prefix="foo")
            eps.run()
            eps.save()
            open("foo_ax.dat")
            open("foo_rad.dat")

    def test_singleline(self, ag):
        """Test for single line 1D case."""
        eps = DielectricCylinder(ag, single=True, bin_width=0.5)
        eps.run()
        assert_allclose(np.mean(eps.results.eps_ax), 90., rtol=1e-1)
