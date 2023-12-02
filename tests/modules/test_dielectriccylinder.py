#!/usr/bin/env python
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
#
# Copyright (c) 2023 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the GNU Public Licence, v3 or any higher version
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for the DielectricCylinder class."""
import sys
from pathlib import Path

import MDAnalysis as mda
import numpy as np
import pytest
from numpy.testing import assert_allclose

from maicos import DielectricCylinder


sys.path.append(str(Path(__file__).parents[1]))
from data import DIPOLE_GRO, DIPOLE_ITP, WATER_GRO, WATER_TPR, WATER_TRR  # noqa: E402


class TestDielectricCylinder(object):
    """Tests for the DielectricCylinder class.

    Number of times DielectricCylinder broke: ||

    If you are reading this, most likely you are investigating a bug in the
    DielectricCylinder class.

    Most problems with dielectric profiles are already discussed in the test Class for
    the DielectricPlanar and DielectricSphere modules.

    The DielectricCylinder may have the following ambiguities:
        - The code uses a quanity called M, which in planar geometry is the total dipole
          moment of the system, because it is int m(z) dz. We then calculate the systems
          total dipole moment instead of the integral for numerical stability. In
          cylindrical geometry, this is not the case for the radial direction, where it
          is int m(r) dr, like the spherical case. This might lead to confusion, but
          just make sure you are using the correct equations. (see
          10.1021/acs.jpcb.9b09269 for more info.)

          To test the charge density integration, we still integrate the diplome moment
          density and check if the integral equals the total dipole moment of the
          system. For the M integral, we just perform the same calculation as the module
          and check the result.

          For the axial direction, we also check that M (The one from the code,
          not the dipole density) is:
              M = (2  * pi * L) int r * m(r) dr

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

    @pytest.mark.parametrize("selection", (1, 2))
    def test_radial_dipole_orientations(self, selection):
        """Check radial dipole moment density.

        Create 4 dipoles radially pointing outwards and check if the volume integral
        over the dipole radial dipole moment density equals the total radial dipole
        moment of the system.
        """
        dipole1 = mda.Universe(DIPOLE_ITP, DIPOLE_GRO, topology_format="itp")
        dipole1.atoms.translate([1, 0, 0])

        dipole2 = mda.Universe(DIPOLE_ITP, DIPOLE_GRO, topology_format="itp")
        dipole2.atoms.rotateby(90, [0, 0, 1])
        dipole2.atoms.translate([0, 1, 0])

        dipole3 = mda.Universe(DIPOLE_ITP, DIPOLE_GRO, topology_format="itp")
        dipole3.atoms.rotateby(180, [0, 0, 1])
        dipole3.atoms.translate([-1, 0, 0])

        dipole4 = mda.Universe(DIPOLE_ITP, DIPOLE_GRO, topology_format="itp")
        dipole4.atoms.rotateby(270, [0, 0, 1])
        dipole4.atoms.translate([0, -1, 0])

        dipole = mda.Merge(
            *[dipole1.atoms, dipole2.atoms, dipole3.atoms, dipole4.atoms]
        )
        dipole.dimensions = [10, 10, 10, 90, 90, 90]
        dipole.atoms.translate(
            -dipole.atoms.center_of_mass() + dipole.dimensions[:3] / 2
        )

        if selection == 2:
            n = int(len(dipole.atoms) / selection)
        else:
            n = len(dipole.atoms)
        # very fine binning to get the correct value for the dipole
        eps = DielectricCylinder(dipole.atoms[:n], bin_width=0.001, vcutwidth=0.001)
        eps.run()
        # Check the dipole moment density by integrating over the system volume
        # and comparing to the diplole moment of the atomgroup.
        assert_allclose(
            np.sum(eps._obs.bin_volume * eps._obs.m_r), 4 / selection, rtol=0.1
        )
        # Then we check if the volume integral over the whole dipole density
        # is the total dipole moment.
        assert_allclose(np.sum(eps._obs.bin_volume * eps._obs.m_r_tot), 4, rtol=0.1)
        # Then we check if the value for M is calculated correctly.
        assert_allclose(
            np.sum(eps._obs.bin_width * eps._obs.m_r_tot), eps._obs.M_r, rtol=0.1
        )
        # Check M by using the same equation as in the module, just to be sure.
        # Check that the axial dipole moment is zero. (Should be due to
        # geometry)
        assert_allclose(np.sum(eps._obs.m_z), 0, rtol=0.1)

    @pytest.mark.parametrize("selection", (1, 2))
    def test_axial_dipole_orientations(self, selection):
        """Check radial dipole moment density.

        create 4 dipoles pointing in the axial direction and check if the volume
        integral over the axial dipole moment density equals the total axial dipole
        moment of the system.
        """
        dipole1 = mda.Universe(DIPOLE_ITP, DIPOLE_GRO, topology_format="itp")
        dipole1.atoms.rotateby(-90, [0, 1, 0])
        dipole1.atoms.translate([1, 0, 1])

        dipole2 = mda.Universe(DIPOLE_ITP, DIPOLE_GRO, topology_format="itp")
        dipole2.atoms.rotateby(-90, [0, 1, 0])
        dipole2.atoms.translate([0, 1, 2])

        dipole3 = mda.Universe(DIPOLE_ITP, DIPOLE_GRO, topology_format="itp")
        dipole3.atoms.rotateby(-90, [0, 1, 0])
        dipole3.atoms.translate([-1, 0, 3])

        dipole4 = mda.Universe(DIPOLE_ITP, DIPOLE_GRO, topology_format="itp")
        dipole4.atoms.rotateby(-90, [0, 1, 0])
        dipole4.atoms.translate([0, -1, 4])

        dipole = mda.Merge(
            *[dipole1.atoms, dipole2.atoms, dipole3.atoms, dipole4.atoms]
        )
        dipole.dimensions = [10, 10, 10, 90, 90, 90]
        dipole.atoms.translate(
            -dipole.atoms.center_of_mass() + dipole.dimensions[:3] / 2
        )

        if selection == 2:
            n = int(len(dipole.atoms) / selection)
        else:
            n = len(dipole.atoms)

        # very fine binning to get the correct value for the dipole
        eps = DielectricCylinder(dipole.atoms[:n], bin_width=1, vcutwidth=0.001)
        eps.run()
        # Check the dipole moment density by integrating over the system volume
        # and comparing to the total diplole moment of the system.
        assert_allclose(
            np.sum(eps._obs.bin_volume * eps._obs.m_z), 4 / selection, rtol=0.1
        )
        assert_allclose(eps._obs.M_z, 4, rtol=0.1)
        # Check that the radial dipole moment is zero. (Should be due to
        # geometry)
        assert_allclose(np.sum(eps._obs.m_r), 0, rtol=0.1)

    def test_output(self, ag_single_frame, monkeypatch, tmp_path):
        """Tests output."""
        monkeypatch.chdir(tmp_path)

        eps = DielectricCylinder(ag_single_frame)
        eps.run()
        eps.save()
        res_z = np.loadtxt("{}_z.dat".format(eps.output_prefix))
        assert_allclose(eps.results["eps_z"], res_z[:, 1], rtol=1e-1)
        res_r = np.loadtxt("{}_r.dat".format(eps.output_prefix))
        assert_allclose(eps.results["eps_r"], res_r[:, 1], rtol=1e-2)

    def test_output_name(self, ag_single_frame, monkeypatch, tmp_path):
        """Tests output name."""
        monkeypatch.chdir(tmp_path)

        eps = DielectricCylinder(ag_single_frame, output_prefix="foo")
        eps.run()
        eps.save()
        open("foo_z.dat")
        open("foo_r.dat")

    def test_singleline(self, ag):
        """Test for single line 1D case."""
        eps = DielectricCylinder(ag, single=True, bin_width=0.5)
        eps.run()
        assert_allclose(np.mean(eps.results.eps_z), 89.850, rtol=1e-1)

    def test_range_warning(self, caplog):
        """Test for range warning."""
        warning = "Setting `rmin` and `rmax` (as well as `zmin` and `zmax`)"
        ag = mda.Universe(DIPOLE_ITP, DIPOLE_GRO, topology_format="itp").atoms
        for i in range(16):
            # Only warn if user sets the ranges
            result = False if i == 0 else True
            # Check all possible combinations of set ranges
            bitmask = f"{i:04b}"
            zmin = float(bitmask[0]) if bitmask[0] != "0" else None
            zmax = float(bitmask[1]) if bitmask[1] != "0" else None
            rmin = float(bitmask[2]) if bitmask[2] != "0" else 0
            rmax = float(bitmask[3]) if bitmask[3] != "0" else None
            DielectricCylinder(ag, zmin=zmin, zmax=zmax, rmin=rmin, rmax=rmax)
            assert result == (
                warning in "".join([rec.message for rec in caplog.records])
            )
