#!/usr/bin/env python3
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
#
# Copyright (c) 2022 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the GNU Public Licence, v3 or any higher version
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for the DielectricSphere class."""
import os
import sys

import MDAnalysis as mda
import numpy as np
import pytest
from numpy.testing import assert_allclose

from maicos import DielectricSphere


sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from data import (  # noqa: E402
    DIPOLE_GRO,
    DIPOLE_ITP,
    WATER_GRO,
    WATER_TPR,
    WATER_TRR,
    )


class TestDielectricSphere(object):
    """Tests for the DielectricSphere class."""

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

    @pytest.mark.parametrize('selection', (1, 2))
    def test_radial_dipole_orientations(self, selection):
        """Check radial dipole moment density.

        create 6 dipoles radially pointing outwards and check if the
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

        dipole5 = mda.Universe(DIPOLE_ITP, DIPOLE_GRO, topology_format='itp')
        dipole5.atoms.rotateby(-90, [0, 1, 0])
        dipole5.atoms.translate([0, 0, 1])

        dipole6 = mda.Universe(DIPOLE_ITP, DIPOLE_GRO, topology_format='itp')
        dipole6.atoms.rotateby(90, [0, 1, 0])
        dipole6.atoms.translate([0, 0, -1])

        dipole = mda.Merge(*[dipole1.atoms, dipole2.atoms,
                             dipole3.atoms, dipole4.atoms,
                             dipole5.atoms, dipole6.atoms])

        dipole.dimensions = [10, 10, 10, 90, 90, 90]
        dipole.atoms.translate(- dipole.atoms.center_of_mass()
                               + dipole.dimensions[:3] / 2)

        if selection == 2:
            n = int(len(dipole.atoms) / selection)
        else:
            n = len(dipole.atoms)

        # very fine binning to get the correct value for the dipole
        eps = DielectricSphere(dipole.atoms[:n], bin_width=0.001)
        eps.run()
        # Check the total dipole moment of the system
        assert_allclose(np.sum(eps._obs.bin_volume * eps._obs.m_rad),
                        6 / selection, rtol=0.1)
        assert_allclose(eps._obs.M_rad,
                        6, rtol=0.1)

    def test_output(self, ag_single_frame, tmpdir):
        """Tests output."""
        with tmpdir.as_cwd():
            eps = DielectricSphere(ag_single_frame)
            eps.run()
            eps.save()
            res_rad = np.loadtxt("{}_rad.dat".format(eps.output_prefix))
            assert_allclose(eps.results["eps_rad"], res_rad[:, 1], rtol=1e-2)

    def test_output_name(self, ag_single_frame, tmpdir):
        """Tests output name."""
        with tmpdir.as_cwd():
            eps = DielectricSphere(ag_single_frame, output_prefix="foo")
            eps.run()
            eps.save()
            open("foo_rad.dat")
