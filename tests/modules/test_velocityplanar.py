#!/usr/bin/env python
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
#
# Copyright (c) 2024 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the GNU Public Licence, v3 or any higher version
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for the VelocityPlanar class."""
import sys
from pathlib import Path

import MDAnalysis as mda
import numpy as np
import pytest
from numpy.testing import assert_allclose

from maicos import VelocityPlanar


sys.path.append(str(Path(__file__).parents[1]))
from data import WATER_TPR_NPT, WATER_TRR_NPT  # noqa: E402
from util import line_of_water_molecules  # noqa: E402


class ReferenceAtomGroups:
    """Super class with methods reference AtomGroups for tests."""

    @pytest.fixture()
    def vel_frame1_TRR(self):
        """
        Set velocity array for test_vel_trr.

        The values of the array correspond to the averaged components of the velocity
        along all 3 dimensions of space, respectively. Only the first frame is
        considered.
        """
        u = mda.Universe(WATER_TPR_NPT, WATER_TRR_NPT)
        v_array_1 = u.atoms.velocities.mean(axis=0)
        return v_array_1

    @pytest.fixture()
    def ag(self):
        """Import MDA universe."""
        u = mda.Universe(WATER_TPR_NPT, WATER_TRR_NPT)
        return u.atoms


class TestVelocityPlanar(ReferenceAtomGroups):
    """Tests for the VelocityPlanar class."""

    def test_wrong_vdim(self, ag):
        """Test a wrong dimension for velocity."""
        with pytest.raises(ValueError, match="Velocity dimension can"):
            VelocityPlanar(ag, dim=2, vdim=3)

    def test_wrong_dim(self, ag):
        """Test a wrong dimension."""
        with pytest.raises(ValueError, match="Dimension can only"):
            VelocityPlanar(ag, dim=3, vdim=2)

    @pytest.mark.parametrize("dim", (0, 1, 2))
    @pytest.mark.parametrize("vdim", (0, 1, 2))
    def test_vel_trr(self, ag, dim, vdim, vel_frame1_TRR):
        """Test VelocityPlanar module using WATER_TPR_NPT data.

        All 9 combinations of dim and vdim are tested.
        """
        vel = VelocityPlanar(ag, dim=dim, vdim=vdim, bin_width=ag.dimensions[dim]).run(
            stop=1
        )
        assert_allclose(vel.results.profile.mean(), vel_frame1_TRR[vdim], rtol=1e-2)

    @pytest.mark.parametrize("dim", (0, 1, 2))
    @pytest.mark.parametrize("vdim", (0, 1, 2))
    def test_vel_grouping_molecules(self, dim, vdim):
        """Test VelocityPlanar module using grouping by molecules.

        Create a universe with one single water molecule with a given velocity of 1
        along dim.

        Call VelocityPlanar module using one single bin and a grouping per atom.

        The expected result corresponds to the identity matrix in the dim-vdim space,
        divided by the volume of the box.

        All 9 combinations of dim and vdim are tested.
        """
        myvel = np.zeros(3)
        myvel[dim] += 1
        ag_v = line_of_water_molecules(n_molecules=1, myvel=myvel)
        vel = VelocityPlanar(
            ag_v, vdim=vdim, bin_width=ag_v.dimensions[dim], grouping="molecules"
        ).run()
        assert_allclose(
            vel.results.profile.mean(), np.identity(3)[dim][vdim], rtol=1e-6
        )

    @pytest.mark.parametrize("dim", (0, 1, 2))
    @pytest.mark.parametrize("vdim", (0, 1, 2))
    def test_vel_grouping_atoms(self, dim, vdim):
        """
        Test VelocityPlanar module using grouping by atoms.

        Create a universe with one single water molecule with a given velocity of 1
        along dim.

        Call VelocityPlanar module using one single bin and a grouping per atom.

        The expected result corresponds to the identity matrix in the dim-vdim space,
        divided by the volume of the box.

        All 9 combinations of dim and vdim are tested.
        """
        myvel = np.zeros(3)
        myvel[dim] += 1
        ag_v = line_of_water_molecules(n_molecules=1, myvel=myvel)
        vel = VelocityPlanar(
            ag_v, vdim=vdim, bin_width=ag_v.dimensions[dim], grouping="atoms"
        ).run()
        assert_allclose(vel.results.profile.mean(), np.identity(3)[dim][vdim])

    @pytest.mark.parametrize("dim", (0, 1, 2))
    @pytest.mark.parametrize("vdim", (0, 1, 2))
    def test_flux(self, dim, vdim):
        """
        Test flux measurement with VelocityPlanar module .

        Create a universe with one single water molecule with a given velocity of 1
        along dim.

        Call VelocityPlanar module to measure the flux, using one single bin, and a
        grouping per atom.

        The expected result corresponds to 3 times the identity matrix (one time per
        atom of the water molecule) in the dim-vdim space, divided by the volume of the
        box.

        All 9 combinations of dim and vdim are tested.
        """
        myvel = np.zeros(3)
        myvel[dim] += 1
        ag_v = line_of_water_molecules(n_molecules=1, myvel=myvel)
        vol = np.prod(ag_v.dimensions[:3])
        vel = VelocityPlanar(
            ag_v, vdim=vdim, bin_width=ag_v.dimensions[dim], grouping="atoms", flux=True
        ).run()
        assert_allclose(
            vel.results.profile,
            ag_v.n_atoms * np.identity(ag_v.n_atoms)[dim][vdim] / vol,
        )
