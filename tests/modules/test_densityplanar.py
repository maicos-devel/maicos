#!/usr/bin/env python
#
# Copyright (c) 2025 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the GNU Public Licence, v3 or any higher version
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for the DensityPlanar class."""

import sys
from pathlib import Path

import MDAnalysis as mda
import numpy as np
import pytest
from MDAnalysisTests.datafiles import TPR, TRR
from numpy.testing import assert_allclose

from maicos import DensityPlanar

sys.path.append(str(Path(__file__).parents[1]))
from data import (  # noqa: E402
    MICA_TPR,
    MICA_XTC,
    SALT_WATER_GRO,
    SALT_WATER_TPR,
    WATER_GRO_NPT,
    WATER_TPR_NPT,
    WATER_TRR_NPT,
)


class ReferenceAtomGroups:
    """Super class with methods reference AtomGroups for tests."""

    @pytest.fixture
    def ag(self):
        """Import MDA universe."""
        u = mda.Universe(WATER_TPR_NPT, WATER_TRR_NPT)
        return u.atoms

    @pytest.fixture
    def ag_single_frame(self):
        """Import MDA universe, single frame."""
        u = mda.Universe(WATER_TPR_NPT, WATER_GRO_NPT)
        return u.atoms

    @pytest.fixture
    def multiple_ags(self):
        """Import MDA universe, multiple ags."""
        u = mda.Universe(TPR, TRR)
        return [u.select_atoms("resname SOL"), u.select_atoms("resname MET")]

    @pytest.fixture
    def multiple_ags_mu(self):
        """Import MDA universe, multiple ags mu."""
        u = mda.Universe(SALT_WATER_TPR, SALT_WATER_GRO)
        return [
            u.select_atoms("resname SOL"),
            u.select_atoms("resname NA"),
            u.select_atoms("resname CL"),
        ]

    @pytest.fixture
    def mica_water(self):
        """Import MDA universe, water components of a slab system."""
        u = mda.Universe(MICA_TPR, MICA_XTC)
        return u.select_atoms("resname SOL")

    @pytest.fixture
    def mica_surface(self):
        """Import MDA universe, surface component of a slab system."""
        u = mda.Universe(MICA_TPR, MICA_XTC)
        return u.select_atoms("resname SURF")

    @pytest.fixture
    def ag_no_masses(self):
        """Atom group with no mass."""
        u = mda.Universe(WATER_TPR_NPT, WATER_TRR_NPT)
        u.del_TopologyAttr("masses")
        return u.atoms

    @pytest.fixture
    def multiple_res_ag(self):
        """Import MDA universe, multiple resname atom group."""
        u = mda.Universe(SALT_WATER_TPR, SALT_WATER_GRO)
        return [u.select_atoms("resname NA or resname CL")]

    @pytest.fixture
    def mult_res_mult_atoms_ag(self):
        """Import MDA universe, multiple resname atom group."""
        u = mda.Universe(SALT_WATER_TPR, SALT_WATER_GRO)
        return [u.select_atoms("resname SOL or resname NA")]


class TestDensityPlanar(ReferenceAtomGroups):
    """Tests for the DensityPlanar class."""

    @pytest.mark.parametrize(
        ("dens_type", "mean"), [("mass", 0.588), ("number", 0.097), ("charge", 0)]
    )
    @pytest.mark.parametrize("dim", [0, 1, 2])
    def test_dens(self, ag, dens_type, mean, dim):
        """Test density."""
        dens = DensityPlanar(ag, dens=dens_type, dim=dim).run()
        assert_allclose(dens.results.profile.mean(), mean, rtol=1e-1, atol=1e-8)

    def test_one_frame(self, ag):
        """Test analysis running for one frame.

        Test if the division by the number of frames is correct.
        """
        dens = DensityPlanar(ag).run(stop=1)
        assert not np.isnan(dens.results.profile).any()

    def test_comshift(self, mica_water):
        """Test comshift."""
        dens = DensityPlanar(mica_water, refgroup=mica_water).run()
        assert_allclose(dens.results.profile[20], 0.581, rtol=1e-1)

    def test_comshift_z2(self, mica_water):
        """Test comshift with an additional shift by z/2."""
        mica_water.atoms.translate((0, 0, mica_water.universe.dimensions[2] / 2))
        dens = DensityPlanar(mica_water, refgroup=mica_water).run()
        assert_allclose(dens.results.profile[20], 0.56, rtol=1e-1)

    def test_comshift_over_boundaries(self, mica_water, mica_surface):
        """Test comshift over box boundaries."""
        dens = DensityPlanar(mica_water, refgroup=mica_surface).run()
        assert_allclose(dens.results.profile[20], 0.0, rtol=1e-1)
