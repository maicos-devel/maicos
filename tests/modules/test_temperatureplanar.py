#!/usr/bin/env python
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
#
# Copyright (c) 2024 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the GNU Public Licence, v3 or any higher version
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for the TemperaturePlanar module."""
import sys
from pathlib import Path

import MDAnalysis as mda
import numpy as np
import pytest
from MDAnalysisTests.datafiles import TPR, TRR
from numpy.testing import assert_allclose

from maicos import TemperaturePlanar


sys.path.append(str(Path(__file__).parents[1]))
from data import WATER_TPR_NPT, WATER_TRR_NPT  # noqa: E402
from util import line_of_water_molecules  # noqa: E402


class ReferenceAtomGroups:
    """Super class with methods reference AtomGroups for tests."""

    @pytest.fixture()
    def ag(self):
        """Import MDA universe."""
        u = mda.Universe(WATER_TPR_NPT, WATER_TRR_NPT)
        return u.atoms

    @pytest.fixture()
    def multiple_ags(self):
        """Import MDA universe, multiple ags."""
        u = mda.Universe(TPR, TRR)
        return [u.select_atoms("resname SOL"), u.select_atoms("resname MET")]


class TestTemperatureProfile(ReferenceAtomGroups):
    """Tests for the TemperaturePlanar class."""

    @pytest.mark.parametrize("dim", (0, 1, 2))
    def test_dens(self, ag, dim):
        """Test TemperaturePlanar temperature.

        Only one frame is used.
        """
        temp = TemperaturePlanar(ag, dim=dim).run(stop=1)
        assert_allclose(temp.results.profile.mean(), 295, rtol=1e1)

    @pytest.mark.parametrize("dim", (0, 1, 2))
    def test_vel_atoms(self, dim):
        """Test TemperaturePlanar from a universe of 1 molecule.

        Create a universe made of one single molecule. A velocity of 1 along dim is
        given to the molecule.
        """
        myvel = np.zeros(3)
        myvel[dim] += 1
        ag_v = line_of_water_molecules(n_molecules=1, myvel=myvel)
        temp = TemperaturePlanar(ag_v, bin_width=ag_v.dimensions[dim]).run()
        assert_allclose(temp.results.profile.mean(), 3.611, rtol=1e-1)
