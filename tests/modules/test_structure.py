#!/usr/bin/env python3
"""Tests for the structure modules."""
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
from datafiles import (
    AIRWATER_TPR,
    AIRWATER_TRR,
    SPCE_GRO,
    SPCE_ITP,
    WATER_TPR,
    WATER_TRR,
    )
from numpy.testing import assert_allclose, assert_equal

from maicos import Diporder, Saxs


def create_universe(n_molecules, angle_deg):
    """Create universe with regularly-spaced water molecules."""
    fluid = []
    for _n in range(n_molecules):
        fluid.append(mda.Universe(SPCE_ITP, SPCE_GRO, topology_format='itp'))
    dimensions = fluid[0].dimensions

    rotations = [[angle_deg, (0, 1, 0)],
                 [angle_deg, (0, 1, 0)],
                 [angle_deg, (0, 1, 0)]]
    translations = [(0, 0, 5),
                    (0, 0, 15),
                    (0, 0, 25)]

    for molecule, rotation, translation in zip(fluid, rotations, translations):
        molecule.atoms.rotateby(rotation[0], rotation[1])
        molecule.atoms.translate(translation)
    u = mda.Merge(*[molecule.atoms for molecule in fluid])

    dimensions[2] *= n_molecules
    u.dimensions = dimensions
    u.residues.molnums = list(range(1, n_molecules + 1))
    return u.select_atoms("name OW HW1 HW2")


class TestSaxs(object):
    """Tests for the Saxs class."""

    @pytest.fixture()
    def ag(self):
        """Import MDA universe."""
        u = mda.Universe(WATER_TPR, WATER_TRR)
        return u.atoms

    def test_saxs(sef, ag):
        """Test Saxs."""
        Saxs(ag, endq=20).run(stop=5)

    def test_one_frame(self, ag):
        """Test analysis running for one frame.

        Test if the division by the number of frames is correct.
        """
        saxs = Saxs(ag, endq=20).run(stop=1)
        assert not np.isnan(saxs.results.scat_factor).any()


class TestDiporder(object):
    """Tests for the Diporder class."""

    @pytest.fixture()
    def result_dict(self):
        """Results dictionary."""
        res = {}

        # x-direction
        res[0] = {}
        res[0]["P0"] = 0
        res[0]["cos_theta"] = 0
        res[0]["cos_2_theta"] = 0.35
        res[0]["rho"] = 0.015

        # y-direction must be the same as x
        res[1] = res[0]

        # z-direction
        res[2] = {}
        res[2]["P0"] = 0
        res[2]["cos_theta"] = [0.02, 0, 0, 0, 0, 0, 0.16, 0, 0, 0, 0, -0.18]
        res[2]["cos_2_theta"] = [0.25, 0.33, 0.33, 0.33, 0.33, 0.26,
                                 0.2, 0, 0, 0, 0, 0.19]
        res[2]["rho"] = [0.021, 0.032, 0.033, 0.033, 0.033, 0.023,
                         0, 0, 0, 0, 0, 0]

        return res

    @pytest.fixture()
    def ag(self):
        """Import MDA universe."""
        u = mda.Universe(AIRWATER_TPR, AIRWATER_TRR)
        return u.atoms

    @pytest.mark.parametrize('dim', (0, 1, 2))
    def test_Diporder_slab(self, ag, dim, result_dict):
        """Test Diporder for slab system in z direction."""
        dip = Diporder(ag, binwidth=5, dim=dim).run()
        assert_allclose(dip.results['P0'], result_dict[dim]['P0'], atol=1e-2)
        assert_allclose(dip.results['rho'],
                        result_dict[dim]['rho'],
                        rtol=1e-2,
                        atol=1e-2)
        assert_allclose(dip.results['cos_theta'],
                        result_dict[dim]['cos_theta'],
                        atol=1e-1, rtol=1e-1)
        assert_allclose(dip.results['cos_2_theta'],
                        result_dict[dim]['cos_2_theta'],
                        atol=1e-2)

    def test_Diporder_3_water_0(self):
        """Test Diporder for 3 water molecules with angle 0."""
        group_H2O_1 = create_universe(3, 0)
        dip = Diporder(group_H2O_1, binwidth=10).run()

        assert_allclose(dip.results['P0'], 4.92e-4, rtol=1e-3)
        assert_equal(dip.results['rho'], 1e-3)
        assert_allclose(dip.results['cos_theta'], 1, rtol=1e-3)
        assert_allclose(dip.results['cos_2_theta'], 1, rtol=1e-3)

    def test_Diporder_3_water_90(self):
        """Test Diporder for 3 water molecules with angle 90 degrees."""
        group_H2O_2 = create_universe(3, 90)
        dip = Diporder(group_H2O_2, binwidth=10).run()
        assert_allclose(dip.results['P0'], 0, atol=1e-9)
        assert_equal(dip.results['rho'], 1e-3)
        assert_allclose(dip.results['cos_theta'], 0, atol=1e-5)
        assert_allclose(dip.results['cos_2_theta'], 0, atol=1e-6)

    def test_broken_molecules(self, ag):
        """Test broken molecules."""
        dip = Diporder(ag, make_whole=False).run()
        assert_allclose(dip.results['P0'].mean(), 0.0006, rtol=1e-1)

    def test_repaired_molecules(self, ag):
        """Test repaired molecules."""
        dip = Diporder(ag, make_whole=True).run()
        assert_allclose(dip.results['P0'].mean(), 0, atol=1e-2)

    def test_output(self, ag, tmpdir):
        """Test output."""
        with tmpdir.as_cwd():
            dip = Diporder(ag, end=20).run()
            dip.save()
            res_dip = np.loadtxt(dip.output)
            assert_allclose(dip.results["P0"], res_dip[:, 1], rtol=1e-2)

    def test_L_cum(self, ag):
        """Test cummulataive box length L_cum."""
        dip = Diporder(ag, bpbc=False).run()
        L_cum = ag.universe.trajectory.n_frames * ag.universe.dimensions[2]
        assert dip.L_cum == L_cum

    def test_one_frame(self, ag):
        """Test analysis running for one frame.

        Test if the division by the number of frames is correct.
        """
        dip = Diporder(ag, make_whole=False).run(stop=1)
        assert not np.isnan(dip.results.P0).any()

    def test_output_name(self, ag, tmpdir):
        """Test output name."""
        with tmpdir.as_cwd():
            dip = Diporder(ag, output="foo.dat", end=20)
            dip.run()
            dip.save()
            open("foo.dat")

    def test_verbose(self, ag):
        """Test verbose."""
        Diporder(ag, verbose=True).run()
