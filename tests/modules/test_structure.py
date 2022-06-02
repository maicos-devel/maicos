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
from numpy.testing import assert_almost_equal

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
        res[0]["P0"] = np.array([0, 0, 0.01, 0])
        res[0]["cos_theta"] = np.array([0, 0, 0.01, 0])
        res[0]["cos_2_theta"] = np.array([0.34, 0.35, 0.35, 0.35])
        res[0]["rho"] = np.array([14.86, 14.58, 14.64, 14.59])

        # y-direction must be the same as x
        res[1] = res[0]

        # z-direction
        res[2] = {}
        res[2]["P0"] = np.array([0.02, 0, -0, 0, -0, -0.02, 0, 0, 0,
                                 0, 0, 0])
        res[2]["cos_theta"] = np.array([0.02, 0, 0, 0, 0, -0.02,
                                        0.16, 0, 0, 0, 0, -0.18])
        res[2]["cos_2_theta"] = np.array([0.25, 0.33, 0.33, 0.33, 0.33,
                                          0.26, 0.2, 0, 0, 0, 0, 0.19])
        res[2]["rho"] = np.array([21.1, 32.97, 33.05, 32.81, 32.7, 22.71,
                                  0.4, 0, 0, 0, 0, 0.26])

        return res

    @pytest.fixture()
    def ag(self):
        """Import MDA universe."""
        u = mda.Universe(AIRWATER_TPR, AIRWATER_TRR)
        return u.atoms

    @pytest.mark.parametrize('dim', (0, 1, 2))
    def test_Diporder(self, ag, dim, result_dict):
        """Test Diporder."""
        dip1 = Diporder(ag, binwidth=0.5, dim=dim).run()
        assert_almost_equal(dip1.results['P0'],
                            result_dict[dim]['P0'],
                            decimal=2)
        assert_almost_equal(dip1.results['cos_theta'],
                            result_dict[dim]['cos_theta'],
                            decimal=1)
        assert_almost_equal(dip1.results['cos_2_theta'],
                            result_dict[dim]['cos_2_theta'],
                            decimal=2)
        assert_almost_equal(dip1.results['rho'],
                            result_dict[dim]['rho'],
                            decimal=0)

        # 3 water molecules with angle 0
        group_H2O_1 = create_universe(3, 0)
        dip2 = Diporder(group_H2O_1, binwidth=1).run()

        assert_almost_equal(np.unique(np.round(dip2.results['P0'], 3)),
                            0.049, decimal=3)
        assert_almost_equal(np.unique(dip2.results['rho']),
                            1.0, decimal=3)
        assert_almost_equal(np.unique(dip2.results['cos_theta']),
                            1.0, decimal=3)
        assert_almost_equal(np.unique(dip2.results['cos_2_theta']),
                            1.0, decimal=3)

        # 3 water molecules with angle 90
        group_H2O_2 = create_universe(3, 90)
        dip3 = Diporder(group_H2O_2, binwidth=1).run()
        assert_almost_equal(np.unique(np.round(dip3.results['P0'], 3)),
                            0.0, decimal=3)
        assert_almost_equal(np.unique(dip3.results['rho']),
                            1.0, decimal=3)
        assert_almost_equal(np.unique(dip3.results['cos_theta']),
                            0.0, decimal=3)
        assert_almost_equal(np.unique(dip3.results['cos_2_theta']),
                            0.0, decimal=3)

    def test_broken_molecules(self, ag):
        """Test broken molecules."""
        dip = Diporder(ag, make_whole=False).run()
        assert_almost_equal(dip.results['P0'].mean(), 0.06, decimal=2)

    def test_repaired_molecules(self, ag):
        """Test repaired molecules."""
        dip = Diporder(ag, make_whole=True).run()
        assert_almost_equal(dip.results['P0'].mean(), 0.00, decimal=2)

    def test_output(self, ag, tmpdir):
        """Test output."""
        with tmpdir.as_cwd():
            dip = Diporder(ag, end=20)
            dip.run()
            dip.save()
            res_dip = np.loadtxt(dip.output)
            assert_almost_equal(dip.results["P0"], res_dip[:, 1], decimal=2)

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
