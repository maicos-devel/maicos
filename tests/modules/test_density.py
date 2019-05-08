#!/usr/bin/env python3
# coding: utf8

import MDAnalysis as mda
import pytest

from MDAnalysisTests import tempdir
from mdtools import density_planar
import numpy as np
from numpy.testing import assert_equal, assert_almost_equal

from datafiles import WATER_GRO, WATER_TPR, WATER_TRR


class Test_density_planar(object):

    @pytest.fixture()
    def ag(self):
        u = mda.Universe(WATER_TPR, WATER_TRR)
        return u.atoms

    @pytest.fixture()
    def ag_single_frame(self):
        u = mda.Universe(WATER_TPR, WATER_GRO)
        return u.atoms

    @pytest.mark.parametrize('dens_type, mean',
                             (('mass', 987.9), ('number', 99.1),
                              ('charge', 0.0), ('temp', 291.6)))
    @pytest.mark.parametrize('dim', (0, 1, 2))
    def test_dens(self, ag, dens_type, mean, dim):
        dens = density_planar(ag, dens=dens_type, dim=dim).run()
        assert_almost_equal(dens.results['dens_mean'].mean(), mean, decimal=0)

    @pytest.mark.parametrize('dim', (0, 1, 2))
    def test_binwidth(self, ag_single_frame, dim):
        dens = density_planar(ag_single_frame, binwidth=0.1).run()
        # Divide by 10: Å -> nm
        n_bins = ag_single_frame.universe.dimensions[dim] / 10 // 0.1
        assert_almost_equal(
            dens.results["z"][1] - dens.results["z"][0], 0.1, decimal=2)
        assert_equal(len(dens.results["z"]), n_bins)

    def test_output(self, ag):
        with tempdir.in_tempdir():
            dens = density_planar(ag, save=True).run()
            res = np.loadtxt("{}.dat".format(dens.output))
            assert_almost_equal(
                dens.results["dens_mean"][:, 0], res[:, 1], decimal=2)

    def test_verbose(self, ag):
        density_planar(ag, verbose=True).run()
