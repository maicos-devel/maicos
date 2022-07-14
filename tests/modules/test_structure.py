#!/usr/bin/env python3
"""Tests for the structure modules."""
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
#
# Copyright (c) 2022 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the GNU Public Licence, v3 or any higher version
# SPDX-License-Identifier: GPL-3.0-or-later
import os

import MDAnalysis as mda
import numpy as np
import pytest
from create_mda_universe import isolated_water_universe
from datafiles import (
    AIRWATER_TPR,
    AIRWATER_TRR,
    SPCE_GRO,
    SPCE_ITP,
    WATER_TPR,
    WATER_TRR,
    )
from numpy.testing import assert_allclose, assert_equal

from maicos import Diporder, RDFPlanar, Saxs, utils


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

    def test_theta(self, ag):
        """Test min & max theta conditions."""
        saxs = Saxs(ag, mintheta=-10, maxtheta=190, verbose=True)
        saxs.run()
        saxs.save()
        assert_allclose(saxs.mintheta, 0)
        assert_equal(os.path.exists("sq.dat"), True)

    def test_nobindata(self, ag):
        """Test when nobindata is True."""
        saxs = Saxs(ag, nobin=True)
        saxs.run()
        assert_equal(type(saxs.q_factor).__name__ == 'ndarray', True)
        saxs.save()
        assert_equal(os.path.exists("sq.dat"), True)


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
        group_H2O_1 = isolated_water_universe(n_molecules=3, angle_deg=0)
        dip = Diporder(group_H2O_1, binwidth=10).run()

        assert_allclose(dip.results['P0'], 4.92e-4, rtol=1e-3)
        assert_equal(dip.results['rho'], 1e-3)
        assert_allclose(dip.results['cos_theta'], 1, rtol=1e-3)
        assert_allclose(dip.results['cos_2_theta'], 1, rtol=1e-3)

    def test_Diporder_3_water_90(self):
        """Test Diporder for 3 water molecules with angle 90 degrees."""
        group_H2O_2 = isolated_water_universe(n_molecules=3, angle_deg=90)
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


class TestRDFPlanar(object):
    """Tests for the RDFPlanar class."""

    def _molecule_positions(self):
        """Positions of 16 molecules in bcc configuration."""
        positions = [[0, 0, 0], [10, 0, 0], [0, 10, 0], [10, 10, 0],
                     [0, 0, 10], [10, 0, 10], [0, 10, 10], [10, 10, 10],
                     [5, 5, 5], [15, 5, 5], [5, 15, 5], [5, 5, 15],
                     [5, 15, 15], [15, 5, 15], [15, 15, 5], [15, 15, 15]]
        return positions

    @pytest.fixture()
    def get_universe(self):
        """Get a test universe with 16 water molecules bcc configuration."""
        fluid = []
        for _n in range(16):
            fluid.append(mda.Universe(SPCE_ITP, SPCE_GRO,
                                      topology_format='itp'))

        for molecule, position in zip(fluid, self._molecule_positions()):
            molecule.atoms.translate(position)
        u = mda.Merge(*[molecule.atoms for molecule in fluid])

        u.dimensions = [20, 20, 20, 90, 90, 90]
        u.residues.molnums = list(range(1, 17))
        u.atoms.write("universe.pdb")
        return u.select_atoms("name OW HW1 HW2")

    def run_rdf(self, get_universe, **kwargs):
        """Calculate the water water RDF of evenly spaced water."""
        grp_water = get_universe.select_atoms("resname SOL")
        rdfplanar = RDFPlanar(grp_water, grp_water, rdf_binwidth=1,
                              range=(7, 10), dzheight=6, binwidth=20, **kwargs)
        rdfplanar.run()
        return rdfplanar

    def run_rdf_OO(self, get_universe):
        """Calculate the OO RDF of evenly spaced water.

        Additionally, set 'g1' and 'g2' to be different atomgroups.
        """
        grpO = get_universe.select_atoms("name OW")
        rdfplanar = RDFPlanar(grpO[0:2], grpO, rdf_binwidth=1, range=(7, 10),
                              dzheight=6, binwidth=25)
        rdfplanar.run()
        return rdfplanar

    def test_edges(self, get_universe):
        """Test whether the RDF edges are correct."""
        rdfplanar = self.run_rdf(get_universe)
        assert_equal(rdfplanar.edges, [7, 8, 9, 10])

    def test_autorange(self, get_universe):
        """Test whether the maximum range of the RDF is set correctly."""
        grp_water = get_universe.select_atoms("resname SOL")
        rdfplanar = RDFPlanar(grp_water)
        rdfplanar.run()
        assert_equal(rdfplanar.range[1], 10)

    def test_count(self, get_universe):
        """Test whether the RDF molecule counts in ring are correct."""
        rdfplanar = self.run_rdf(get_universe)
        assert_equal(rdfplanar.count[0], [0, 128, 32])

    def test_n_g1_total(self, get_universe):
        """Test the number of g1 atoms."""
        rdfplanar = self.run_rdf(get_universe)
        assert_equal(rdfplanar.n_g1_total, 16)

    def test_rdf(self, get_universe):
        """Test the water water RDF."""
        rdfplanar = self.run_rdf(get_universe)
        bin1 = 0
        bin2 = 8 / (np.pi * (9**2 - 8**2)) / 6 / 2
        bin3 = 2 / (np.pi * (10**2 - 9**2)) / 6 / 2
        assert_allclose(rdfplanar.results.rdf[:, 0], [bin1, bin2, bin3])

    def test_different_axis(self, get_universe):
        """Test using x axis for binning."""
        rdfplanar = self.run_rdf(get_universe, dim=0)
        bin1 = 0
        bin2 = 8 / (np.pi * (9**2 - 8**2)) / 6 / 2
        bin3 = 2 / (np.pi * (10**2 - 9**2)) / 6 / 2
        assert(rdfplanar.dim == 0)
        assert_allclose(rdfplanar.results.rdf[:, 0], [bin1, bin2, bin3])

    def test_single_atom_com(self, get_universe):
        """Test whether the com of single atoms is correct."""
        rdfplanar = self.run_rdf_OO(get_universe)
        assert_equal(rdfplanar.g1.center_of_mass(compound=utils.
                                                 check_compound(rdfplanar.g1)),
                     self._molecule_positions()[0:2])

    def test_n_g1_total_OO(self, get_universe):
        """Test the number of g1 atoms for OO RDF."""
        rdfplanar = self.run_rdf_OO(get_universe)
        assert_equal(rdfplanar.n_g1_total, 2)

    def test_count_OO(self, get_universe):
        """Test whether the RDF atom count is correct."""
        rdfplanar = self.run_rdf_OO(get_universe)
        assert_equal(rdfplanar.count, [[0, 16, 4]])

    def test_rdf_OO(self, get_universe):
        """Test the OO RDF."""
        rdfplanar = self.run_rdf_OO(get_universe)
        bin1 = 0
        bin2 = 8 / (np.pi * (9**2 - 8**2)) / 6 / 2
        bin3 = 2 / (np.pi * (10**2 - 9**2)) / 6 / 2
        assert_allclose(rdfplanar.results.rdf[:, 0], [bin1, bin2, bin3])

    def run_rdf_with_binmethod(self, get_universe, binmethod):
        """Run rdf with binmethod and zmax.

        Because  0 < z < 10.1 com has 3 atom layers, but cog only 2
        rdf counts differ between com and cog.
        """
        grp_water = get_universe.select_atoms("resname SOL")
        z_dist_OH = 0.95 * np.sin(38 / 180 * np.pi)  # due to water geometry
        rdfplanar = RDFPlanar(grp_water, grp_water, rdf_binwidth=1,
                              range=(7, 10), dzheight=2,
                              zmax=10 + z_dist_OH / 4, binwidth=20,
                              binmethod=binmethod)
        rdfplanar.run()
        return rdfplanar

    @pytest.mark.parametrize("binmethod, count", [("com", 24),
                                                  ("coc", 16),
                                                  ("cog", 16)])
    def test_binmethod(self, get_universe, binmethod, count):
        """Test binmethods."""
        rdfplanar = self.run_rdf_with_binmethod(
            get_universe, binmethod=binmethod)
        assert_allclose(rdfplanar.count[0], [0, 0, count])

    def test_wrong_binmethod(self, get_universe):
        """Test grouping for a non existing binmethod."""
        with pytest.raises(ValueError, match="is an unknown binning"):
            self.run_rdf(get_universe, binmethod="foo")

    def test_large_range(self, get_universe):
        """Test that range is larger thanhalf of the cell length."""
        L = min(get_universe.universe.dimensions[:2]) / 2
        rdfplanar = self.run_rdf(get_universe)

        with pytest.raises(ValueError, match="exceeds half of the box"):
            rdfplanar.range = (0, L + 1e-3)
            rdfplanar.run()
