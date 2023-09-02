#!/usr/bin/env python
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
#
# Copyright (c) 2023 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the GNU Public Licence, v3 or any higher version
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for the RDFPlanar class."""
import os
import sys

import MDAnalysis as mda
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_equal

from maicos import RDFPlanar
from maicos.lib.util import get_compound


sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from data import SPCE_GRO, SPCE_ITP  # noqa: E402


class TestRDFPlanar(object):
    """Tests for the RDFPlanar class."""

    def _molecule_positions(self):
        """Positions of 16 molecules in bcc configuration."""
        positions = [
            [0, 0, 0],
            [10, 0, 0],
            [0, 10, 0],
            [10, 10, 0],
            [0, 0, 10],
            [10, 0, 10],
            [0, 10, 10],
            [10, 10, 10],
            [5, 5, 5],
            [15, 5, 5],
            [5, 15, 5],
            [5, 5, 15],
            [5, 15, 15],
            [15, 5, 15],
            [15, 15, 5],
            [15, 15, 15],
        ]
        return positions

    @pytest.fixture()
    def get_universe(self):
        """Get a test universe with 16 water molecules bcc configuration."""
        fluid = []
        for _n in range(16):
            fluid.append(mda.Universe(SPCE_ITP, SPCE_GRO, topology_format="itp"))

        for molecule, position in zip(fluid, self._molecule_positions()):
            molecule.atoms.translate(position)
        u = mda.Merge(*[molecule.atoms for molecule in fluid])

        u.dimensions = [20, 20, 20, 90, 90, 90]
        u.residues.molnums = list(range(1, 17))
        return u.select_atoms("name OW HW1 HW2")

    def run_rdf(self, get_universe, **kwargs):
        """Calculate the water water RDF of evenly spaced water."""
        grp_water = get_universe.select_atoms("resname SOL")
        rdfplanar = RDFPlanar(
            grp_water,
            grp_water,
            rdf_bin_width=1,
            dmin=7,
            dmax=10,
            dzheight=6,
            bin_width=20,
            **kwargs,
        )
        rdfplanar.run()
        return rdfplanar

    def run_rdf_OO(self, get_universe):
        """Calculate the OO RDF of evenly spaced water.

        Additionally, set 'g1' and 'g2' to be different atomgroups.
        """
        grpO = get_universe.select_atoms("name OW")
        rdfplanar = RDFPlanar(
            grpO[0:2], grpO, rdf_bin_width=1, dmin=7, dmax=10, dzheight=6, bin_width=25
        )
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
        assert_equal(rdfplanar.dmax, 10)

    def test_count(self, get_universe):
        """Test whether the RDF molecule counts in ring are correct."""
        rdfplanar = self.run_rdf(get_universe)
        assert_equal(rdfplanar.means.count[0], [0, 128, 32])

    def test_n_g1_total(self, get_universe):
        """Test the number of g1 atoms."""
        rdfplanar = self.run_rdf(get_universe)
        assert_equal(rdfplanar.means.n_g1, 16)

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
        assert rdfplanar.dim == 0
        assert_allclose(rdfplanar.results.rdf[:, 0], [bin1, bin2, bin3])

    def test_single_atom_com(self, get_universe):
        """Test whether the com of single atoms is correct."""
        rdfplanar = self.run_rdf_OO(get_universe)
        assert_equal(
            rdfplanar.g1.center_of_mass(compound=get_compound(rdfplanar.g1)),
            self._molecule_positions()[0:2],
        )

    def test_n_g1_total_OO(self, get_universe):
        """Test the number of g1 atoms for OO RDF."""
        rdfplanar = self.run_rdf_OO(get_universe)
        assert_equal(rdfplanar.means.n_g1, 2)

    def test_count_OO(self, get_universe):
        """Test whether the RDF atom count is correct."""
        rdfplanar = self.run_rdf_OO(get_universe)
        assert_equal(rdfplanar.means.count, [[0, 16, 4]])

    def test_rdf_OO(self, get_universe):
        """Test the OO RDF."""
        rdfplanar = self.run_rdf_OO(get_universe)
        bin1 = 0
        bin2 = 8 / (np.pi * (9**2 - 8**2)) / 6 / 2
        bin3 = 2 / (np.pi * (10**2 - 9**2)) / 6 / 2
        assert_allclose(rdfplanar.results.rdf[:, 0], [bin1, bin2, bin3])

    def run_rdf_with_bin_method(self, get_universe, bin_method):
        """Run rdf with bin_method and zmax.

        Because  0 < z < 10.1 com has 3 atom layers, but cog only 2 rdf counts differ
        between com and cog.
        """
        grp_water = get_universe.select_atoms("resname SOL")
        z_dist_OH = 0.95 * np.sin(38 / 180 * np.pi)  # due to water geometry
        rdfplanar = RDFPlanar(
            grp_water,
            grp_water,
            rdf_bin_width=1,
            dmin=7,
            dmax=10,
            dzheight=2,
            zmax=10 + z_dist_OH / 4,
            bin_width=20,
            bin_method=bin_method,
        )
        rdfplanar.run()
        return rdfplanar

    @pytest.mark.parametrize(
        "bin_method, count", [("com", 24), ("coc", 16), ("cog", 16)]
    )
    def test_bin_method(self, get_universe, bin_method, count):
        """Test bin_methods."""
        rdfplanar = self.run_rdf_with_bin_method(get_universe, bin_method=bin_method)
        assert_allclose(rdfplanar.means.count[0], [0, 0, count])

    def test_wrong_bin_method(self, get_universe):
        """Test grouping for a non existing bin_method."""
        with pytest.raises(ValueError, match="is an unknown binning"):
            self.run_rdf(get_universe, bin_method="foo")

    def test_large_range(self, get_universe):
        """Test that range is larger thanhalf of the cell length."""
        L = min(get_universe.universe.dimensions[:2]) / 2
        rdfplanar = self.run_rdf(get_universe)

        with pytest.raises(ValueError, match="exceeds half of the box"):
            rdfplanar.dmax = L + 0.1
            rdfplanar.run()
