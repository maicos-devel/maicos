#!/usr/bin/env python
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
#
# Copyright (c) 2024 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the GNU Public Licence, v3 or any higher version
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for the PDFPlanar class."""
import sys
from pathlib import Path

import MDAnalysis as mda
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_equal

from maicos import PDFPlanar
from maicos.lib.util import get_compound


sys.path.append(str(Path(__file__).parents[1]))
from data import SPCE_GRO, SPCE_ITP  # noqa: E402


class TestPDFPlanar(object):
    """Tests for the PDFPlanar class."""

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

    def run_pdf(self, get_universe, **kwargs):
        """Calculate the water water PDF of evenly spaced water."""
        grp_water = get_universe.select_atoms("resname SOL")
        pdfplanar = PDFPlanar(
            grp_water,
            grp_water,
            pdf_bin_width=1,
            dmin=7,
            dmax=10,
            dzheight=6,
            bin_width=20,
            **kwargs,
        )
        pdfplanar.run()
        return pdfplanar

    def run_pdf_OO(self, get_universe):
        """Calculate the OO PDF of evenly spaced water.

        Additionally, set 'g1' and 'g2' to be different atomgroups.
        """
        grpO = get_universe.select_atoms("name OW")
        pdfplanar = PDFPlanar(
            grpO[0:2], grpO, pdf_bin_width=1, dmin=7, dmax=10, dzheight=6, bin_width=25
        )
        pdfplanar.run()
        return pdfplanar

    def test_edges(self, get_universe):
        """Test whether the PDF edges are correct."""
        pdfplanar = self.run_pdf(get_universe)
        assert_equal(pdfplanar.edges, [7, 8, 9, 10])

    def test_autorange(self, get_universe):
        """Test whether the maximum range of the PDF is set correctly."""
        grp_water = get_universe.select_atoms("resname SOL")
        pdfplanar = PDFPlanar(grp_water)
        pdfplanar.run()
        assert_equal(pdfplanar.dmax, 10)

    def test_count(self, get_universe):
        """Test whether the PDF molecule counts in ring are correct."""
        pdfplanar = self.run_pdf(get_universe)
        assert_equal(pdfplanar.means.count[0], [0, 128, 32])

    def test_n_g1_total(self, get_universe):
        """Test the number of g1 atoms."""
        pdfplanar = self.run_pdf(get_universe)
        assert_equal(pdfplanar.means.n_g1, 16)

    def test_pdf(self, get_universe):
        """Test the water water PDF."""
        pdfplanar = self.run_pdf(get_universe)
        bin1 = 0
        bin2 = 8 / (np.pi * (9**2 - 8**2)) / 6 / 2
        bin3 = 2 / (np.pi * (10**2 - 9**2)) / 6 / 2
        assert_allclose(pdfplanar.results.pdf[:, 0], [bin1, bin2, bin3])

    def test_different_axis(self, get_universe):
        """Test using x axis for binning."""
        pdfplanar = self.run_pdf(get_universe, dim=0)
        bin1 = 0
        bin2 = 8 / (np.pi * (9**2 - 8**2)) / 6 / 2
        bin3 = 2 / (np.pi * (10**2 - 9**2)) / 6 / 2
        assert pdfplanar.dim == 0
        assert_allclose(pdfplanar.results.pdf[:, 0], [bin1, bin2, bin3])

    def test_single_atom_com(self, get_universe):
        """Test whether the com of single atoms is correct."""
        pdfplanar = self.run_pdf_OO(get_universe)
        assert_equal(
            pdfplanar.g1.center_of_mass(compound=get_compound(pdfplanar.g1)),
            self._molecule_positions()[0:2],
        )

    def test_n_g1_total_OO(self, get_universe):
        """Test the number of g1 atoms for OO PDF."""
        pdfplanar = self.run_pdf_OO(get_universe)
        assert_equal(pdfplanar.means.n_g1, 2)

    def test_count_OO(self, get_universe):
        """Test whether the PDF atom count is correct."""
        pdfplanar = self.run_pdf_OO(get_universe)
        assert_equal(pdfplanar.means.count, [[0, 16, 4]])

    def test_pdf_OO(self, get_universe):
        """Test the OO PDF."""
        pdfplanar = self.run_pdf_OO(get_universe)
        bin1 = 0
        bin2 = 8 / (np.pi * (9**2 - 8**2)) / 6 / 2
        bin3 = 2 / (np.pi * (10**2 - 9**2)) / 6 / 2
        assert_allclose(pdfplanar.results.pdf[:, 0], [bin1, bin2, bin3])

    def run_pdf_with_bin_method(self, get_universe, bin_method):
        """Run pdf with bin_method and zmax.

        Because  0 < z < 10.1 com has 3 atom layers, but cog only 2 pdf counts differ
        between com and cog.
        """
        grp_water = get_universe.select_atoms("resname SOL")
        z_dist_OH = 0.95 * np.sin(38 / 180 * np.pi)  # due to water geometry
        pdfplanar = PDFPlanar(
            grp_water,
            grp_water,
            pdf_bin_width=1,
            dmin=7,
            dmax=10,
            dzheight=2,
            zmax=10 + z_dist_OH / 4,
            bin_width=20,
            bin_method=bin_method,
        )
        pdfplanar.run()
        return pdfplanar

    @pytest.mark.parametrize(
        "bin_method, count", [("com", 24), ("coc", 16), ("cog", 16)]
    )
    def test_bin_method(self, get_universe, bin_method, count):
        """Test bin_methods."""
        pdfplanar = self.run_pdf_with_bin_method(get_universe, bin_method=bin_method)
        assert_allclose(pdfplanar.means.count[0], [0, 0, count])

    def test_wrong_bin_method(self, get_universe):
        """Test grouping for a non existing bin_method."""
        with pytest.raises(ValueError, match="is an unknown binning"):
            self.run_pdf(get_universe, bin_method="foo")

    def test_large_range(self, get_universe):
        """Test that range is larger thanhalf of the cell length."""
        L = min(get_universe.universe.dimensions[:2]) / 2
        pdfplanar = self.run_pdf(get_universe)

        with pytest.raises(ValueError, match="exceeds half of the box"):
            pdfplanar.dmax = L + 0.1
            pdfplanar.run()
