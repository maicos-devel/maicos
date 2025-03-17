#!/usr/bin/env python
#
# Copyright (c) 2025 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the GNU Public Licence, v3 or any higher version
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for the PDFCylinder class."""

import sys
from pathlib import Path

import MDAnalysis as mda
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_equal

from maicos import PDFCylinder

sys.path.append(str(Path(__file__).parents[1]))
from data import SPCE_GRO, SPCE_ITP  # noqa: E402
from util import circle_of_water_molecules, line_of_water_molecules  # noqa: E402


class TestPDFCylinder:
    """Tests for the PDFCylinder class."""

    @pytest.fixture
    def spce_water(self):
        """Return a universe with 1 water molecule."""
        return mda.Universe(SPCE_ITP, SPCE_GRO)

    @pytest.mark.parametrize("dim", [0, 1, 2])
    def test_dmax(self, spce_water, dim):
        """Test that dmax is larger than half of the cell length."""
        L = spce_water.dimensions[dim] / 2
        # Raise error if dmax is larger than half of the box size
        with pytest.raises(
            ValueError, match="Axial range of PDF exceeds half of the box size."
        ):
            PDFCylinder(spce_water.atoms, dmax=L + 1e3, dim=dim)._prepare()
        # No error if dmax is smaller than or equal half of the box size
        PDFCylinder(spce_water.atoms, dmax=L, dim=dim)._prepare()

    @pytest.mark.parametrize("dim", [0, 1, 2])
    def test_dmax_default(self, spce_water, dim):
        """Test that dmax defaults to half the box size in the given dimension."""
        L = spce_water.dimensions[dim] / 2
        ana_obj = PDFCylinder(spce_water.atoms, dim=dim)
        ana_obj._prepare()
        assert ana_obj.dmax == L

    def test_origin_shape_error(self, spce_water):
        """Test error raise when origin paramater has wrong shape."""
        match = r"Origin has length \(1,\) but only \(3,\) is allowed."
        with pytest.raises(ValueError, match=match):
            PDFCylinder(spce_water.atoms, origin=np.array([1]))

    # TODO(@kirafischer): Do the same for phi
    @pytest.mark.parametrize("dim", [0, 1, 2])
    @pytest.mark.parametrize("dmin", [0, 10])
    def test_pdf_bin_width_calculation(self, spce_water, dim, dmin):
        """Test that pdf_nbins and bin_width_pdf is calculated correctly."""
        bin_width_pdf_z = 0.1
        dmax = 15
        # Make the dimensions of the univese a bit easier to handle
        spce_water.dimensions = np.array([30, 40, 50, 90, 90, 90])
        ana_obj = PDFCylinder(
            spce_water.atoms,
            dim=dim,
            dmin=dmin,
            dmax=dmax,
            bin_width_pdf_z=bin_width_pdf_z,
        )
        ana_obj._prepare()
        # Test that the number of bins is calculated correctly
        assert ana_obj.nbins_pdf_z == int(np.ceil((dmax - dmin) / bin_width_pdf_z))
        # Test that the resulting bin width is calculated correctly
        assert ana_obj.bin_width_pdf_z == (dmax - dmin) / ana_obj.nbins_pdf_z

    def test_pdf_bin_width_smaller_zero(self, spce_water):
        """Test that bin width is larger than zero."""
        with pytest.raises(
            ValueError, match="PDF bin_width must be a positive number."
        ):
            PDFCylinder(spce_water.atoms, bin_width_pdf_z=-1)._prepare()
        with pytest.raises(
            ValueError, match="PDF bin_width must be a positive number."
        ):
            PDFCylinder(spce_water.atoms, bin_width_pdf_phi=-1)._prepare()

    @pytest.mark.parametrize(
        ("dim", "results"),
        [
            (0, [0, 1, 2, 3, 4, 5]),
            (1, [0, 2, 4, 6, 8, 10]),
            (2, [0, 3, 6, 9, 12, 15]),
        ],
    )
    def test_z_edges(self, spce_water, dim, results):
        """Test whether the PDF edges and the deduced bin positions are correct."""
        pdf_nbins = 5
        results = np.array(results)
        # Make the dimensions of the univese a bit easier to handle
        spce_water.dimensions = np.array([10, 20, 30, 90, 90, 90])
        ana_obj = PDFCylinder(
            spce_water.atoms,
            dim=dim,
            bin_width_pdf_z=spce_water.dimensions[dim] / 2 / pdf_nbins,
        ).run()

        assert_allclose(ana_obj.results.bins_z, 0.5 * (results[:-1] + results[1:]))

    def test_phi_edges(self, spce_water):
        """Test whether the PDF edges and the deduced bin positions are correct."""
        # Make the dimensions of the univese a bit easier to handle
        spce_water.dimensions = np.array([10, 20, 30, 90, 90, 90])
        ana_obj = PDFCylinder(
            spce_water.atoms,
            dim=2,
            bin_width_pdf_phi=np.pi / 6,  # this should result in 6 bins
            rmin=0,
            rmax=2,
            bin_width=2,
        ).run()
        assert_equal(len(ana_obj.results.bins_phi), 6)
        # binwidth is pi/6, so the bin center is multiple of that plus half of it
        results = np.array([0, 1, 2, 3, 4, 5]) * np.pi / 6 + np.pi / 12
        assert_allclose(ana_obj.results.bins_phi, results)

    def test_count_z(self):
        """Test whether the PDF molecule counts in a line are correct."""
        g2 = line_of_water_molecules(n_molecules=5, distance=1.0)
        g2.translate([0, 0, 0.5])
        g1 = line_of_water_molecules(n_molecules=1, distance=1.0)
        # Merge the two groups into one universe
        g = mda.Merge(g1, g2)
        g.dimensions = (20, 20, 20, 90, 90, 90)
        # Select the two groups
        g1 = g.atoms[:3]
        g2 = g.atoms[3:]

        ana_obj = PDFCylinder(
            g1,
            g2,
            dim=2,
            bin_width_pdf_z=1,
            origin=np.array([0, 0, 0]),
            rmin=0,
            rmax=1,
            bin_width=1,
        ).run()

        assert_allclose(ana_obj._obs.count_z[0], [1, 1, 1, 1, 1, 0, 0, 0, 0, 0])

    def test_count_phi(self):
        """Test whether the PDF molecule counts in ring are correct.

        Here, g2 is a circle of 6 water molecules with radius 1 and center (10, 10, 10).
        The box is (20, 20, 20), so the circle sits in the center of the box.
        g1 is a single water molecule sitting on the same circle, but in between two
        other molecules. With a bin width of pi/6, there should be 3 phi bins with
        2 molecules each.

                     #  *  #            *: g1
                   #         #          #: g2
                     #     #
        """
        g1, _ = circle_of_water_molecules(
            radius=1,
            n_molecules=1,
            angle_deg=0,
        )
        g1.rotateby(30, [0, 0, 1], [10, 10, 10])
        g2, _ = circle_of_water_molecules(
            radius=1,
            n_molecules=6,
            angle_deg=0,
        )

        # Merge the two groups into one universe
        g = mda.Merge(g1, g2)
        g.dimensions = (20, 20, 20, 90, 90, 90)

        # Select the two groups
        g1 = g.atoms[:3]
        g2 = g.atoms[3:]

        ana_obj = PDFCylinder(
            g1,
            g2,
            dim=2,
            bin_width_pdf_phi=np.pi / 3,
            rmin=0,
            rmax=2,
            bin_width=2,
        ).run()

        assert_allclose(ana_obj._obs.count_phi[0], [2, 2, 2])

    @pytest.mark.parametrize(
        ("name", "output"),
        [
            ("foo", ["z_foo.dat", "phi_foo.dat"]),
            ("bar.dat", ["z_bar.dat", "phi_bar.dat"]),
        ],
    )
    def test_output_name(self, spce_water, name, output, monkeypatch, tmp_path):
        """Test output name."""
        monkeypatch.chdir(tmp_path)

        ana_obj = PDFCylinder(spce_water.atoms, output=name)
        ana_obj.run()
        ana_obj.save()
        for file in output:
            assert Path(file).exists()

    def test_wrong_bin_method(self, spce_water):
        """Test grouping for a non existing bin_method."""
        ana_obj = PDFCylinder(spce_water.atoms, bin_method="foo")
        with pytest.raises(ValueError, match="is an unknown binning"):
            ana_obj._prepare()
