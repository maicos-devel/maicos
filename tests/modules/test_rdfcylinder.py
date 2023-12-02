#!/usr/bin/env python
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
#
# Copyright (c) 2023 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the GNU Public Licence, v3 or any higher version
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for the RDFPlanar class."""
import sys
from pathlib import Path

import MDAnalysis as mda
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_equal

from maicos import RDFCylinder


sys.path.append(str(Path(__file__).parents[1]))
from data import SPCE_GRO, SPCE_ITP  # noqa: E402
from modules.create_mda_universe import line_of_water_molecules  # noqa: E402


class TestRDFPlanar(object):
    """Tests for the RDFPlanar class."""

    @pytest.fixture()
    def spce_water(self):
        """Return a universe with 1 water molecule."""
        return mda.Universe(SPCE_ITP, SPCE_GRO)

    @pytest.mark.parametrize("dim", [0, 1, 2])
    def test_dmax(self, spce_water, dim):
        """Test that dmax is larger than half of the cell length."""
        L = spce_water.dimensions[dim] / 2
        # Raise error if dmax is larger than half of the box size
        with pytest.raises(
            ValueError, match="Range of RDF exceeds half of the box size."
        ):
            RDFCylinder(spce_water.atoms, dmax=L + 1e3, dim=dim)
        # No error if dmax is smaller than or equal half of the box size
        RDFCylinder(spce_water.atoms, dmax=L, dim=dim)

    @pytest.mark.parametrize("dim", [0, 1, 2])
    def test_dmax_default(self, spce_water, dim):
        """Test that dmax defaults to half the box size in the given dimension."""
        L = spce_water.dimensions[dim] / 2
        ana_obj = RDFCylinder(spce_water.atoms, dim=dim)
        assert ana_obj.dmax == L

    @pytest.mark.parametrize("dim", [0, 1, 2])
    @pytest.mark.parametrize("dmin", [0, 10])
    def test_rdf_bin_width_calculation(self, spce_water, dim, dmin):
        """Test that rdf_nbins and rdf_bin_width is calculated correctly."""
        rdf_bin_width = 0.1
        dmax = 15
        # Make the dimensions of the univese a bit easier to handle
        spce_water.dimensions = np.array([30, 40, 50, 90, 90, 90])
        ana_obj = RDFCylinder(
            spce_water.atoms, dim=dim, dmin=dmin, dmax=dmax, rdf_bin_width=rdf_bin_width
        )
        # Test that the number of bins is calculated correctly
        assert ana_obj.rdf_nbins == int(np.ceil((dmax - dmin) / rdf_bin_width))
        # Test that the resulting bin width is calculated correctly
        assert ana_obj.rdf_bin_width == (dmax - dmin) / ana_obj.rdf_nbins

    def test_rdf_bin_width_smaller_zero(self, spce_water):
        """Test that bin width is larger than zero."""
        with pytest.raises(
            ValueError, match="RDF bin_width must be a positive number."
        ):
            RDFCylinder(spce_water.atoms, rdf_bin_width=-1)

    @pytest.mark.parametrize(
        "dim, results",
        [
            (0, [0, 1, 2, 3, 4, 5]),
            (1, [0, 2, 4, 6, 8, 10]),
            (2, [0, 3, 6, 9, 12, 15]),
        ],
    )
    def test_edges(self, spce_water, dim, results):
        """Test whether the RDF edges and the dedused bin positions are correct."""
        rdf_nbins = 5
        results = np.array(results)
        # Make the dimensions of the univese a bit easier to handle
        spce_water.dimensions = np.array([10, 20, 30, 90, 90, 90])
        ana_obj = RDFCylinder(
            spce_water.atoms,
            dim=dim,
            rdf_bin_width=spce_water.dimensions[dim] / 2 / rdf_nbins,
        )
        assert_allclose(ana_obj.edges, results)
        assert_allclose(ana_obj.results.bins, 0.5 * (results[:-1] + results[1:]))

    def test_count(self):
        """Test whether the RDF molecule counts in ring are correct."""
        u = line_of_water_molecules(10, distance=2)
        print(u.atoms.positions)
        ana_obj = RDFCylinder(
            u.atoms,
            u.atoms,
            origin=(0, 0, 10),
            dim=2,
            rdf_bin_width=3,
        )
        ana_obj.run()
        assert_equal(ana_obj._obs.count[0], [20.0, 20.0, 20.0, 30.0])
        assert_allclose(ana_obj.norm[0] / np.pi, 1, rtol=1e-2)
        assert_equal(ana_obj.norm[1:].sum(), 0)
        assert_equal(
            ana_obj.results.rdf.T, np.nan_to_num(ana_obj._obs.count / ana_obj.norm / 2)
        )

    def test_output(self, spce_water, monkeypatch, tmp_path):
        """Test output."""
        monkeypatch.chdir(tmp_path)

        ana_obj = RDFCylinder(spce_water.atoms)
        ana_obj.run()
        ana_obj.save()
        res = np.loadtxt(f"{ana_obj.output}")
        assert_allclose(
            res,
            np.hstack([ana_obj.results.bins[:, np.newaxis], ana_obj.results.rdf]),
        )

    @pytest.mark.parametrize(
        "name, output", [("foo", "foo.dat"), ("bar.dat", "bar.dat")]
    )
    def test_output_name(self, spce_water, name, output, monkeypatch, tmp_path):
        """Test output name."""
        monkeypatch.chdir(tmp_path)

        ana_obj = RDFCylinder(spce_water.atoms, output=name)
        ana_obj.run()
        ana_obj.save()
        assert Path(output).exists()

    def test_wrong_bin_method(self, spce_water):
        """Test grouping for a non existing bin_method."""
        with pytest.raises(ValueError, match="is an unknown binning"):
            RDFCylinder(spce_water.atoms, bin_method="foo")
