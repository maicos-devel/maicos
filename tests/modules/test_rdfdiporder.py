#!/usr/bin/env python
#
# Copyright (c) 2025 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the GNU Public Licence, v3 or any higher version
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for the RDFDiporder class."""

from pathlib import Path

import MDAnalysis as mda
import numpy as np
import pytest
from MDAnalysis.analysis.rdf import InterRDF
from MDAnalysis.core._get_readers import get_reader_for
from numpy.testing import assert_allclose

from maicos import RDFDiporder


class TestRDFDiporder:
    """Test the RDFDiporder class."""

    @pytest.fixture
    def u(self):
        """Generate a universe containing 100 dimers at random positions.

        All dimers have a distance of 0.1 Å and point in the z-direction.
        Dimer positions are drawn from a uniform distribution from 0 Å to 1 Å in
        each direction. The mass of the first atom is 1 and the of the second is 0.
        The charges of the first atom in each dimer is 1 and the second -1.
        The cell dimensions are 2Å x 2Å  x 2Å .
        """
        n_atoms = 200
        n_frames = 2
        n_residues = n_atoms // 2

        universe = mda.Universe.empty(
            n_atoms=n_atoms,
            n_residues=n_residues,
            n_segments=n_residues,
            atom_resindex=np.repeat(np.arange(n_residues), 2),
            residue_segindex=np.arange(n_residues),
        )

        universe.add_TopologyAttr("mass", values=n_residues * [1, 0])
        universe.add_TopologyAttr("charges", values=n_residues * [1, -1])
        universe.add_TopologyAttr("name", n_residues * ["A1", "A2"])
        universe.add_TopologyAttr("resids", np.arange(n_residues))
        universe.add_TopologyAttr("bonds", np.arange(n_atoms).reshape(n_residues, 2))

        rng = np.random.default_rng(1634123)
        coords = rng.random((n_frames, n_atoms, 3))

        # shift second atom per dimer by 0.1 A in z direction with respect to first one.
        coords[:, 1::2, :2] = coords[:, ::2, :2]
        coords[:, 1::2, 2] = coords[:, ::2, 2] + 0.1

        universe.trajectory = get_reader_for(coords)(
            coords, order="fac", n_atoms=n_atoms
        )

        for ts in universe.trajectory:
            ts.dimensions = np.array([2, 2, 2, 90, 90, 90])

        return universe

    @pytest.mark.parametrize("norm", ["rdf", "density", "none"])
    def test_rdf(self, u, norm):
        """Test RDFDiporder against the MDA InterRDF class for a special system."""
        # Select the first atom for the `InterRDF` analysis
        a1 = u.select_atoms("name A1")

        mda_rdf = InterRDF(g1=a1, g2=a1, nbins=10, range=(0.2, 1.2), norm=norm).run()
        maicos_rdf = RDFDiporder(
            g1=u.atoms,
            bin_width=0.1,
            rmin=0.2,
            rmax=1.2,
            unwrap=False,
            grouping="residues",
            norm=norm,
        ).run()

        assert_allclose(mda_rdf.results.bins, maicos_rdf.results.bins)
        assert_allclose(mda_rdf.results.rdf, maicos_rdf.results.rdf)

    def test_explicit_g2(self, u):
        """Test that leaving out the `g2` leads to the results."""
        rdf_explicit = RDFDiporder(g1=u.atoms, g2=u.atoms).run()
        rdf_default = RDFDiporder(g1=u.atoms).run()

        assert_allclose(rdf_explicit.results.bins, rdf_default.results.bins)
        assert_allclose(rdf_explicit.results.rdf, rdf_default.results.rdf)

    @pytest.mark.parametrize("norm", ["rdf", "density", "none"])
    def test_save(self, u, norm, monkeypatch, tmp_path):
        """Tests output name and that units are written in header."""
        monkeypatch.chdir(tmp_path)

        rdf = RDFDiporder(g1=u.atoms, norm=norm, output="foo")
        rdf.run(stop=1)
        rdf.save()

        with Path("foo.dat").open() as f:
            out = f.read()

        if norm in ["rdf", "density"]:
            assert "^3" in out
        else:
            assert "rdf" in out

    def test_unknown_norm(self, u):
        """Test unknown argument to `norm`."""
        with pytest.raises(ValueError, match="'foo' is an invalid `norm`."):
            RDFDiporder(u.atoms, norm="foo").run()
