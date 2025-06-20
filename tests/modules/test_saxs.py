#!/usr/bin/env python
#
# Copyright (c) 2025 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the GNU Public Licence, v3 or any higher version
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for the SAXS modules."""

import sys
from pathlib import Path

import MDAnalysis as mda
import numpy as np
import pytest
from data import WATER_GRO_NPT, WATER_TPR_NPT, WATER_TRR_NPT
from MDAnalysis.analysis.rdf import InterRDF
from numpy.testing import assert_allclose, assert_equal

from maicos import Saxs
from maicos.lib.math import compute_form_factor, compute_rdf_structure_factor

sys.path.append(str(Path(__file__).parents[1]))


class ReferenceAtomGroups:
    """Super class with methods reference AtomGroups for tests."""

    @pytest.fixture
    def ag_single_frame(self):
        """Import MDA universe."""
        u = mda.Universe(WATER_TPR_NPT, WATER_GRO_NPT)
        return u.atoms


class TestSaxs(ReferenceAtomGroups):
    """Tests for the Saxs class."""

    @pytest.fixture
    def ag_single_frame(self):
        """Import MDA universe."""
        u = mda.Universe(WATER_GRO_NPT)
        u.guess_TopologyAttrs(to_guess=["elements"])
        u.atoms.elements = np.array([el.title() for el in u.atoms.elements])

        return u.atoms

    @pytest.fixture
    def ag(self):
        """Import MDA universe."""
        u = mda.Universe(WATER_TPR_NPT, WATER_TRR_NPT)
        u.guess_TopologyAttrs(to_guess=["elements"])
        u.atoms.elements = np.array([el.title() for el in u.atoms.elements])

        return u.atoms

    def test_one_frame(sef, ag_single_frame, monkeypatch, tmp_path):
        """Test Saxs on one frame.

        Test if the division by the number of frames is correct.
        """
        monkeypatch.chdir(tmp_path)

        saxs = Saxs(ag_single_frame, qmax=20, output="foo").run()
        assert_allclose(saxs.results.scattering_intensities[0], 1.6047, rtol=1e-3)

        saxs.save()
        saxs_loaded = np.loadtxt("foo.dat")
        assert_allclose(saxs.results.scattering_vectors, saxs_loaded[:, 0])
        assert_allclose(saxs.results.structure_factors, saxs_loaded[:, 1])
        assert_allclose(saxs.results.scattering_intensities, saxs_loaded[:, 2])

    def test_theta(self, ag_single_frame):
        """Smoke test for min and max theta conditions on one frame."""
        with pytest.raises(ValueError, match=r"thetamin \(-10°\) has to between 0"):
            Saxs(ag_single_frame, thetamin=-10, thetamax=190).run()

    def test_unknown_element(self, ag_single_frame):
        """Test that an error is raised if an element is unknown."""
        d = {"O": "foo", "H": "H"}
        ag_single_frame.elements = np.array([d[t] for t in ag_single_frame.elements])

        match = (
            "Element 'foo' not found. Known elements are listed in the "
            "`maicos.lib.tables.elements` set."
        )
        with pytest.raises(ValueError, match=match):
            Saxs(ag_single_frame).run()

    def test_not_binned_spectrum(self, ag_single_frame, monkeypatch, tmp_path):
        """Test when ``bin_spectrum`` is False."""
        monkeypatch.chdir(tmp_path)
        saxs = Saxs(ag_single_frame, bin_spectrum=False, output="foo").run()
        assert type(saxs.scattering_vector_factors).__name__ == "ndarray"

        # test that values are sorted in an increasing order
        assert_equal(
            saxs.results.scattering_vectors, np.sort(saxs.results.scattering_vectors)
        )

        # test output
        saxs.save()
        saxs_loaded = np.loadtxt("foo.dat")
        assert_allclose(saxs.results.scattering_vectors, saxs_loaded[:, 0])
        assert_allclose(saxs.results.miller_indices, saxs_loaded[:, 1:4])
        assert_allclose(saxs.results.structure_factors, saxs_loaded[:, 4])
        assert_allclose(saxs.results.scattering_intensities, saxs_loaded[:, 5])

    def scattering_intensity(self, ag):
        """Test that for a single component system the scattering intensity is correct.

        Given by I(q) = [F(q)]^2 S(q)
        """
        oxy = ag.select_atoms("name OW")
        saxs = Saxs(atomgroup=oxy, dq=0.1).run()
        scattering_vectors = saxs.results.scattering_vectors

        scattering_intensities = (
            compute_form_factor(scattering_vectors, "O") ** 2 * scattering_vectors
        )
        assert_allclose(saxs.results.scattering_intensities, scattering_intensities)

    def test_rdf_comparison(self, ag):
        """Test if the Fourier transformation of an RDF is the structure factor."""
        oxy = ag.select_atoms("name OW")
        L = ag.universe.dimensions[0]  # we have a cubic box

        density = oxy.n_atoms / ag.universe.trajectory.ts.volume

        inter_rdf = InterRDF(
            oxy,
            oxy,
            nbins=300,
            range=(0, L / 2),
            exclude_same="residue",
        ).run()

        scattering_vectors_rdf, structure_factors_rdf = compute_rdf_structure_factor(
            rdf=inter_rdf.results.rdf,
            r=inter_rdf.results.bins,
            density=density,
        )

        saxs = Saxs(atomgroup=oxy, dq=0.1).run()
        scattering_vectors = saxs.results.scattering_vectors
        structure_factors = saxs.results.structure_factors

        # Interpolate direct method to have same scattering vector values. The
        # scattering vectors from the RDF cover a larger q range -> only take those
        # values up to the last value of the direct method.
        max_index = sum(scattering_vectors_rdf <= scattering_vectors[-1])
        structure_factors_interp = np.interp(
            x=scattering_vectors_rdf[:max_index],
            xp=scattering_vectors,
            fp=structure_factors,
        )

        assert_allclose(
            structure_factors_interp, structure_factors_rdf[:max_index], atol=4e-2
        )
