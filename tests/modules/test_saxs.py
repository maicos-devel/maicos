#!/usr/bin/env python
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
#
# Copyright (c) 2023 Authors and contributors
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
from data import WATER_GRO, WATER_TPR, WATER_TRR
from MDAnalysis.analysis.rdf import InterRDF
from numpy.testing import assert_allclose, assert_equal

from maicos import Saxs
from maicos.lib.math import compute_form_factor, compute_rdf_structure_factor


sys.path.append(str(Path(__file__).parents[1]))


class ReferenceAtomGroups:
    """Super class with methods reference AtomGroups for tests."""

    @pytest.fixture()
    def ag_single_frame(self):
        """Import MDA universe."""
        u = mda.Universe(WATER_TPR, WATER_GRO)
        return u.atoms


class TestSaxs(ReferenceAtomGroups):
    """Tests for the Saxs class."""

    @pytest.fixture()
    def ag_single_frame(self):
        """Import MDA universe."""
        u = mda.Universe(WATER_TPR, WATER_GRO)
        return u.atoms

    @pytest.fixture()
    def ag(self):
        """Import MDA universe."""
        u = mda.Universe(WATER_TPR, WATER_TRR)
        return u.atoms

    def test_one_frame(sef, ag_single_frame):
        """Test Saxs on one frame.

        Test if the division by the number of frames is correct.
        """
        saxs = Saxs(ag_single_frame, endq=20).run()
        assert_allclose(saxs.results.scat_factor[0], 1.6047, rtol=1e-3)

    def test_theta(self, ag_single_frame, monkeypatch, tmp_path):
        """Test min & max theta conditions on one frame."""
        monkeypatch.chdir(tmp_path)

        with pytest.raises(ValueError, match=r"mintheta \(-10°\) has to between 0"):
            Saxs(ag_single_frame, mintheta=-10, maxtheta=190).run()

    def test_bin_spectrum(self, ag_single_frame):
        """Test when bin_spectrum is False."""
        saxs = Saxs(ag_single_frame, bin_spectrum=False).run()
        assert_equal(type(saxs.q_factor).__name__ == "ndarray", True)

    def test_rdf_comparison(self, ag):
        """Test if the Fourier transformation of an RDF is the structure factor."""
        oxy = ag.select_atoms("name OW")
        L = ag.universe.dimensions[0]  # we have a cubic box

        density = oxy.n_atoms / np.prod(ag.universe.trajectory.ts.volume)

        inter_rdf = InterRDF(
            oxy,
            oxy,
            nbins=300,
            range=(0, L / 2),
            exclude_same="residue",
        ).run()

        q_rdf, struct_factor_rdf = compute_rdf_structure_factor(
            rdf=inter_rdf.results.rdf,
            r=inter_rdf.results.bins,
            density=density,
        )

        S_fac = Saxs(atomgroup=oxy, dq=0.1).run()

        q = S_fac.results.q
        scat_factor = S_fac.results.scat_factor

        # Normalize ONLY with respect to the number of particles -> Divide by the form
        # factor which is applie in the SAXS module
        struct_factor = scat_factor / compute_form_factor(q, "O") ** 2

        # Interpolate direct method to have same q values. q_rdf covers a larger q
        # range -> only take those values up the maximum value of 1
        max_index = sum(q_rdf <= q[-1])
        struct_factor_interp = np.interp(q_rdf[:max_index], q, struct_factor)

        assert_allclose(struct_factor_interp, struct_factor_rdf[:max_index], atol=3e-2)
