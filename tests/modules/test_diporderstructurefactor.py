#!/usr/bin/env python
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
#
# Copyright (c) 2024 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the GNU Public Licence, v3 or any higher version
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for the DiporderStructureFactor class."""
import sys
from pathlib import Path

import MDAnalysis as mda
import numpy as np
import pytest
from numpy.testing import assert_allclose

from maicos import DiporderStructureFactor, RDFDiporder
from maicos.lib.math import compute_rdf_structure_factor


sys.path.append(str(Path(__file__).parents[1]))
from data import WATER_TPR_NVT, WATER_XTC_NVT  # noqa: E402


class TestDiporderStructureFactor:
    """Tests for the DiporderStructureFactor class."""

    @pytest.fixture()
    def ag(self):
        """Import MDA universe of water in the NVT ensemble."""
        u = mda.Universe(WATER_TPR_NVT, WATER_XTC_NVT)
        return u.atoms

    def test_rdf_comparison(self, ag):
        """Test if the Fourier transformation of an RDF is the structure factor."""
        L = ag.universe.dimensions[0]  # we have a cubic box

        density = ag.residues.n_residues / ag.universe.trajectory.ts.volume

        inter_rdf = RDFDiporder(ag, bin_width=0.1, rmax=L / 2).run(step=10)

        scaterring_vectors_rdf, structure_factors_rdf = compute_rdf_structure_factor(
            rdf=1 + inter_rdf.results.rdf,
            r=inter_rdf.results.bins,
            density=density,
        )

        diporder_structure_factors = DiporderStructureFactor(
            atomgroup=ag.atoms, dq=0.1
        ).run(step=10)

        scattering_vectors = diporder_structure_factors.results.scattering_vectors
        structure_factors = diporder_structure_factors.results.structure_factors

        # Interpolate direct method to have same q values. q_rdf covers a larger q
        # range -> only take those values up to the last value of the direct method.
        max_index = sum(scaterring_vectors_rdf <= scattering_vectors[-1])
        struct_factor_interp = np.interp(
            scaterring_vectors_rdf[:max_index], scattering_vectors, structure_factors
        )

        # Ignore low q values because they converge very slowly.
        q_min = 0.5  # 1/Å
        min_index = np.argmin((scattering_vectors - q_min) ** 2)

        assert_allclose(
            struct_factor_interp[min_index:],
            structure_factors_rdf[min_index:max_index],
            atol=3e-2,
        )

    def test_q_values(self, ag):
        """Tests if claculates q values are within all possible q values."""
        qmin = 0
        qmax = 3
        dq = 0.05

        S_fac = DiporderStructureFactor(ag.atoms, qmin=qmin, qmax=qmax, dq=dq)
        S_fac.run(stop=1)
        q_ref = np.arange(qmin, qmax, dq) + 0.5 * dq

        assert set(S_fac.results.scattering_vectors).issubset(q_ref)

    def test_output(self, ag, monkeypatch, tmp_path):
        """Tests output name."""
        monkeypatch.chdir(tmp_path)

        S_fac = DiporderStructureFactor(ag.atoms, output="foo")
        S_fac.run(stop=1)

        S_fac.save()
        saxs_loaded = np.loadtxt("foo.dat")
        assert_allclose(S_fac.results.scattering_vectors, saxs_loaded[:, 0])
        assert_allclose(S_fac.results.structure_factors, saxs_loaded[:, 1])
