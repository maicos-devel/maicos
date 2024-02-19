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

        q_rdf, struct_factor_rdf = compute_rdf_structure_factor(
            rdf=1 + inter_rdf.results.rdf,
            r=inter_rdf.results.bins,
            density=density,
        )

        S_fac = DiporderStructureFactor(atomgroup=ag.atoms, dq=0.1).run(step=10)

        q = S_fac.results.q
        struct_factor = S_fac.results.struct_factor

        # Interpolate direct method to have same q values. q_rdf covers a larger q
        # range -> only take those values up to the last value of the direct method.
        max_index = sum(q_rdf <= q[-1])
        struct_factor_interp = np.interp(q_rdf[:max_index], q, struct_factor)

        # Ignore qt low q values because they converge very slowly.
        q_min = 0.5  # 1/Å
        min_index = np.argmin((q - q_min) ** 2)

        assert_allclose(
            struct_factor_interp[min_index:],
            struct_factor_rdf[min_index:max_index],
            atol=3e-2,
        )

    def test_q_values(self, ag):
        """Tests if claculates q values are within all possible q values."""
        startq = 0
        endq = 3
        dq = 0.05

        S_fac = DiporderStructureFactor(ag.atoms, startq=startq, endq=endq, dq=dq)
        S_fac.run(stop=1)
        q_ref = np.arange(startq, endq, dq) + 0.5 * dq

        assert set(S_fac.results.q).issubset(q_ref)

    def test_output_name(self, ag, monkeypatch, tmp_path):
        """Tests output name."""
        monkeypatch.chdir(tmp_path)

        S_fac = DiporderStructureFactor(ag.atoms, output="foo")
        S_fac.run(stop=1)
        S_fac.save()
        open("foo.dat")
