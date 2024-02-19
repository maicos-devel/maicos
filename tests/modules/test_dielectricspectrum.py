#!/usr/bin/env python
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
#
# Copyright (c) 2024 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the GNU Public Licence, v3 or any higher version
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for the DielectricSpectrum class."""
import sys
from pathlib import Path

import MDAnalysis as mda
import numpy as np
import pytest
from numpy.testing import assert_allclose

from maicos import DielectricSpectrum


sys.path.append(str(Path(__file__).parents[1]))
from data import WATER_TPR_NPT, WATER_TRR_NPT  # noqa: E402


class TestDielectricSpectrum(object):
    """Tests for the DielectricSpectrum class."""

    @pytest.fixture()
    def ag(self):
        """Import MDA universe."""
        u = mda.Universe(WATER_TPR_NPT, WATER_TRR_NPT)
        return u.atoms

    def test_output_name(self, ag, monkeypatch, tmp_path):
        """Test output name."""
        monkeypatch.chdir(tmp_path)

        ds = DielectricSpectrum(ag)
        ds.run()
        ds.save()
        open("susc.dat")
        open("P_tseries.npy")
        open("tseries.npy")
        open("V.txt")

    def test_output_name_prefix(self, ag, monkeypatch, tmp_path):
        """Test output name with custom prefix."""
        monkeypatch.chdir(tmp_path)

        ds = DielectricSpectrum(ag, output_prefix="foo")
        ds.run()
        ds.save()
        open("foo_susc.dat")
        open("foo_P_tseries.npy")
        open("foo_tseries.npy")
        open("foo_V.txt")

    def test_output_name_binned(self, ag, monkeypatch, tmp_path):
        """Test output name of binned data."""
        """
        The parameters are not meant to be sensible,
        but just to force the binned output.
        """
        monkeypatch.chdir(tmp_path)

        ds = DielectricSpectrum(ag, bins=5, binafter=0, segs=5)
        ds.run()
        ds.save()
        open("susc.dat")
        open("susc_binned.dat")
        open("P_tseries.npy")
        open("tseries.npy")
        open("V.txt")

    def test_output(self, ag, monkeypatch, tmp_path):
        """Test output values by comparing with magic numbers."""
        monkeypatch.chdir(tmp_path)

        ds = DielectricSpectrum(ag)
        ds.run()

        V = 1559814.4
        nu = [0.0, 0.2, 0.5, 0.7, 1.0]
        susc = [27.5 + 0.0j, 2.9 + 22.3j, -5.0 + 3.6j, -0.5 + 10.7j, -16.8 + 3.5j]
        dsusc = [3.4 + 0.0j, 0.4 + 2.9j, 1.0 + 0.5j, 0.3 + 1.5j, 2.0 + 0.6j]

        assert_allclose(ds.V, V, rtol=1e-1)
        assert_allclose(ds.results.nu, nu, rtol=1)
        assert_allclose(ds.results.susc, susc, rtol=1e-1)
        assert_allclose(ds.results.dsusc, dsusc, rtol=1e-1)

    def test_binning(self, ag, monkeypatch, tmp_path):
        """Test binning & seglen case."""
        monkeypatch.chdir(tmp_path)

        ds = DielectricSpectrum(ag, nobin=False, segs=2, bins=49)
        ds.run()
        assert_allclose(np.mean(ds.results.nu_binned), 0.57, rtol=1e-2)
        ds.save()
