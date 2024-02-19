#!/usr/bin/env python
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
#
# Copyright (c) 2024 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the GNU Public Licence, v3 or any higher version
# SPDX-License-Identifier: GPL-3.0-or-later
#
# Note: The custom test module has to be evaluated at first otherwise the module are not
# inplace and the tests fail...
"""Tests for custom modules."""

import shutil
import subprocess
from pathlib import Path

import MDAnalysis as mda
import numpy as np
import pytest
from data import WATER_TPR_NPT, WATER_TRR_NPT
from numpy.testing import assert_allclose


COSTUM_DIR = Path().home() / ".maicos"
COSTUM_DIR.mkdir(exist_ok=True)

REPO_PATH = Path(__file__).parents[1]
shutil.copy(REPO_PATH / "examples/maicos_custom_modules.py", COSTUM_DIR)
shutil.copy(REPO_PATH / "examples/own_module.py", COSTUM_DIR)

custom_modules = pytest.importorskip("maicos_custom_modules")


class TestAnalysisExample(object):
    """Test for class AnalysisExample."""

    @pytest.fixture()
    def ag(self):
        """Import MDA universe."""
        u = mda.Universe(WATER_TPR_NPT, WATER_TRR_NPT)
        return u.atoms

    def test_cli(self):
        """Test cli."""
        subprocess.check_call(["maicos", "analysisexample", "--help"])

    def test_analysis_example(self, ag):
        """Test analysis example."""
        example = custom_modules.AnalysisExample(ag).run()
        assert_allclose(example.results["volume"].mean(), 15443.7, rtol=1e-1)

    def test_output(self, ag, monkeypatch, tmp_path):
        """Test outputs."""
        monkeypatch.chdir(tmp_path)
        example = custom_modules.AnalysisExample(ag).run()
        example.save()
        res_volume = np.loadtxt(example.output)
        assert_allclose(example.results["volume"], res_volume, rtol=1e-2)
