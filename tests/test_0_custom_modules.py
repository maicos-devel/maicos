#!/usr/bin/env python
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
#
# Copyright (c) 2023 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the GNU Public Licence, v3 or any higher version
# SPDX-License-Identifier: GPL-3.0-or-later
#
# Note: The custom test module has to be evaluated at first otherwise the module are not
# inplace and the tests fail...
"""Tests for custom modules."""

import os
import shutil
import subprocess

import MDAnalysis as mda
import numpy as np
import pytest
from data import WATER_TPR, WATER_TRR
from numpy.testing import assert_almost_equal
from pkg_resources import resource_filename


# Ugly constrection but I was not able to get the code running with a fixture...
custom_dir = os.path.join(os.path.expanduser("~"), ".maicos")
try:
    os.mkdir(custom_dir)
except FileExistsError:
    pass

EXAMPLE_PATH = resource_filename(__name__, "../examples")
shutil.copy(os.path.join(EXAMPLE_PATH, "maicos_custom_modules.py"), custom_dir)
shutil.copy(os.path.join(EXAMPLE_PATH, "own_module.py"), custom_dir)

from maicos import AnalysisExample  # noqa: E402


class TestAnalysisExample(object):
    """Test for class AnalysisExample."""

    @pytest.fixture()
    def ag(self):
        """Import MDA universe."""
        u = mda.Universe(WATER_TPR, WATER_TRR)
        return u.atoms

    def test_cli(self):
        """Test cli."""
        subprocess.check_call(["maicos", "analysisexample", "--help"])

    def test_analysis_example(self, ag):
        """Test analysis example."""
        example = AnalysisExample(ag).run()
        assert_almost_equal(example.results["volume"].mean(), 15443.7, decimal=1)

    def test_output(self, ag, tmpdir):
        """Test outputs."""
        with tmpdir.as_cwd():
            example = AnalysisExample(ag).run()
            example.save()
            res_volume = np.loadtxt(example.output)
            assert_almost_equal(example.results["volume"], res_volume, decimal=2)
