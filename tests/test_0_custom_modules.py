#!/usr/bin/env python3
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
#
# Copyright (c) 2019 Authors and contributors
# (see the file AUTHORS for the full list of names)
#
# Released under the GNU Public Licence, v2 or any higher version
# SPDX-License-Identifier: GPL-2.0-or-later

# Note: The custom test module has to be evaluated at first
# otherwise the module are not inplace and the tests fail...

import os
import shutil
import subprocess

import numpy as np
from numpy.testing import assert_almost_equal
import MDAnalysis as mda
import pytest


from modules.datafiles import WATER_TPR, WATER_TRR

# Ugly constrection but I was not able to get the code running with a fixture...
custom_dir = os.path.join(os.path.expanduser("~"), ".maicos")
try:
    os.mkdir(custom_dir)
except FileExistsError:
    pass

shutil.copy("../examples/maicos_costum_modules.py", custom_dir)
shutil.copy("../examples/analysis_example.py", custom_dir)
from maicos import AnalysisExample


class TestAnalysisExample(object):

    @pytest.fixture()
    def ag(self):
        u = mda.Universe(WATER_TPR, WATER_TRR)
        return u.atoms

    def test_cli(self):
        subprocess.check_call(['maicos', "analysisexample", "--help"])

    def test_analysis_example(self, ag):
        example = AnalysisExample(ag).run()
        assert_almost_equal(example.results['volume'].mean(),
                            15443.7,
                            decimal=1)

    def test_output(self, ag, tmpdir):
        with tmpdir.as_cwd():
            example = AnalysisExample(ag, save=True)
            example.run()
            example.save()
            res_volume = np.loadtxt(example.output)
            assert_almost_equal(example.results["volume"],
                                res_volume,
                                decimal=2)
