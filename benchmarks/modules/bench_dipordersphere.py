#!/usr/bin/env python
#
# Copyright (c) 2026 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the GNU Public Licence, v3 or any higher version
# SPDX-License-Identifier: GPL-3.0-or-later
"""Benchmarks for :class:`maicos.DiporderSphere`."""

import MDAnalysis as mda

from maicos import DiporderSphere
from tests.data import WATER_TPR_NPT, WATER_TRR_NPT


class DiporderSphereBenchmark:
    """Benchmark the DiporderSphere class."""

    timeout = 120

    def setup(self):
        """Set up the analysis objects."""
        self.universe = mda.Universe(WATER_TPR_NPT, WATER_TRR_NPT)
        self.atomgroup = self.universe.select_atoms("resname SOL")

    def time_diporder_sphere_run(self):
        """Benchmark DiporderSphere.run() over the trajectory."""
        diporder = DiporderSphere(self.atomgroup, bin_width=0.5)
        diporder.run()
