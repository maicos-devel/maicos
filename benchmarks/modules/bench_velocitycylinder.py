#!/usr/bin/env python
#
# Copyright (c) 2026 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the GNU Public Licence, v3 or any higher version
# SPDX-License-Identifier: GPL-3.0-or-later
"""Benchmarks for :class:`maicos.VelocityCylinder`."""

import MDAnalysis as mda

from maicos import VelocityCylinder
from tests.data import WATER_TPR_NPT, WATER_TRR_NPT


class VelocityCylinderBenchmark:
    """Benchmark the VelocityCylinder class."""

    timeout = 120

    def setup(self):
        """Set up the analysis objects."""
        self.universe = mda.Universe(WATER_TPR_NPT, WATER_TRR_NPT)
        self.atomgroup = self.universe.atoms

    def time_velocity_cylinder_run(self):
        """Benchmark VelocityCylinder.run() over the trajectory."""
        velocity = VelocityCylinder(self.atomgroup, bin_width=0.5)
        velocity.run()
