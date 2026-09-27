# Copyright (c) 2026 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the GNU Public Licence, v3 or any higher version
# SPDX-License-Identifier: GPL-3.0-or-later
"""Benchmarks for :class:`maicos.DensityCylinder`."""

import MDAnalysis as mda

from maicos import DensityCylinder
from tests.data import WATER_TPR_NPT, WATER_TRR_NPT


class DensityCylinderBenchmark:
    """Benchmark the DensityCylinder class."""

    timeout = 120

    def setup(self):
        """Set up the analysis objects."""
        self.universe = mda.Universe(WATER_TPR_NPT, WATER_TRR_NPT)
        self.atomgroup = self.universe.atoms

    def time_density_cylinder_run(self):
        """Benchmark DensityCylinder.run() over the trajectory."""
        density = DensityCylinder(self.atomgroup, dens="mass", bin_width=0.5)
        density.run()

    def peakmem_density_cylinder(self):
        """Peak memory for DensityCylinder analysis."""
        density = DensityCylinder(self.atomgroup, dens="mass", bin_width=0.5)
        density.run()
