# Copyright (c) 2026 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the GNU Public Licence, v3 or any higher version
# SPDX-License-Identifier: GPL-3.0-or-later
"""Benchmarks for :class:`maicos.DensitySphere`."""

import MDAnalysis as mda

from maicos import DensitySphere
from tests.data import WATER_TPR_NPT, WATER_TRR_NPT


class DensitySphereBenchmark:
    """Benchmark the DensitySphere class."""

    timeout = 120

    def setup(self):
        """Set up the analysis objects."""
        self.universe = mda.Universe(WATER_TPR_NPT, WATER_TRR_NPT)
        self.atomgroup = self.universe.atoms

    def time_density_sphere_run(self):
        """Benchmark DensitySphere.run() over the trajectory."""
        density = DensitySphere(self.atomgroup, dens="mass", bin_width=0.5)
        density.run()

    def peakmem_density_sphere(self):
        """Peak memory for DensitySphere analysis."""
        density = DensitySphere(self.atomgroup, dens="mass", bin_width=0.5)
        density.run()
