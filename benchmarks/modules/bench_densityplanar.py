# Copyright (c) 2026 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the GNU Public Licence, v3 or any higher version
# SPDX-License-Identifier: GPL-3.0-or-later
"""Benchmarks for :class:`maicos.DensityPlanar`."""

from typing import ClassVar

import MDAnalysis as mda

from maicos import DensityPlanar
from tests.data import AIRWATER_TPR, AIRWATER_TRR


class DensityPlanarBenchmark:
    """Benchmark the DensityPlanar class."""

    timeout = 120

    def setup(self):
        """Set up the analysis objects."""
        self.universe = mda.Universe(AIRWATER_TPR, AIRWATER_TRR)
        self.atomgroup = self.universe.atoms

    def time_density_planar_run(self):
        """Benchmark DensityPlanar.run() over the trajectory."""
        density = DensityPlanar(self.atomgroup, dens="mass", bin_width=0.5)
        density.run()

    def time_density_planar_number_density(self):
        """Benchmark DensityPlanar with number density."""
        density = DensityPlanar(self.atomgroup, dens="number", bin_width=0.5)
        density.run()

    def time_density_planar_fine_bins(self):
        """Benchmark DensityPlanar with fine binning."""
        density = DensityPlanar(self.atomgroup, dens="mass", bin_width=0.1)
        density.run()

    def peakmem_density_planar(self):
        """Peak memory for DensityPlanar analysis."""
        density = DensityPlanar(self.atomgroup, dens="mass", bin_width=0.5)
        density.run()


class DensityPlanarScalingBenchmark:
    """Benchmark DensityPlanar across a range of bin widths."""

    timeout = 180
    params: ClassVar[list[float]] = [0.1, 0.5, 1.0, 2.0]
    param_names: ClassVar[list[str]] = ["bin_width"]

    def setup(self, _bin_width):
        """Set up the analysis objects."""
        self.universe = mda.Universe(AIRWATER_TPR, AIRWATER_TRR)
        self.atomgroup = self.universe.atoms

    def time_density_planar_scaling(self, bin_width):
        """Time DensityPlanar with varying bin widths."""
        density = DensityPlanar(self.atomgroup, dens="mass", bin_width=bin_width)
        density.run()


class DensityPlanarGroupingBenchmark:
    """Benchmark DensityPlanar across grouping modes."""

    timeout = 180
    params: ClassVar[list[str]] = ["atoms", "residues", "molecules"]
    param_names: ClassVar[list[str]] = ["grouping"]

    def setup(self, _grouping):
        """Set up the analysis objects."""
        self.universe = mda.Universe(AIRWATER_TPR, AIRWATER_TRR)
        self.atomgroup = self.universe.select_atoms("resname SOL")

    def time_density_planar_grouping(self, grouping):
        """Time DensityPlanar with different grouping modes."""
        density = DensityPlanar(
            self.atomgroup, dens="mass", bin_width=0.5, grouping=grouping
        )
        density.run()
