#!/usr/bin/env python
#
# Copyright (c) 2026 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the GNU Public Licence, v3 or any higher version
# SPDX-License-Identifier: GPL-3.0-or-later
"""Benchmarks for :class:`maicos.TemperaturePlanar`."""

import MDAnalysis as mda

from maicos import TemperaturePlanar
from tests.data import AIRWATER_TPR, AIRWATER_TRR


class TemperaturePlanarBenchmark:
    """Benchmark the TemperaturePlanar class."""

    timeout = 120

    def setup(self):
        """Set up the analysis objects."""
        self.universe = mda.Universe(AIRWATER_TPR, AIRWATER_TRR)
        self.atomgroup = self.universe.atoms

    def time_temperature_planar_run(self):
        """Benchmark TemperaturePlanar.run() over the trajectory."""
        temperature = TemperaturePlanar(self.atomgroup, bin_width=0.5)
        temperature.run()
