#!/usr/bin/env python
#
# Copyright (c) 2025 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the GNU Public Licence, v3 or any higher version
# SPDX-License-Identifier: GPL-3.0-or-later
"""airspeed velocity of an unladen MAICoS."""

import sys
from pathlib import Path

import MDAnalysis as mda

from maicos import DielectricPlanar

sys.path.append(str(Path(__file__).parents[1]))

from tests.data import (
    DIPOLE_GRO,
    DIPOLE_ITP,
)


class DielectricPlanarBenchmark:
    """Benchmark the DielectricPlanar class."""

    def setup(self):
        """Setup the analysis objects."""
        self.dipole1 = mda.Universe(DIPOLE_ITP, DIPOLE_GRO, topology_format="itp").atoms
        self.dielectric = DielectricPlanar(self.dipole1)
        self.dielectric._prepare()

    def time_single_dielectric_planar(self):
        """Benchmark of a complete run over a single frame."""
        self.dielectric.run()
