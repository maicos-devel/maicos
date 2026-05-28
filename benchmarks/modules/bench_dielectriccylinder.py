#!/usr/bin/env python
#
# Copyright (c) 2026 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the GNU Public Licence, v3 or any higher version
# SPDX-License-Identifier: GPL-3.0-or-later
"""Benchmarks for :class:`maicos.DielectricCylinder`."""

import MDAnalysis as mda

from maicos import DielectricCylinder
from tests.data import DIPOLE_GRO, DIPOLE_ITP


class DielectricCylinderBenchmark:
    """Benchmark the DielectricCylinder class."""

    timeout = 120

    def setup(self):
        """Set up the analysis objects."""
        self.dipole_universe = mda.Universe(
            DIPOLE_ITP, DIPOLE_GRO, topology_format="itp"
        )
        self.dipole_atoms = self.dipole_universe.atoms

    def time_dielectric_cylinder_run(self):
        """Benchmark DielectricCylinder on dipole system."""
        dielectric = DielectricCylinder(self.dipole_atoms)
        dielectric.run()
