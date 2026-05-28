#!/usr/bin/env python
#
# Copyright (c) 2026 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the GNU Public Licence, v3 or any higher version
# SPDX-License-Identifier: GPL-3.0-or-later
"""Benchmarks for :class:`maicos.DielectricSphere`."""

import MDAnalysis as mda

from maicos import DielectricSphere
from tests.data import DIPOLE_GRO, DIPOLE_ITP


class DielectricSphereBenchmark:
    """Benchmark the DielectricSphere class."""

    timeout = 120

    def setup(self):
        """Set up the analysis objects."""
        self.dipole_universe = mda.Universe(
            DIPOLE_ITP, DIPOLE_GRO, topology_format="itp"
        )
        self.dipole_atoms = self.dipole_universe.atoms

    def time_dielectric_sphere_run(self):
        """Benchmark DielectricSphere on dipole system."""
        dielectric = DielectricSphere(self.dipole_atoms)
        dielectric.run()
