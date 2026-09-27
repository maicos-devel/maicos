# Copyright (c) 2026 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the GNU Public Licence, v3 or any higher version
# SPDX-License-Identifier: GPL-3.0-or-later
"""Benchmarks for :class:`maicos.DielectricPlanar`."""

import MDAnalysis as mda

from maicos import DielectricPlanar
from tests.data import DIPOLE_GRO, DIPOLE_ITP, SPCE_GRO, SPCE_ITP


class DielectricPlanarBenchmark:
    """Benchmark the DielectricPlanar class."""

    timeout = 120

    def setup(self):
        """Set up the analysis objects."""
        self.dipole_universe = mda.Universe(
            DIPOLE_ITP, DIPOLE_GRO, topology_format="itp"
        )
        self.dipole_atoms = self.dipole_universe.atoms

        self.spce_universe = mda.Universe(SPCE_ITP, SPCE_GRO, topology_format="itp")
        self.spce_atoms = self.spce_universe.atoms

    def time_dielectric_planar_single_frame(self):
        """Benchmark DielectricPlanar on a single frame dipole system."""
        dielectric = DielectricPlanar(self.dipole_atoms)
        dielectric.run()

    def time_dielectric_planar_spce(self):
        """Benchmark DielectricPlanar on an SPC/E water molecule."""
        dielectric = DielectricPlanar(self.spce_atoms)
        dielectric.run()
