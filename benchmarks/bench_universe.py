# Copyright (c) 2026 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the GNU Public Licence, v3 or any higher version
# SPDX-License-Identifier: GPL-3.0-or-later
"""Baseline MDAnalysis universe-loading benchmarks (no maicos analysis)."""

import MDAnalysis as mda

from tests.data import AIRWATER_TPR, AIRWATER_TRR, WATER_GRO_NPT


class UniverseLoadingBenchmark:
    """Benchmark universe loading and atom selection operations."""

    timeout = 60

    def time_load_universe_gro(self):
        """Time loading a GRO file."""
        mda.Universe(str(WATER_GRO_NPT))

    def time_load_universe_tpr_trr(self):
        """Time loading a TPR with TRR trajectory."""
        mda.Universe(AIRWATER_TPR, AIRWATER_TRR)

    def time_atom_selection(self):
        """Time common atom selections."""
        universe = mda.Universe(AIRWATER_TPR, AIRWATER_TRR)
        universe.select_atoms("resname SOL")
        universe.select_atoms("name OW")
        universe.select_atoms("type OW HW")
