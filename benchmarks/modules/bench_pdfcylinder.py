#!/usr/bin/env python
#
# Copyright (c) 2026 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the GNU Public Licence, v3 or any higher version
# SPDX-License-Identifier: GPL-3.0-or-later
"""Benchmarks for :class:`maicos.PDFCylinder`."""

import MDAnalysis as mda

from maicos import PDFCylinder
from tests.data import WATER_TPR_NPT, WATER_TRR_NPT


class PDFCylinderBenchmark:
    """Benchmark the PDFCylinder class."""

    timeout = 180

    def setup(self):
        """Set up the analysis objects."""
        self.universe = mda.Universe(WATER_TPR_NPT, WATER_TRR_NPT)
        self.atomgroup = self.universe.select_atoms("name OW")

    def time_pdf_cylinder_run(self):
        """Benchmark PDFCylinder.run() over the trajectory."""
        pdf = PDFCylinder(self.atomgroup)
        pdf.run()
