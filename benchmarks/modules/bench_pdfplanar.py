#!/usr/bin/env python
#
# Copyright (c) 2026 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the GNU Public Licence, v3 or any higher version
# SPDX-License-Identifier: GPL-3.0-or-later
"""Benchmarks for :class:`maicos.PDFPlanar`."""

import MDAnalysis as mda

from maicos import PDFPlanar
from tests.data import AIRWATER_TPR, AIRWATER_TRR


class PDFPlanarBenchmark:
    """Benchmark the PDFPlanar class."""

    timeout = 180

    def setup(self):
        """Set up the analysis objects."""
        self.universe = mda.Universe(AIRWATER_TPR, AIRWATER_TRR)
        self.atomgroup = self.universe.select_atoms("name OW")

    def time_pdf_planar_run(self):
        """Benchmark PDFPlanar.run() over the trajectory."""
        pdf = PDFPlanar(self.atomgroup)
        pdf.run()
