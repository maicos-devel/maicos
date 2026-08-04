#!/usr/bin/env python
#
# Copyright (c) 2026 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the GNU Public Licence, v3 or any higher version
# SPDX-License-Identifier: GPL-3.0-or-later
"""Per-module ``_single_frame`` cost as a function of atom count.

``setup`` builds a one-frame synthetic universe and runs the analysis over it, which
initialises every attribute the per-frame kernel needs. The benchmark itself then calls
``_single_frame`` alone, so neither the trajectory loop nor the coordinate transforms
(``unwrap``, ``pack``) are timed. The pair distribution modules get their own class
because their pairwise kernels need smaller systems to stay within the time budget.

The single frame makes :func:`maicos.lib.util.correlation_analysis` warn about the
trajectory being too short, which is expected here and muted so that it does not drown
the benchmark output.
"""

import warnings

from benchmarks.synthetic import make_universe
from maicos import (
    DensityCylinder,
    DensityPlanar,
    DensitySphere,
    DielectricCylinder,
    DielectricPlanar,
    DielectricSphere,
    DiporderCylinder,
    DiporderPlanar,
    DiporderSphere,
    PDFCylinder,
    PDFPlanar,
    TemperaturePlanar,
    VelocityCylinder,
    VelocityPlanar,
)

LINEAR = (
    DensityCylinder,
    DensityPlanar,
    DensitySphere,
    DielectricCylinder,
    DielectricPlanar,
    DielectricSphere,
    DiporderCylinder,
    DiporderPlanar,
    DiporderSphere,
    TemperaturePlanar,
    VelocityCylinder,
    VelocityPlanar,
)
PAIRWISE = (PDFCylinder, PDFPlanar)
MODULES = {cls.__name__: cls for cls in LINEAR + PAIRWISE}


class _AtomScaling:
    """Time ``_single_frame`` of a prepared analysis over growing atom counts."""

    timeout = 300
    param_names = ["module", "n_atoms"]

    def setup(self, module, n_atoms):
        """Build a one-frame universe and prepare the analysis on it."""
        self.analysis = MODULES[module](make_universe(n_atoms=n_atoms, n_frames=1))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.analysis.run()

    def time_single_frame(self, _module, _n_atoms):
        """Time one call of the per-frame kernel."""
        self.analysis._single_frame()


class AtomScaling(_AtomScaling):
    """Atom-count scaling of the modules whose kernel is linear in the atom count."""

    params = [[cls.__name__ for cls in LINEAR], [3_000, 10_000, 30_000, 100_000]]


class PairwiseAtomScaling(_AtomScaling):
    """Atom-count scaling of the pairwise pair distribution modules."""

    params = [[cls.__name__ for cls in PAIRWISE], [1_500, 3_000, 6_000, 12_000]]
