#!/usr/bin/env python
#
# Copyright (c) 2026 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the GNU Public Licence, v3 or any higher version
# SPDX-License-Identifier: GPL-3.0-or-later
"""Per-module ``_single_frame`` cost as a function of atom count.

Each class benchmarks one module over a 1-frame synthetic universe at
``n_atoms`` in [300, 1000, 3000, 10000] (PDF modules use [300, 1000, 3000]
because their O(N²) pairwise kernel makes 10 k atoms prohibitively slow).

``unwrap=False, pack=False`` are passed wherever possible to isolate the
analysis kernel from coordinate-transform overhead.
"""

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

_N_ATOMS = [300, 1000, 3000, 10000]
_N_ATOMS_PDF = [300, 1000, 3000]


def _make(n_atoms):
    """1-frame synthetic atomgroup with water-like residue layout."""
    return make_universe(n_atoms=n_atoms, n_frames=1, n_residues=n_atoms // 3)


class DensityPlanarAtomScaling:
    """_single_frame cost of DensityPlanar vs. atom count."""

    timeout = 300
    params = _N_ATOMS
    param_names = ["n_atoms"]

    def setup(self, n_atoms):
        self.ag = _make(n_atoms)

    def time_single_frame(self, n_atoms):
        DensityPlanar(self.ag, dens="mass", bin_width=1.0, unwrap=False, pack=False).run()


class DensityCylinderAtomScaling:
    """_single_frame cost of DensityCylinder vs. atom count."""

    timeout = 300
    params = _N_ATOMS
    param_names = ["n_atoms"]

    def setup(self, n_atoms):
        self.ag = _make(n_atoms)

    def time_single_frame(self, n_atoms):
        DensityCylinder(
            self.ag, dens="mass", bin_width=1.0, unwrap=False, pack=False
        ).run()


class DensitySphereAtomScaling:
    """_single_frame cost of DensitySphere vs. atom count."""

    timeout = 300
    params = _N_ATOMS
    param_names = ["n_atoms"]

    def setup(self, n_atoms):
        self.ag = _make(n_atoms)

    def time_single_frame(self, n_atoms):
        DensitySphere(
            self.ag, dens="mass", bin_width=1.0, unwrap=False, pack=False
        ).run()


class DielectricPlanarAtomScaling:
    """_single_frame cost of DielectricPlanar vs. atom count."""

    timeout = 300
    params = _N_ATOMS
    param_names = ["n_atoms"]

    def setup(self, n_atoms):
        self.ag = _make(n_atoms)

    def time_single_frame(self, n_atoms):
        DielectricPlanar(self.ag, bin_width=1.0, unwrap=False, pack=False).run()


class DielectricCylinderAtomScaling:
    """_single_frame cost of DielectricCylinder vs. atom count."""

    timeout = 300
    params = _N_ATOMS
    param_names = ["n_atoms"]

    def setup(self, n_atoms):
        self.ag = _make(n_atoms)

    def time_single_frame(self, n_atoms):
        DielectricCylinder(self.ag, bin_width=1.0, unwrap=False, pack=False).run()


class DielectricSphereAtomScaling:
    """_single_frame cost of DielectricSphere vs. atom count."""

    timeout = 300
    params = _N_ATOMS
    param_names = ["n_atoms"]

    def setup(self, n_atoms):
        self.ag = _make(n_atoms)

    def time_single_frame(self, n_atoms):
        DielectricSphere(self.ag, bin_width=1.0, unwrap=False, pack=False).run()


class DiporderPlanarAtomScaling:
    """_single_frame cost of DiporderPlanar vs. atom count."""

    timeout = 300
    params = _N_ATOMS
    param_names = ["n_atoms"]

    def setup(self, n_atoms):
        self.ag = _make(n_atoms)

    def time_single_frame(self, n_atoms):
        DiporderPlanar(self.ag, bin_width=1.0, unwrap=False, pack=False).run()


class DiporderCylinderAtomScaling:
    """_single_frame cost of DiporderCylinder vs. atom count."""

    timeout = 300
    params = _N_ATOMS
    param_names = ["n_atoms"]

    def setup(self, n_atoms):
        self.ag = _make(n_atoms)

    def time_single_frame(self, n_atoms):
        DiporderCylinder(self.ag, bin_width=1.0, unwrap=False, pack=False).run()


class DiporderSphereAtomScaling:
    """_single_frame cost of DiporderSphere vs. atom count."""

    timeout = 300
    params = _N_ATOMS
    param_names = ["n_atoms"]

    def setup(self, n_atoms):
        self.ag = _make(n_atoms)

    def time_single_frame(self, n_atoms):
        DiporderSphere(self.ag, bin_width=1.0, unwrap=False, pack=False).run()


class TemperaturePlanarAtomScaling:
    """_single_frame cost of TemperaturePlanar vs. atom count."""

    timeout = 300
    params = _N_ATOMS
    param_names = ["n_atoms"]

    def setup(self, n_atoms):
        self.ag = _make(n_atoms)

    def time_single_frame(self, n_atoms):
        TemperaturePlanar(self.ag, bin_width=1.0, unwrap=False, pack=False).run()


class VelocityPlanarAtomScaling:
    """_single_frame cost of VelocityPlanar vs. atom count."""

    timeout = 300
    params = _N_ATOMS
    param_names = ["n_atoms"]

    def setup(self, n_atoms):
        self.ag = _make(n_atoms)

    def time_single_frame(self, n_atoms):
        VelocityPlanar(self.ag, bin_width=1.0, unwrap=False, pack=False).run()


class VelocityCylinderAtomScaling:
    """_single_frame cost of VelocityCylinder vs. atom count."""

    timeout = 300
    params = _N_ATOMS
    param_names = ["n_atoms"]

    def setup(self, n_atoms):
        self.ag = _make(n_atoms)

    def time_single_frame(self, n_atoms):
        VelocityCylinder(self.ag, bin_width=1.0, unwrap=False, pack=False).run()



class PDFPlanarAtomScaling:
    """_single_frame cost of PDFPlanar vs. atom count (O(N²), capped at 3000)."""

    timeout = 600
    params = _N_ATOMS_PDF
    param_names = ["n_atoms"]

    def setup(self, n_atoms):
        self.ag = _make(n_atoms)

    def time_single_frame(self, n_atoms):
        PDFPlanar(self.ag, bin_width=1.0, unwrap=False, pack=False).run()


class PDFCylinderAtomScaling:
    """_single_frame cost of PDFCylinder vs. atom count (O(N²), capped at 3000)."""

    timeout = 600
    params = _N_ATOMS_PDF
    param_names = ["n_atoms"]

    def setup(self, n_atoms):
        self.ag = _make(n_atoms)

    def time_single_frame(self, n_atoms):
        PDFCylinder(self.ag, bin_width=1.0, unwrap=False, pack=False).run()
