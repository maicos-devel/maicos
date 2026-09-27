# Copyright (c) 2026 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the GNU Public Licence, v3 or any higher version
# SPDX-License-Identifier: GPL-3.0-or-later
"""Benchmarks for :mod:`maicos.lib.math`."""

from typing import ClassVar

import numpy as np

from maicos.lib.math import transform_cylinder, transform_sphere


class CoordinateTransformBenchmark:
    """Benchmarks for cylindrical and spherical coordinate transformations."""

    params: ClassVar[list[int]] = [1_000, 100_000, 1_000_000]
    param_names: ClassVar[list[str]] = ["number_of_atoms"]

    def setup(self, number_of_atoms):
        """Generate random positions and an origin for the transformations."""
        rng = np.random.default_rng(42)
        self.positions = rng.random((number_of_atoms, 3)) * 50.0
        self.origin = np.array([25.0, 25.0, 25.0])

    def time_transform_cylinder(self, _number_of_atoms):
        """Time the cylindrical coordinate transformation."""
        transform_cylinder(self.positions, self.origin, dim=2)

    def time_transform_sphere(self, _number_of_atoms):
        """Time the spherical coordinate transformation."""
        transform_sphere(self.positions, self.origin)
