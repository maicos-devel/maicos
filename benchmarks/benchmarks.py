#!/usr/bin/env python
#
# Copyright (c) 2026 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the GNU Public Licence, v3 or any higher version
# SPDX-License-Identifier: GPL-3.0-or-later
"""Airspeed velocity benchmarks for MAICoS.

These benchmarks cover the core analysis modules across different geometries
(planar, cylindrical, spherical) and various analysis types.
"""

import sys
from pathlib import Path

import MDAnalysis as mda
import numpy as np

# Add tests directory to path to access test data
sys.path.insert(0, str(Path(__file__).parents[1]))

from maicos import (
    DensityCylinder,
    DensityPlanar,
    DensitySphere,
    DielectricCylinder,
    DielectricPlanar,
    DielectricSphere,
    DiporderPlanar,
    VelocityPlanar,
)
from maicos.core import AnalysisBase
from maicos.lib.math import transform_cylinder, transform_sphere
from tests.data import (
    AIRWATER_TPR,
    AIRWATER_TRR,
    DIPOLE_GRO,
    DIPOLE_ITP,
    SPCE_GRO,
    SPCE_ITP,
    WATER_GRO_NPT,
    WATER_TPR_NPT,
    WATER_TRR_NPT,
)


class StubAnalysis(AnalysisBase):
    """Minimal analysis that writes random observables — measures framework overhead."""

    def __init__(self, atomgroup, n_obs=10):
        self._n_obs = n_obs
        super().__init__(
            atomgroup=atomgroup,
            unwrap=False,
            pack=False,
            refgroup=None,
            jitter=0.0,
            wrap_compound="atoms",
            concfreq=0,
        )

    def _single_frame(self):
        self._obs.data = np.random.rand(self._n_obs)


# =============================================================================
# Density Profile Benchmarks
# =============================================================================


class DensityPlanarBenchmark:
    """Benchmark the DensityPlanar class."""

    timeout = 120

    def setup(self):
        """Set up the analysis objects."""
        self.universe = mda.Universe(AIRWATER_TPR, AIRWATER_TRR)
        self.atomgroup = self.universe.atoms

    def time_density_planar_run(self):
        """Benchmark DensityPlanar.run() over the trajectory."""
        density = DensityPlanar(self.atomgroup, dens="mass", bin_width=0.5)
        density.run()

    def time_density_planar_number_density(self):
        """Benchmark DensityPlanar with number density."""
        density = DensityPlanar(self.atomgroup, dens="number", bin_width=0.5)
        density.run()

    def time_density_planar_fine_bins(self):
        """Benchmark DensityPlanar with fine binning."""
        density = DensityPlanar(self.atomgroup, dens="mass", bin_width=0.1)
        density.run()


class DensityCylinderBenchmark:
    """Benchmark the DensityCylinder class."""

    timeout = 120

    def setup(self):
        """Set up the analysis objects."""
        self.universe = mda.Universe(WATER_TPR_NPT, WATER_TRR_NPT)
        self.atomgroup = self.universe.atoms

    def time_density_cylinder_run(self):
        """Benchmark DensityCylinder.run() over the trajectory."""
        density = DensityCylinder(self.atomgroup, dens="mass", bin_width=0.5)
        density.run()


class DensitySphereBenchmark:
    """Benchmark the DensitySphere class."""

    timeout = 120

    def setup(self):
        """Set up the analysis objects."""
        self.universe = mda.Universe(WATER_TPR_NPT, WATER_TRR_NPT)
        self.atomgroup = self.universe.atoms

    def time_density_sphere_run(self):
        """Benchmark DensitySphere.run() over the trajectory."""
        density = DensitySphere(self.atomgroup, dens="mass", bin_width=0.5)
        density.run()


# =============================================================================
# Dielectric Profile Benchmarks
# =============================================================================


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


# =============================================================================
# Dipole Order Benchmarks
# =============================================================================


class DiporderPlanarBenchmark:
    """Benchmark the DiporderPlanar class."""

    timeout = 120

    def setup(self):
        """Set up the analysis objects."""
        self.universe = mda.Universe(AIRWATER_TPR, AIRWATER_TRR)
        self.atomgroup = self.universe.select_atoms("resname SOL")

    def time_diporder_planar_run(self):
        """Benchmark DiporderPlanar.run() over the trajectory."""
        diporder = DiporderPlanar(self.atomgroup, bin_width=0.5)
        diporder.run()


# =============================================================================
# Velocity Profile Benchmarks
# =============================================================================


class VelocityPlanarBenchmark:
    """Benchmark the VelocityPlanar class."""

    timeout = 120

    def setup(self):
        """Set up the analysis objects."""
        self.universe = mda.Universe(AIRWATER_TPR, AIRWATER_TRR)
        self.atomgroup = self.universe.atoms

    def time_velocity_planar_run(self):
        """Benchmark VelocityPlanar.run() over the trajectory."""
        velocity = VelocityPlanar(self.atomgroup, bin_width=0.5)
        velocity.run()


# =============================================================================
# Memory Benchmarks
# =============================================================================


class MemoryBenchmarks:
    """Memory usage benchmarks for core analysis classes."""

    timeout = 180

    def setup(self):
        """Set up the analysis objects."""
        self.universe = mda.Universe(AIRWATER_TPR, AIRWATER_TRR)
        self.atomgroup = self.universe.atoms

    def peakmem_density_planar(self):
        """Peak memory for DensityPlanar analysis."""
        density = DensityPlanar(self.atomgroup, dens="mass", bin_width=0.5)
        density.run()

    def peakmem_density_cylinder(self):
        """Peak memory for DensityCylinder analysis."""
        density = DensityCylinder(self.atomgroup, dens="mass", bin_width=0.5)
        density.run()

    def peakmem_density_sphere(self):
        """Peak memory for DensitySphere analysis."""
        density = DensitySphere(self.atomgroup, dens="mass", bin_width=0.5)
        density.run()


# =============================================================================
# Scaling Benchmarks
# =============================================================================


class ScalingBenchmarks:
    """Benchmarks to test scaling with different parameters."""

    timeout = 180
    params = [0.1, 0.5, 1.0, 2.0]
    param_names = ["bin_width"]

    def setup(self, bin_width):
        """Set up the analysis objects."""
        self.universe = mda.Universe(AIRWATER_TPR, AIRWATER_TRR)
        self.atomgroup = self.universe.atoms
        self.bin_width = bin_width

    def time_density_planar_scaling(self, bin_width):
        """Time DensityPlanar with varying bin widths."""
        density = DensityPlanar(self.atomgroup, dens="mass", bin_width=bin_width)
        density.run()


# =============================================================================
# Universe Loading Benchmarks
# =============================================================================


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


# =============================================================================
# Grouping Mode Benchmarks
# =============================================================================


class GroupingBenchmark:
    """Benchmark different grouping modes."""

    timeout = 180
    params = ["atoms", "residues", "molecules"]
    param_names = ["grouping"]

    def setup(self, grouping):
        """Set up the analysis objects."""
        self.universe = mda.Universe(AIRWATER_TPR, AIRWATER_TRR)
        self.atomgroup = self.universe.select_atoms("resname SOL")
        self.grouping = grouping

    def time_density_planar_grouping(self, grouping):
        """Time DensityPlanar with different grouping modes."""
        density = DensityPlanar(
            self.atomgroup, dens="mass", bin_width=0.5, grouping=grouping
        )
        density.run()


# =============================================================================
# Transformation Benchmarks
# =============================================================================


class CoordinateTransformBenchmark:
    """Benchmarks for cylindrical and spherical coordinate transformations."""

    params = [1_000, 100_000, 1_000_000]
    param_names = ["number_of_atoms"]

    def setup(self, number_of_atoms):
        rng = np.random.default_rng(42)
        self.positions = rng.random((number_of_atoms, 3)) * 50.0
        self.origin = np.array([25.0, 25.0, 25.0])

    def time_transform_cylinder(self, _number_of_atoms):
        """Time the cylindrical coordinate transformation."""
        transform_cylinder(self.positions, self.origin, dim=2)

    def time_transform_sphere(self, _number_of_atoms):
        """Time the spherical coordinate transformation."""
        transform_sphere(self.positions, self.origin)
