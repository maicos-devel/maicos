#!/usr/bin/env python
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
#
# Copyright (c) 2023 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the GNU Public Licence, v3 or any higher version
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for the base modules."""
import inspect
import logging
import sys
from pathlib import Path

import MDAnalysis as mda
import numpy as np
import pytest
from MDAnalysis.analysis.base import Results
from MDAnalysis.core._get_readers import get_reader_for
from numpy.testing import assert_allclose, assert_equal

import maicos
from maicos.core import CylinderBase, ProfileCylinderBase
from maicos.lib.math import transform_cylinder
from maicos.lib.weights import density_weights


sys.path.append(Path(__file__).parents[1])

from data import AIRWATER_TPR, AIRWATER_TRR, WATER_GRO, WATER_TPR  # noqa: E402


class CylinderClass(CylinderBase):
    """Tests for the Planar Base class."""

    def __init__(
        self,
        atomgroups,
        pos_arg,
        opt_arg="foo",
        dim=2,
        zmin=None,
        zmax=None,
        rmin=0,
        rmax=None,
        bin_width=1,
        **kwargs,
    ):
        super().__init__(
            atomgroups=atomgroups,
            dim=dim,
            zmin=zmin,
            zmax=zmax,
            rmin=rmin,
            rmax=rmax,
            bin_width=bin_width,
            multi_group=True,
            **kwargs,
        )
        self.pos_arg = pos_arg
        self.opt_arg = opt_arg

    def _prepare(self):
        super()._prepare()
        self.prepared = True

    def _single_frame(self):
        super()._single_frame()
        self.ran = True

    def _conclude(self):
        super()._conclude()
        self.calculated_results = True


class TestCylinderBase(object):
    """Tests for the TestPlanarBase class."""

    @pytest.fixture()
    def ag(self):
        """Import MDA universe."""
        u = mda.Universe(AIRWATER_TPR, AIRWATER_TRR, in_memory=True)
        return u.atoms

    @pytest.fixture()
    def cylinder_class_obj(self, ag):
        """Planar class object."""
        return CylinderBase(ag, pos_arg=42)

    def test_origi_init(self, ag):
        """Test origin init."""
        cylinder_class_obj = CylinderClass(ag, pos_arg=42, opt_arg="bar")
        assert cylinder_class_obj.pos_arg == 42
        assert cylinder_class_obj.opt_arg == "bar"

    def test_wrong_rlims(self, ag):
        """Test wrong r limits."""
        with pytest.raises(ValueError, match="can not be smaller"):
            cylinder_class_obj = CylinderClass(ag, pos_arg=42, rmax=1, rmin=2)
            cylinder_class_obj._prepare()

    @pytest.mark.parametrize("bin_width", (0, -0.5, "x"))
    def test_wrong_bin_width(self, ag, bin_width):
        """Test bin_width error."""
        with pytest.raises(ValueError, match=r"Binwidth must be a.* number."):
            cylinder_class_obj = CylinderClass(ag, pos_arg=42, bin_width=bin_width)
            cylinder_class_obj._prepare()

    @pytest.mark.parametrize("bin_width", (1, 7.75, 125))
    @pytest.mark.parametrize("dim", (0, 1, 2))
    def test_bin_width(self, ag, dim, bin_width):
        """Test bin_width."""
        cylinder_class_obj = CylinderClass(
            ag, pos_arg=42, dim=dim, bin_width=bin_width
        ).run()

        assert cylinder_class_obj.n_bins == int(
            np.ceil(cylinder_class_obj.means.R / bin_width)
        )
        assert (
            cylinder_class_obj.means.bin_width
            == cylinder_class_obj.means.R / cylinder_class_obj.n_bins
        )

    def bindwidth_neg(self, ag):
        """Raise error for negative bin_width."""
        with pytest.raises(ValueError, match="positive number"):
            CylinderClass(ag, pos_arg=42, bin_width=-1)._preepare()

    def bindwidth_nan(self, ag):
        """Raise error for bin_width not a number."""
        with pytest.raises(ValueError, match="must be a number"):
            CylinderClass(ag, pos_arg=42, bin_width="foo")._preepare()

    def test_n_bins(self, ag, caplog):
        """Test n bins."""
        cylinder_class_obj = CylinderClass(ag, pos_arg=42)
        caplog.set_level(logging.INFO)
        cylinder_class_obj.run()

        assert cylinder_class_obj.n_bins == 10
        assert "Using 10 bins." in [rec.message for rec in caplog.records]

    def test_rmin_default(self, ag):
        """Test default rmin."""
        bin_width = 2
        cylinder_class_obj = CylinderClass(ag, pos_arg=42, bin_width=bin_width)
        cylinder_class_obj._prepare()

        assert cylinder_class_obj.rmin == 0
        assert cylinder_class_obj.n_bins == 10 / bin_width

    def test_rmin(self, ag):
        """Test rmin."""
        bin_width = 2
        cylinder_class_obj = CylinderClass(ag, pos_arg=42, rmin=2, bin_width=bin_width)
        cylinder_class_obj._prepare()

        assert cylinder_class_obj.rmin == 2
        assert cylinder_class_obj.n_bins == (10 - 2) / bin_width

    def rmin_too_small(self, ag):
        """Test error raise for too small rmin."""
        with pytest.raises(ValueError, match="Only values for rmin largere 0"):
            CylinderClass(ag, pos_arg=42, rmin=-1)._prepare()

    def test_rmax(self, ag):
        """Test rmax."""
        bin_width = 2
        cylinder_class_obj = CylinderClass(ag, rmax=6, pos_arg=42, bin_width=bin_width)
        cylinder_class_obj._prepare()

        assert cylinder_class_obj.rmax == 6
        assert cylinder_class_obj.n_bins == 6 / bin_width

    def test_rmax_default(self, ag):
        """Test rmax default value."""
        cylinder_class_obj = CylinderClass(ag, pos_arg=42)
        cylinder_class_obj._prepare()
        assert cylinder_class_obj.rmax == ag.universe.dimensions[:2].min() / 2

    def test_rmax_odims(self, ag):
        """Test rmax dim."""
        cylinder_class_obj = CylinderClass(ag, zmax=None, pos_arg=42)
        cylinder_class_obj._prepare()

        assert cylinder_class_obj.rmax == ag.universe.dimensions[:2].min() / 2

    def test_rmin_rmax(self, ag):
        """Test rmin rmax."""
        bin_width = 2
        cylinder_class_obj = CylinderClass(
            ag, rmin=10, rmax=20, pos_arg=42, bin_width=bin_width
        )
        cylinder_class_obj._prepare()

        assert cylinder_class_obj.n_bins == (20 - 10) / bin_width

    def test_rmin_rmax_error(self, ag):
        """Test rmax dim."""
        cylinder_class_obj = CylinderClass(ag, rmin=1, rmax=0, pos_arg=42)
        with pytest.raises(ValueError, match="can not be smaller than"):
            cylinder_class_obj._prepare()

    @pytest.mark.parametrize("dim", (0, 1, 2))
    @pytest.mark.parametrize("bin_width_in", (0.1, 0.775))
    def test_bin_pos(self, ag, dim, bin_width_in):
        """Test bin positions."""
        cylinder_class_obj = CylinderClass(
            ag, dim=dim, bin_width=bin_width_in, pos_arg=42
        )
        cylinder_class_obj.run(stop=5)
        rmax = ag.universe.dimensions[cylinder_class_obj.odims].min() / 2

        bin_pos = np.linspace(0, rmax, cylinder_class_obj.n_bins, endpoint=False)
        bin_pos += cylinder_class_obj.means.bin_width / 2

        assert_allclose(cylinder_class_obj.results.bin_pos, bin_pos)

    def test_bin_edges(self, ag):
        """Test edges of the bins."""
        cylinder_class_obj = CylinderClass(ag, bin_width=1, rmax=3, pos_arg=42)
        cylinder_class_obj.run(stop=5)
        bin_edges = [0, 1, 2, 3]
        assert_allclose(cylinder_class_obj.means.bin_edges, bin_edges)

    def test_bin_area_and_volume(self, ag):
        """Test correct area and volume of each bin."""
        cylinder_class_obj = CylinderClass(ag, bin_width=1, rmax=3, pos_arg=42)
        cylinder_class_obj.run(stop=5)
        bin_area = np.pi * np.array([1**2 - 0**2, 2**2 - 1**2, 3**2 - 2**2])

        assert_equal(cylinder_class_obj.means.bin_area, bin_area)

        bin_volume = bin_area * ag.universe.dimensions[2]
        assert_equal(cylinder_class_obj.means.bin_volume, bin_volume)

    @pytest.mark.parametrize("dim", [0, 1, 2])
    def test_R(self, ag, dim):
        """Test radius of the cylinder."""
        cylinder_class_obj = CylinderClass(ag, dim=dim, pos_arg=42)
        cylinder_class_obj.run(stop=5)

        R = ag.universe.dimensions[cylinder_class_obj.odims].min() / 2
        assert cylinder_class_obj.means.R == R

    @pytest.mark.parametrize("dim", (0, 1, 2))
    def test_compute_lab_frame_cylinder_default(self, ag, dim):
        """Test lab frame values with default values."""
        cls = CylinderClass(ag, pos_arg=42, dim=dim)
        cls._compute_lab_frame_cylinder()

        assert_equal(
            cls.pos_cyl,
            transform_cylinder(ag.positions, origin=cls.box_center, dim=cls.dim),
        )
        assert cls.rmax == ag.universe.dimensions[cls.odims].min() / 2

    @pytest.mark.parametrize("rmax", (1, 2, 4.5))
    def test_compute_lab_frame_cylinder(self, ag, rmax):
        """Test lab frame values with explicit values."""
        p_obj = CylinderClass(ag, **{"pos_arg": 42, "rmax": rmax})
        p_obj._compute_lab_frame_cylinder()

        assert p_obj.rmax == rmax

    def test_transformed_positions(self, ag):
        """Test that all universe coordinates are transformed."""
        cls = CylinderClass(ag, pos_arg=42)

        u = ag.universe

        cls._prepare()
        assert_equal(
            cls.pos_cyl,
            transform_cylinder(u.atoms.positions, origin=cls.box_center, dim=cls.dim),
        )

        # Test if _single_frame updates the positions.
        cls._obs = Results()
        u.trajectory[10]

        cls._single_frame()
        assert_equal(
            cls.pos_cyl,
            transform_cylinder(u.atoms.positions, origin=cls.box_center, dim=cls.dim),
        )


class TestCylinderBaseChilds:
    """Tests for the CylindereBase child classes."""

    @pytest.fixture()
    def ag_single_frame(self):
        """Import MDA univers."""
        u = mda.Universe(WATER_TPR, WATER_GRO)
        return u.atoms

    members = []
    for _, member in inspect.getmembers(maicos):
        if (
            inspect.isclass(member)
            and issubclass(member, CylinderBase)
            and member is not CylinderBase
        ):
            members.append(member)

    @pytest.mark.parametrize("Member", members)
    def test_check_attr_change(self, Member, ag_single_frame):
        """Test check attr change."""
        params = dict(dim=2, zmin=None, zmax=None, rmin=0, rmax=None, bin_width=1)
        ana_obj = Member(ag_single_frame, **params).run()
        pb_obj = CylinderBase(ag_single_frame, **params).run()

        assert_equal(ana_obj.results.bin_pos, pb_obj.results.bin_pos)
        assert_equal(ana_obj.n_bins, pb_obj.n_bins)

        assert ana_obj.zmin == pb_obj.zmin
        assert ana_obj.zmax == pb_obj.zmax

        assert ana_obj.rmin == pb_obj.rmin
        assert ana_obj.rmax == pb_obj.rmax


class TestProfileCylinderBase:
    """Test the ProfileCylinderBase class."""

    @pytest.fixture()
    def u(self):
        """Generate a universe containing 125 atoms at random positions.

        Atom positions are drawn inside a cylinder from a uniform distribution
        in a radial distance from 0 A to 1 A from the center of the cell and
        from 0 A to 1 A in the z direction.
        Masses and charges of all particles are 1.
        The cell dimension is 2A x 2A x 2A.
        """
        n_atoms = 125
        n_frames = 4_000

        universe = mda.Universe.empty(
            n_atoms=n_atoms,
            n_residues=n_atoms,
            n_segments=n_atoms,
            atom_resindex=np.arange(n_atoms),
            residue_segindex=np.arange(n_atoms),
        )

        for attr in ["charges", "masses"]:
            universe.add_TopologyAttr(attr, values=np.ones(n_atoms))

        universe.add_TopologyAttr("resids", np.arange(n_atoms))

        rng = np.random.default_rng(1634123)

        coords = rng.random((n_frames, n_atoms, 3))

        # Transform cylinder into cartesian coordinates and shift in cell center
        r_coords = 0.5 * np.sqrt(np.copy(coords[:, :, 0]))
        phi_coords = 2 * np.pi * np.copy(coords[:, :, 1])

        coords[:, :, 0] = r_coords * np.cos(phi_coords) + 1
        coords[:, :, 1] = r_coords * np.sin(phi_coords) + 1

        universe.trajectory = get_reader_for(coords)(
            coords, order="fac", n_atoms=n_atoms
        )

        for ts in universe.trajectory:
            ts.dimensions = np.array([2, 2, 2, 90, 90, 90])

        return universe

    @pytest.fixture()
    def u_dimers(self):
        """Generate a universe containing two dimers with a dipole moment.

        The first atom of each dimer is in the center of the cylinder the
        second is further out on the x axis.
        """
        universe = mda.Universe.empty(
            n_atoms=4,
            n_residues=2,
            n_segments=2,
            atom_resindex=[0, 0, 1, 1],
            residue_segindex=[0, 1],
        )

        universe.add_TopologyAttr("masses", [1, 0, 1, 0])
        universe.add_TopologyAttr("charges", [1, -1, -1, 1])
        universe.add_TopologyAttr("bonds", ((0, 1), (2, 3)))
        universe.add_TopologyAttr("resids", [0, 1])
        universe.add_TopologyAttr("molnums", [0, 1])

        positions = np.array([[2, 2, 0], [5, 2, 0], [2, 2, 0], [1, 2, 0]])

        universe.trajectory = get_reader_for(positions)(
            positions, order="fac", n_atoms=4
        )

        for ts in universe.trajectory:
            ts.dimensions = np.array([4, 4, 1, 90, 90, 90])

        return universe

    def weights(self, ag, grouping, scale=1):
        """Scalable weights for profile calculations."""
        return scale * density_weights(ag, grouping, dens="number")

    @pytest.fixture()
    def params(self, u):
        """Fixture for CylinderBase class atributes."""
        p = dict(
            weighting_function=self.weights,
            atomgroups=u.atoms,
            normalization="number",
            dim=2,
            zmin=None,
            zmax=None,
            rmin=0,
            rmax=None,
            bin_width=0.1,
            refgroup=None,
            grouping="atoms",
            unwrap=False,
            bin_method="com",
            concfreq=0,
            output="profile.dat",
        )
        return p

    @pytest.mark.parametrize("normalization", ["volume", "number", "None"])
    def test_profile(self, u, normalization, params):
        """Test profile with different normalizations."""
        params.update(normalization=normalization)
        profile = ProfileCylinderBase(**params).run()

        if normalization == "volume":
            # Divide by 2 since only half of the box is filled with atoms.
            profile_vals = u.atoms.n_atoms / (np.pi * 0.5**2 * 2)
        elif normalization == "number":
            profile_vals = 1
        else:  # == None
            # Divide by 2: Half of the box is filled with atoms.
            profile_vals = 2 * u.atoms.n_atoms * profile.n_bins / 100

            # Particle number increase linear...
            profile_vals *= 0.4 * np.arange(5)
            profile_vals += 5

        actual = profile.results.profile.flatten()
        desired = np.zeros(profile.n_bins)
        desired[:5] = profile_vals

        if normalization == "number":
            desired[5:] = np.nan

        assert_allclose(actual, desired, rtol=1e-2)

        # TODO: Add test for error and standard deviation.
        # Needs analytical estimaton of the error

    @pytest.mark.parametrize(
        "grouping", ["atoms", "segments", "residues", "molecules", "fragments"]
    )
    def test_grouping(self, u_dimers, grouping, params):
        """Test profile grouping."""
        params.update(
            atomgroups=u_dimers.atoms,
            bin_width=1,
            normalization="None",
            grouping=grouping,
        )
        profile = ProfileCylinderBase(**params).run()
        actual = profile.results.profile.flatten()

        if grouping == "atoms":
            desired = [2, 2]
        else:
            desired = [2, 0]

        assert_equal(actual, desired)

    @pytest.mark.parametrize(
        "bin_method, desired", [("cog", [1, 1]), ("com", [2, 0]), ("coc", [1, 1])]
    )
    def test_bin_method(self, u_dimers, bin_method, desired, params):
        """Test different bin methods."""
        params.update(
            atomgroups=u_dimers.atoms,
            bin_width=1,
            bin_method=bin_method,
            normalization="none",
            grouping="molecules",
        )
        profile = ProfileCylinderBase(**params).run()
        actual = profile.results.profile.flatten()
        assert_equal(actual, desired)

    def test_histogram(self, params):
        """Test the histogram method."""
        p = ProfileCylinderBase(**params)
        p._prepare()
        hist = p._compute_histogram(
            np.linspace(3 * [p.zmin], 3 * [p.zmax], p.n_bins), weights=None
        )

        assert_equal(hist, [0, 2, 0, 0, 2, 0, 0, 2, 0, 0])

    def test_histogram_weight(self, params):
        """Test the histogram method with weights."""
        p = ProfileCylinderBase(**params)
        p._prepare()
        hist = p._compute_histogram(
            np.linspace(3 * [p.zmin], 3 * [p.zmax], p.n_bins),
            weights=5 * np.ones(p.n_bins),
        )

        assert_equal(hist, [0, 10, 0, 0, 10, 0, 0, 10, 0, 0])

    def test_correlation_bin(self, params):
        """Test that the 0th bin is taken for the analysis."""
        profile = ProfileCylinderBase(**params).run(stop=1)
        selected_bin = profile._single_frame()
        assert selected_bin == profile._obs.profile[0, 0]

    @pytest.mark.parametrize("dimension", [0, 1, 2])
    def test_range_warning(self, u_dimers, params, caplog, dimension):
        """Test warning if rmax is larger than the smallest box vector in odims."""
        warning = "`rmax` is bigger than half the smallest box vector"
        odims = np.roll(np.arange(3), -dimension)[1:]

        params = dict(
            dim=dimension, zmin=None, zmax=None, rmin=0, rmax=None, bin_width=1
        )
        params["atomgroups"] = u_dimers.atoms
        params["rmax"] = 1.1 * u_dimers.dimensions[odims].min() / 2
        ana_obj = CylinderBase(**params)
        ana_obj._compute_lab_frame_cylinder()
        assert warning in "".join([rec.message for rec in caplog.records])
