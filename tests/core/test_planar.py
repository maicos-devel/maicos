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
import os
import sys

import MDAnalysis as mda
import numpy as np
import pytest
from MDAnalysis.core._get_readers import get_reader_for
from numpy.testing import assert_allclose, assert_equal

import maicos
from maicos.core import CylinderBase, PlanarBase, ProfilePlanarBase
from maicos.lib.weights import density_weights


sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from data import AIRWATER_TPR, AIRWATER_TRR, WATER_GRO, WATER_TPR  # noqa: E402


class PlanarClass(PlanarBase):
    """Tests for the Planar Base class."""

    def __init__(
        self,
        atomgroups,
        pos_arg,
        opt_arg="foo",
        dim=2,
        zmin=None,
        zmax=None,
        bin_width=1,
        refgroup=None,
        **kwargs,
    ):
        super().__init__(
            atomgroups=atomgroups,
            dim=dim,
            zmin=zmin,
            zmax=zmax,
            bin_width=bin_width,
            refgroup=refgroup,
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


class TestPlanarBase(object):
    """Tests for the TestPlanarBase class."""

    @pytest.fixture()
    def ag(self):
        """Import MDA universe."""
        u = mda.Universe(AIRWATER_TPR, AIRWATER_TRR)
        return u.atoms

    @pytest.fixture()
    def planar_class_obj(self, ag):
        """Planar class object."""
        return PlanarClass(ag, pos_arg=42)

    def test_origi_init(self, ag):
        """Test origin  init."""
        planar_class_obj = PlanarClass(ag, pos_arg=42, opt_arg="bar")
        assert planar_class_obj.pos_arg == 42
        assert planar_class_obj.opt_arg == "bar"

    @pytest.mark.parametrize("dim", (0, 1, 2))
    def test_dim(self, ag, dim):
        """Test dim."""
        planar_class_obj = PlanarClass(ag, pos_arg=42, dim=dim)
        assert planar_class_obj.dim == dim

    @pytest.mark.parametrize("dim, odims", [(0, (1, 2)), (1, (2, 0)), (2, (0, 1))])
    def test_odim(self, ag, dim, odims):
        """Test odims."""
        planar_class_obj = PlanarClass(ag, pos_arg=42, dim=dim)
        planar_class_obj._prepare()
        assert_equal(planar_class_obj.odims, odims)

    @pytest.mark.parametrize("dim", (3, "x", -1))
    def test_wrong_dim(self, ag, dim):
        """Test dim."""
        with pytest.raises(ValueError, match="Dimension can only be x=0, y=1 or z=2."):
            planar_class_obj = PlanarClass(ag, pos_arg=42, dim=dim)
            planar_class_obj._prepare()

    def test_wrong_zlims(self, ag):
        """Test wrong z limits."""
        with pytest.raises(ValueError, match="can not be smaller"):
            planar_class_obj = PlanarClass(ag, pos_arg=42, zmax=-1, zmin=0)
            planar_class_obj._prepare()

    def test_box_center(self, ag):
        """Test the center of the simulation box."""
        planar_class_obj = PlanarClass(ag, pos_arg=42)
        assert_equal(planar_class_obj.box_center, ag.universe.dimensions[:3] / 2)

    @pytest.mark.parametrize("dim", (0, 1, 2))
    def test_compute_lab_frame_planar_default(self, ag, dim):
        """Test lab frame values with default values."""
        planar_class_obj = PlanarClass(ag, pos_arg=42, dim=dim)
        planar_class_obj._compute_lab_frame_planar()

        assert planar_class_obj.zmin == 0
        assert planar_class_obj.zmax == ag.universe.dimensions[dim]

    @pytest.mark.parametrize("lim", ("zmin", "zmax"))
    @pytest.mark.parametrize("pos", (-2, -1, 0, 1, 2, 4.5))
    @pytest.mark.parametrize("dim", (0, 1, 2))
    def test_compute_lab_frame_planar(self, ag, lim, pos, dim):
        """Test lab frame values with explicit values."""
        p_obj = PlanarClass(ag, **{"pos_arg": 42, "dim": dim, lim: pos})
        p_obj._compute_lab_frame_planar()

        assert getattr(p_obj, lim) == p_obj.box_center[dim] + pos

    @pytest.mark.parametrize("bin_width", (0, -0.5, "x"))
    def test_wrong_bin_width(self, ag, bin_width):
        """Test bin_width."""
        with pytest.raises(ValueError, match=r"Binwidth must be a.* number."):
            planar_class_obj = PlanarClass(ag, pos_arg=42, bin_width=bin_width)
            planar_class_obj._prepare()

    @pytest.mark.parametrize("bin_width", (1, 7.75, 125))
    @pytest.mark.parametrize("dim", (0, 1, 2))
    def test_bin_width(self, ag, dim, bin_width):
        """Test bin_width."""
        planar_class_obj = PlanarClass(
            ag, pos_arg=42, dim=dim, bin_width=bin_width
        ).run()

        assert planar_class_obj.n_bins == int(
            np.ceil(planar_class_obj.means.L / bin_width)
        )
        assert_allclose(
            planar_class_obj.means.bin_width,
            planar_class_obj.means.L / planar_class_obj.n_bins,
        )

    @pytest.mark.parametrize("dim", (0, 1, 2))
    def test_bin_edges(self, ag, dim):
        """Test edges of the bins."""
        planar_class_obj = PlanarClass(ag, pos_arg=42, dim=dim).run(stop=1)

        bin_edges = np.linspace(
            0, ag.universe.dimensions[dim], planar_class_obj.n_bins + 1, endpoint=True
        )

        assert_allclose(planar_class_obj.means.bin_edges, bin_edges)

    def test_bin_area(self, ag):
        """Test area and volume of the bins."""
        planar_class_obj = PlanarClass(
            ag, bin_width=0.5, zmin=0, zmax=3, pos_arg=42
        ).run(stop=1)

        bin_area = np.ones(6) * np.prod(ag.universe.dimensions[:2])
        assert_allclose(planar_class_obj.means.bin_area, bin_area)

        bin_volume = bin_area * 0.5
        assert_allclose(planar_class_obj.means.bin_volume, bin_volume)

    def bin_width_neg(self, ag):
        """Raise error for negative bin_width."""
        with pytest.raises(ValueError, match="positive number"):
            PlanarClass(ag, pos_arg=42, bin_width=-1)._preepare()

    def bin_width_nan(self, ag):
        """Raise error for bin_width not a number."""
        with pytest.raises(ValueError, match="must be a number"):
            PlanarClass(ag, pos_arg=42, bin_width="foo")._preepare()

    def test_n_bins(self, planar_class_obj, caplog):
        """Test n bins."""
        planar_class_obj._verbose = True
        caplog.set_level(logging.INFO)
        planar_class_obj.run()

        assert planar_class_obj.n_bins == 60
        assert "Using 60 bins." in [rec.message for rec in caplog.records]

    def test_zmin_default(self, ag):
        """Test default zmin."""
        bin_width = 2
        planar_class_obj = PlanarClass(ag, pos_arg=42, bin_width=bin_width)
        planar_class_obj._prepare()

        assert planar_class_obj.zmin == 0
        assert planar_class_obj.n_bins == 60 / bin_width

    def test_zmin(self, ag):
        """Test zmin."""
        bin_width = 2
        planar_class_obj = PlanarClass(ag, pos_arg=42, zmin=-20, bin_width=bin_width)
        planar_class_obj._prepare()

        assert planar_class_obj.zmin == 30 - 20
        assert planar_class_obj.n_bins == (60 - 10) / bin_width

    def test_zmax(self, ag):
        """Test zmax."""
        bin_width = 2
        planar_class_obj = PlanarClass(ag, zmax=20, pos_arg=42, bin_width=bin_width)
        planar_class_obj._prepare()

        assert planar_class_obj.zmax == 30 + 20
        assert planar_class_obj.n_bins == (30 + 20) / bin_width

    def test_zmax_dim(self, ag):
        """Test zmax dim."""
        planar_class_obj = PlanarClass(ag, zmax=None, pos_arg=42)
        planar_class_obj._prepare()

        assert planar_class_obj.zmax == ag.universe.dimensions[2]

    def test_L(self, ag):
        """Test cummulative box length L."""
        planar_class_obj = PlanarClass(ag, pos_arg=42)
        planar_class_obj._zmax = None
        planar_class_obj._trajectory = ag.universe.trajectory
        planar_class_obj.run()

        assert planar_class_obj.means.L == ag.universe.dimensions[2]

    def test_zmin_zmax(self, ag):
        """Test zmin zmax."""
        bin_width = 2
        planar_class_obj = PlanarClass(
            ag, zmin=-20, zmax=20, pos_arg=42, bin_width=bin_width
        )
        planar_class_obj._prepare()

        assert planar_class_obj.n_bins == (50 - 10) / bin_width

    def test_zmin_zmax_error(self, ag):
        """Test zmax dim."""
        planar_class_obj = PlanarClass(ag, zmin=0, zmax=-1, pos_arg=42)
        with pytest.raises(ValueError, match="can not be smaller or equal"):
            planar_class_obj._prepare()

    @pytest.mark.parametrize("dim", (0, 1, 2))
    @pytest.mark.parametrize("bin_width_in", (1, 7.75))
    def test_bin_pos(self, ag, dim, bin_width_in):
        """Test bin positions."""
        planar_class_obj = PlanarClass(ag, dim=dim, bin_width=bin_width_in, pos_arg=42)
        planar_class_obj.run(stop=5)

        bin_pos = np.linspace(
            -ag.universe.dimensions[dim] / 2,
            ag.universe.dimensions[dim] / 2,
            planar_class_obj.n_bins,
            endpoint=False,
        )
        bin_pos += planar_class_obj.means.bin_width / 2

        # Small numbers != 0 are problematic for `assert_allclose`...
        bin_pos[np.abs(bin_pos) < 1e-15] = 0

        assert_allclose(planar_class_obj.results.bin_pos, bin_pos)

    @pytest.mark.parametrize("zmin", (-1, 0, 1))
    @pytest.mark.parametrize("zmax", (2, 3, 4))
    def test_results_bin_pos_zmin_zmax(self, ag, zmin, zmax):
        """Test the bin positions for non default zmin and zmax."""
        planar_class_obj = PlanarClass(ag, zmin=zmin, zmax=zmax, pos_arg=42)
        planar_class_obj.run(stop=1)

        bin_pos = np.linspace(zmin, zmax, planar_class_obj.n_bins, endpoint=False)
        bin_pos += planar_class_obj.means.bin_width / 2

        # Small numbers != 0 are problematic for `assert_allclose`...
        bin_pos[np.abs(bin_pos) < 1e-15] = 0

        assert_allclose(planar_class_obj.results.bin_pos, bin_pos)


class TestPlanarBaseChilds:
    """Tests for the PlanarBase child classes."""

    @pytest.fixture()
    def ag_single_frame(self):
        """Import MDA univers."""
        u = mda.Universe(WATER_TPR, WATER_GRO)
        return u.atoms

    members = []
    # Exclude CylinderBase since it is tested individually.
    for _, member in inspect.getmembers(maicos):
        if (
            inspect.isclass(member)
            and issubclass(member, PlanarBase)
            and not issubclass(member, CylinderBase)
            and member not in [PlanarBase, CylinderBase]
        ):
            members.append(member)

    @pytest.mark.parametrize("Member", members)
    def test_check_attr_change(self, Member, ag_single_frame):
        """Test check attr change."""
        params = dict(dim=2, zmin=None, zmax=None, bin_width=1, refgroup=None)
        ana_obj = Member(ag_single_frame, **params).run()
        pb_obj = PlanarBase(ag_single_frame, **params).run()

        assert_equal(ana_obj.results.bin_pos, pb_obj.results.bin_pos)
        assert_equal(ana_obj.n_bins, pb_obj.n_bins)

        assert ana_obj.zmin == pb_obj.zmin
        assert ana_obj.zmax == pb_obj.zmax


class TestProfilePlanarBase:
    """Test the ProfilePlanarBase class."""

    @pytest.fixture()
    def u(self):
        """Generate a universe containing 125 atoms at random positions.

        Atom positions are drawn from a uniform distribution from 0 A to 1 A in
        each direction. Masses and charges of all particles are 1.
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

        # Toggle this comment to get a universe with types
        # universe.add_TopologyAttr("type", ["X"] * n_atoms)
        universe.add_TopologyAttr("resids", np.arange(n_atoms))

        rng = np.random.default_rng(1634123)
        coords = rng.random((n_frames, n_atoms, 3))

        universe.trajectory = get_reader_for(coords)(
            coords, order="fac", n_atoms=n_atoms
        )

        for ts in universe.trajectory:
            ts.dimensions = np.array([2, 2, 2, 90, 90, 90])

        return universe

    @pytest.fixture()
    def u_dimers(self):
        """Generate a universe containing two dimers with a dipole moment."""
        universe = mda.Universe.empty(
            n_atoms=4,
            n_residues=2,
            n_segments=2,
            atom_resindex=[0, 0, 1, 1],
            residue_segindex=[0, 1],
        )

        universe.add_TopologyAttr("masses", [1, 0, 0, 1])
        universe.add_TopologyAttr("charges", [1, -1, -1, 1])
        universe.add_TopologyAttr("bonds", ((0, 1), (2, 3)))
        universe.add_TopologyAttr("resids", [0, 1])
        universe.add_TopologyAttr("molnums", [0, 1])

        positions = np.array([[0, 0, 0], [0, 1, 0], [2, 1, 0], [2, 2, 0]])

        universe.trajectory = get_reader_for(positions)(
            positions, order="fac", n_atoms=4
        )

        for ts in universe.trajectory:
            ts.dimensions = np.array([2, 3, 3, 90, 90, 90])

        return universe

    def weights(self, ag, grouping, scale=1):
        """Scalable weights for profile calculations."""
        return scale * density_weights(ag, grouping, dens="number")

    @pytest.fixture()
    def params(self, u):
        """Fixture for PlanarBase class atributes."""
        p = dict(
            weighting_function=self.weights,
            atomgroups=u.atoms,
            normalization="number",
            dim=2,
            zmin=None,
            zmax=None,
            bin_width=0.1,
            refgroup=None,
            sym=False,
            grouping="atoms",
            unwrap=False,
            bin_method="com",
            concfreq=0,
            output="profile.dat",
            jitter=False,
        )
        return p

    @pytest.mark.parametrize("normalization", ["volume", "number", "None"])
    def test_profile(self, u, normalization, params):
        """Test profile with different normalizations."""
        params.update(normalization=normalization)
        profile = ProfilePlanarBase(**params).run()

        if normalization == "volume":
            # Divide by 2 since only half of the box is filled with atoms.
            profile_vals = u.atoms.n_atoms / (u.trajectory.ts.volume / 2)
        elif normalization == "number":
            profile_vals = 1
        else:  # == None
            # Divide by 2: Half of the box is filled with atoms.
            profile_vals = u.atoms.n_atoms * profile.n_bins / 2 / 100

        actual = profile.results.profile.flatten()
        desired = np.zeros(profile.n_bins)
        desired[:10] = profile_vals

        if normalization == "number":
            desired[10:] = np.nan

        assert_allclose(actual, desired, rtol=1e-2)

        # TODO: Add test for error and standard deviation.
        # Needs analytical estimaton of the error

    def test_sym(self, u, params):
        """Test profile symmetrization."""
        params.update(refgroup=u.atoms, sym=True)
        profile = ProfilePlanarBase(**params).run()

        actual = profile.results.profile.flatten()
        desired = [np.nan, np.nan, np.nan, 1e-04, 0.4, 1, 1, 1, 1, 1]
        desired += desired[::-1]

        assert_allclose(actual, desired, atol=1e-2)

    @pytest.mark.parametrize(
        "grouping", ["atoms", "segments", "residues", "molecules", "fragments"]
    )
    def test_grouping(self, u_dimers, grouping, params):
        """Test profile grouping."""
        params.update(atomgroups=u_dimers.atoms, dim=1, bin_width=1, grouping=grouping)
        profile = ProfilePlanarBase(**params).run()
        actual = profile.results.profile.flatten()

        if grouping == "atoms":
            desired = [1, 1, 1]
        else:
            desired = [1, np.nan, 1]

        assert_equal(actual, desired)

    def test_multigroup(self, u, params):
        """Test analysis for a list of atomgroups."""
        params.update(atomgroups=[u.atoms, u.atoms])
        profile = ProfilePlanarBase(**params).run()

        actual = profile.results.profile
        desired = np.nan * np.ones([profile.n_bins, 2])
        desired[:10, :] = 1

        assert_allclose(actual, desired)

    def test_histogram(self, params):
        """Test the histogram method."""
        p = ProfilePlanarBase(**params)
        p._prepare()
        hist = p._compute_histogram(
            np.linspace(3 * [p.zmin], 3 * [p.zmax], p.n_bins), weights=None
        )

        assert_equal(hist, 1)

    def test_histogram_weight(self, params):
        """Test the histogram method with weights."""
        p = ProfilePlanarBase(**params)
        p._prepare()
        hist = p._compute_histogram(
            np.linspace(3 * [p.zmin], 3 * [p.zmax], p.n_bins),
            weights=5 * np.ones(p.n_bins),
        )

        assert_equal(hist, 5)

    def test_raise_sym_no_refgroup(self, params):
        """Test error raise for symmetrization without provided refgroup."""
        params.update(sym=True)
        with pytest.raises(ValueError, match="For symmetrization the"):
            ProfilePlanarBase(**params).run()

    @pytest.mark.parametrize("n_bins", [1, 2, 3])
    def test_correlation_bin(self, params, u, n_bins):
        """Test that the center bin is taken for the analysis."""
        L = u.dimensions[2]
        params.update(bin_width=L / n_bins)

        profile = ProfilePlanarBase(**params).run(stop=1)

        selected_bin = profile._single_frame()
        assert selected_bin == profile._obs.profile[n_bins // 2, 0]
