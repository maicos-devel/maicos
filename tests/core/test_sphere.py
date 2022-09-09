"""Tests for the base modules."""
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
#
# Copyright (c) 2022 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the GNU Public Licence, v3 or any higher version
# SPDX-License-Identifier: GPL-3.0-or-later

import inspect
import logging
import os
import sys

import MDAnalysis as mda
import numpy as np
import pytest
from MDAnalysis.analysis.base import Results
from MDAnalysis.core._get_readers import get_reader_for
from numpy.testing import assert_allclose, assert_equal

import maicos
from maicos.core import ProfileSphereBase, SphereBase
from maicos.lib.weights import density_weights


sys.path.append("..")

from data import AIRWATER_TPR, AIRWATER_TRR, WATER_GRO, WATER_TPR  # noqa: E402


class SphereClass(SphereBase):
    """Tests for the Planar Base class."""

    def __init__(self,
                 atomgroups,
                 pos_arg,
                 opt_arg="foo",
                 rmin=0,
                 rmax=None,
                 binwidth=1,
                 **kwargs):
        super(SphereClass, self).__init__(atomgroups=atomgroups,
                                          rmin=rmin,
                                          rmax=rmax,
                                          binwidth=binwidth,
                                          multi_group=True,
                                          **kwargs)
        self.pos_arg = pos_arg
        self.opt_arg = opt_arg

    def _prepare(self):
        super(SphereClass, self)._prepare()
        self.prepared = True

    def _single_frame(self):
        super(SphereClass, self)._single_frame()
        self.ran = True

    def _conclude(self):
        super(SphereClass, self)._conclude()
        self.calculated_results = True


class TestSphereBase(object):
    """Tests for the TestPlanarBase class."""

    @pytest.fixture()
    def ag(self):
        """Import MDA universe."""
        u = mda.Universe(AIRWATER_TPR, AIRWATER_TRR, in_memory=True)
        return u.atoms

    @pytest.fixture()
    def sphere_class_obj(self, ag):
        """Planar class object."""
        return SphereClass(ag, pos_arg=42)

    def test_origi_init(self, ag):
        """Test origin init."""
        sphere_class_obj = SphereClass(ag, pos_arg=42, opt_arg="bar")
        assert sphere_class_obj.pos_arg == 42
        assert sphere_class_obj.opt_arg == "bar"

    def test_wrong_rlims(self, ag):
        """Test wrong r limits."""
        with pytest.raises(ValueError,
                           match='can not be smaller'):
            sphere_class_obj = SphereClass(ag, pos_arg=42, rmax=1, rmin=2)
            sphere_class_obj._prepare()

    @pytest.mark.parametrize('binwidth', (0, -0.5, 'x'))
    def test_wrong_binwidth(self, ag, binwidth):
        """Test binwidth error."""
        with pytest.raises(ValueError,
                           match=r'Binwidth must be a.* number.'):
            sphere_class_obj = SphereClass(ag, pos_arg=42,
                                           binwidth=binwidth)
            sphere_class_obj._prepare()

    @pytest.mark.parametrize('binwidth', (1, 7.75, 125))
    def test_binwidth(self, ag, binwidth):
        """Test binwidth."""
        sphere_class_obj = SphereClass(ag,
                                       pos_arg=42,
                                       binwidth=binwidth)
        sphere_class_obj._frame_index = 0

        sphere_class_obj._prepare()
        sphere_class_obj.means = Results()
        sphere_class_obj.means.R = ag.universe.dimensions.min()
        sphere_class_obj.means.binvolume = 0

        sphere_class_obj.means.R /= 2
        sphere_class_obj._index = 1
        sphere_class_obj._conclude()

        assert sphere_class_obj.n_bins == \
            int(np.ceil(sphere_class_obj.R / binwidth))
        assert sphere_class_obj.binwidth \
            == sphere_class_obj.R / sphere_class_obj.n_bins

    def bindwidth_neg(self, ag):
        """Raise error for negative binwidth."""
        with pytest.raises(ValueError, match="positive number"):
            SphereClass(ag, pos_arg=42, binwidth=-1)._preepare()

    def bindwidth_nan(self, ag):
        """Raise error for binwidth not a number."""
        with pytest.raises(ValueError, match="must be a number"):
            SphereClass(ag, pos_arg=42, binwidth="foo")._preepare()

    def test_n_bins(self, ag, caplog):
        """Test n bins."""
        sphere_class_obj = SphereClass(ag, pos_arg=42)
        caplog.set_level(logging.INFO)
        sphere_class_obj._prepare()

        assert sphere_class_obj.n_bins == 10
        assert "Using 10 bins" in [rec.message for rec in caplog.records]

    def test_rmin_default(self, ag):
        """Test default rmin."""
        binwidth = 2
        sphere_class_obj = SphereClass(ag, pos_arg=42, binwidth=binwidth)
        sphere_class_obj._prepare()

        assert sphere_class_obj.rmin == 0
        assert sphere_class_obj.n_bins == 10 / binwidth

    def test_rmin(self, ag):
        """Test rmin."""
        binwidth = 2
        sphere_class_obj = SphereClass(ag,
                                       pos_arg=42,
                                       rmin=2,
                                       binwidth=binwidth)
        sphere_class_obj._prepare()

        assert sphere_class_obj.rmin == 2
        assert sphere_class_obj.n_bins == (10 - 2) / binwidth

    def rmin_too_small(self, ag):
        """Test error raise for too small rmin."""
        with pytest.raises(ValueError, match="Only values for rmin largere 0"):
            SphereClass(ag, pos_arg=42, rmin=-1)._prepare()

    def test_rmax(self, ag):
        """Test rmax."""
        binwidth = 2
        sphere_class_obj = SphereClass(ag,
                                       rmax=6,
                                       pos_arg=42,
                                       binwidth=binwidth)
        sphere_class_obj._prepare()

        assert sphere_class_obj.rmax == 6
        assert sphere_class_obj.n_bins == 6 / binwidth

    def test_rmin_rmax(self, ag):
        """Test rmin rmax."""
        binwidth = 2
        sphere_class_obj = SphereClass(ag,
                                       rmin=10,
                                       rmax=20,
                                       pos_arg=42,
                                       binwidth=binwidth)
        sphere_class_obj._prepare()

        assert sphere_class_obj.n_bins == (20 - 10) / binwidth

    def test_rmin_rmax_error(self, ag):
        """Test rmax."""
        sphere_class_obj = SphereClass(ag, rmin=1, rmax=0, pos_arg=42)
        with pytest.raises(ValueError, match="can not be smaller than"):
            sphere_class_obj._prepare()

    @pytest.mark.parametrize('binwidth_in', (0.1, .775))
    def test_results_r(self, ag, binwidth_in):
        """Test results r."""
        sphere_class_obj = SphereClass(ag, binwidth=binwidth_in, pos_arg=42)
        sphere_class_obj.run(stop=5)
        rmax = ag.universe.dimensions.min() / 2
        n_bins = int(np.ceil(rmax / binwidth_in))
        binwidth = rmax / n_bins
        assert sphere_class_obj.binwidth == binwidth
        assert_allclose(sphere_class_obj.results.r,
                        np.linspace(0, rmax, n_bins) + binwidth / 2)

    def test_binvolume(self, ag):
        """Test correct volume of ach bin."""
        rmax = 3
        binwidth = 1

        sphere_class_obj = SphereClass(ag,
                                       binwidth=binwidth,
                                       rmax=rmax,
                                       pos_arg=42)
        sphere_class_obj.run(stop=5)
        binvolume = 4 * np.pi * np.array([1**3 - 0**3,
                                         2**3 - 1**3,
                                         3**3 - 2**3]) / 3

        assert_allclose(sphere_class_obj.binvolume, binvolume)

    def test_R(self, ag):
        """Test radius of the sphere."""
        sphere_class_obj = SphereClass(ag, pos_arg=42)
        sphere_class_obj.run(stop=5)

        R = ag.universe.dimensions.min() / 2
        assert sphere_class_obj.R == R

    def test_compute_lab_frame_sphere_default(self, ag):
        """Test lab frame values with default values."""
        cls = SphereClass(ag, pos_arg=42)
        cls._compute_lab_frame_sphere()

        assert_equal(cls.pos_sph, cls.transform_positions(ag.positions))
        assert cls.rmax == ag.universe.dimensions.min() / 2

    @pytest.mark.parametrize("rmax", (1, 2, 4.5))
    def test_compute_lab_frame_sphere(self, ag, rmax):
        """Test lab frame values with explicit values."""
        p_obj = SphereClass(ag, **{"pos_arg": 42, "rmax": rmax})
        p_obj._compute_lab_frame_sphere()

        assert p_obj.rmax == rmax

    def test_transform_positions(self, ag):
        """Test spherical transformation of positions."""
        u = ag.universe

        # Manipulate universe
        u.dimensions = np.array([2, 2, 2, 90, 90, 90])

        sel = u.atoms[:4]

        # Put one atom at each quadrant on different z positions
        sel[0].position = np.array([0, 0, 1])
        sel[1].position = np.array([0, 2, 1])
        sel[2].position = np.array([2, 2, 1])
        sel[3].position = np.array([2, 0, 1])

        cls = SphereClass(sel, pos_arg=42)
        cls._prepare()
        pos_sph = cls.transform_positions(sel.positions)

        assert_allclose(pos_sph[:, 0], np.sqrt(2))

        # phi component
        assert_allclose(pos_sph[0, 1], np.arctan(1) - np.pi)
        assert_allclose(pos_sph[1, 1], np.arctan(-1) + np.pi)
        assert_allclose(pos_sph[2, 1], np.arctan(1))
        assert_allclose(pos_sph[3, 1], np.arctan(-1))

        # theta component
        assert_allclose(pos_sph[:, 2], np.arccos(0))

    def test_transformed_positions(self, ag):
        """Test that all universe coordinates are transformed."""
        cls = SphereClass(ag, pos_arg=42)

        u = ag.universe

        cls._prepare()
        assert_equal(cls.pos_sph,
                     cls.transform_positions(u.atoms.positions))

        # Test if _single_frame updates the positions.
        cls._obs = Results()
        u.trajectory[10]

        cls._single_frame()
        assert_equal(cls.pos_sph,
                     cls.transform_positions(u.atoms.positions))


class TestSphereBaseChilds:
    """Tests for the CylindereBaseChilds class."""

    @pytest.fixture()
    def ag_single_frame(self):
        """Import MDA univers."""
        u = mda.Universe(WATER_TPR, WATER_GRO)
        return u.atoms

    members = []
    for _, member in inspect.getmembers(maicos):
        if inspect.isclass(member) and issubclass(member, SphereBase) \
                and member is not SphereBase:
            members.append(member)

    @pytest.mark.parametrize("Member", members)
    def test_check_attr_change(self, Member, ag_single_frame):
        """Test check attr change."""
        params = dict(rmin=0,
                      rmax=None,
                      binwidth=1)
        ana_obj = Member(ag_single_frame, **params).run()
        pb_obj = SphereBase(ag_single_frame, **params).run()

        assert_equal(ana_obj.results.r, pb_obj.results.r)
        assert_equal(ana_obj.n_bins, pb_obj.n_bins)
        assert_equal(ana_obj.binwidth, pb_obj.binwidth)
        assert_equal(ana_obj.binvolume, pb_obj.binvolume)

        assert ana_obj.rmin == pb_obj.rmin
        assert ana_obj.rmax == pb_obj.rmax
        assert ana_obj.R == pb_obj.R


class TestProfileSphereBase:
    """Test the ProfileSphereBase class."""

    @pytest.fixture()
    def u(self):
        """Generate a universe containing 125 atoms at random positions.

        Atom positions are drawn inside a sphere from a uniform distribution
        in a radial distance from 0 A to 0.5 A from the center of the cell.
        Masses and charges of all particles are 1.
        The cell dimension is 2A x 2A x 2A.
        """
        n_atoms = 125
        n_frames = 4_000

        universe = mda.Universe.empty(n_atoms=n_atoms,
                                      n_residues=n_atoms,
                                      n_segments=n_atoms,
                                      atom_resindex=np.arange(n_atoms),
                                      residue_segindex=np.arange(n_atoms))

        for attr in ["charges", "masses"]:
            universe.add_TopologyAttr(attr, values=np.ones(n_atoms))

        universe.add_TopologyAttr("resids", np.arange(n_atoms))

        rng = np.random.default_rng(1634123)

        coords = rng.random((n_frames, n_atoms, 3))

        # Transform cylinder into cartesian coordinates and shift in cell center
        r_coords = 0.5 * np.cbrt(np.copy(coords[:, :, 0]))
        phi_coords = 2 * np.pi * np.copy(coords[:, :, 1])
        theta_coords = np.pi * np.copy(coords[:, :, 2])

        coords[:, :, 0] = r_coords * np.cos(phi_coords) * np.sin(theta_coords)
        coords[:, :, 1] = r_coords * np.sin(phi_coords) * np.sin(theta_coords)
        coords[:, :, 2] = r_coords * np.cos(theta_coords)

        coords += 1

        universe.trajectory = get_reader_for(coords)(coords,
                                                     order='fac',
                                                     n_atoms=n_atoms)

        for ts in universe.trajectory:
            ts.dimensions = np.array([2, 2, 2, 90, 90, 90])

        return universe

    @pytest.fixture()
    def u_dimers(self):
        """Generate a universe containing two dimers with a dipole moment."""
        universe = mda.Universe.empty(n_atoms=4,
                                      n_residues=2,
                                      n_segments=2,
                                      atom_resindex=[0, 0, 1, 1],
                                      residue_segindex=[0, 1])

        universe.add_TopologyAttr("masses", [1, 0, 0, 1])
        universe.add_TopologyAttr("charges", [1, -1, -1, 1])
        universe.add_TopologyAttr("bonds", ((0, 1), (2, 3)))
        universe.add_TopologyAttr("resids", [0, 1])
        universe.add_TopologyAttr("molnums", [0, 1])

        positions = np.array([[1, 1, 1], [1, 1, -1],
                              [1, 1, 1], [1, 1, 3]])

        universe.trajectory = get_reader_for(positions)(positions,
                                                        order='fac',
                                                        n_atoms=4)

        for ts in universe.trajectory:
            ts.dimensions = np.array([2, 2, 2, 90, 90, 90])

        return universe

    def weights(self, ag, grouping, scale=1):
        """Scalable weights for profile calculations."""
        return scale * density_weights(ag, grouping, dens="number")

    @pytest.fixture()
    def params(self, u):
        """Fixture for CylinderBase class atributes."""
        p = dict(function=self.weights,
                 atomgroups=u.atoms,
                 normalization="number",
                 rmin=0,
                 rmax=None,
                 binwidth=0.1,
                 refgroup=None,
                 grouping="atoms",
                 unwrap=False,
                 binmethod="com",
                 concfreq=0,
                 output="profile.dat")
        return p

    @pytest.mark.parametrize("normalization", ["volume", "number", "None"])
    def test_profile(self, u, normalization, params):
        """Test profile with different normalizations."""
        params.update(normalization=normalization)
        profile = ProfileSphereBase(**params).run()

        if normalization == "volume":
            # Divide by 2 since only half of the box is filled with atoms.
            profile_vals = u.atoms.n_atoms / (4 * np.pi / 3 * 0.5**3)
        elif normalization == "number":
            profile_vals = np.ones(5)
            profile_vals[0] = 0.63  # Only very few particles in first bin.
        else:  # == None
            profile_vals = 6 * np.array([0, 1, 3, 6, 10]) + 1

        actual = profile.results.profile_mean.flatten()
        desired = np.zeros(profile.n_bins)
        desired[:5] = profile_vals

        if normalization == "number":
            desired[5:] = np.nan

        assert_allclose(actual, desired, rtol=1e-2)

        # TODO: Add test for error and standard deviation.
        # Needs analytical estimaton of the error

    @pytest.mark.parametrize("grouping", ["atoms", "segments", "residues",
                                          "molecules", "fragments"])
    def test_grouping(self, u_dimers, grouping, params):
        """Test profile grouping."""
        params.update(atomgroups=u_dimers.atoms,
                      binwidth=1,
                      rmax=2,
                      normalization="None",
                      grouping=grouping)
        profile = ProfileSphereBase(**params).run()
        actual = profile.results.profile_mean.flatten()

        if grouping == "atoms":
            desired = [2, 2]
        else:
            desired = [1, 1]

        assert_equal(actual, desired)

    @pytest.mark.parametrize("binmethod, desired",
                             [("cog", [np.nan, 1]),
                              ("com", [1, 1]),
                              ("coc", [np.nan, 1])])
    def test_binmethod(self, u_dimers, binmethod, desired, params):
        """Test different bin methods."""
        params.update(atomgroups=u_dimers.atoms,
                      binwidth=1,
                      rmax=2,
                      binmethod=binmethod,
                      grouping="molecules")
        profile = ProfileSphereBase(**params).run()
        actual = profile.results.profile_mean.flatten()
        assert_equal(actual, desired)

    @pytest.mark.parametrize("unwrap, desired",
                             [(False, [1, 1]), (True, [2, 0])])
    def test_unwrap(self, u_dimers, unwrap, desired, params):
        """Test making molecules whole."""
        params.update(atomgroups=u_dimers.atoms,
                      binwidth=1,
                      rmax=2,
                      unwrap=unwrap,
                      binmethod='com',
                      normalization="none",
                      grouping="molecules")

        profile = ProfileSphereBase(**params).run()
        actual = profile.results.profile_mean.flatten()
        assert_equal(actual, desired)

    def test_wrong_normalization(self, params):
        """Test profile for a non existing normalization."""
        params.update(normalization="foo")

        with pytest.raises(ValueError, match="not supported"):
            ProfileSphereBase(**params).run()

    def test_wrong_grouping(self, params):
        """Test grouping for a non existing type."""
        params.update(grouping="foo")

        with pytest.raises(ValueError, match="is not a valid option for"):
            ProfileSphereBase(**params).run()

    def test_wrong_binmethod(self, params):
        """Test error raise for a non existing binmethod."""
        params.update(binmethod="foo")

        with pytest.raises(ValueError, match="is an unknown binning"):
            ProfileSphereBase(**params).run()

    def test_unwrap_atoms(self, params):
        """Test that unwrap is always False for grouping wrt to atoms."""
        params.update(unwrap=True, grouping="atoms")
        profile = ProfileSphereBase(**params).run(stop=1)
        assert profile.unwrap is False

    def test_f_kwargs(self, params):
        """Test an extra keyword argument."""
        scale = 3
        params.update(f_kwargs={"scale": scale})
        profile = ProfileSphereBase(**params).run()

        actual = profile.results.profile_mean.flatten()
        desired = np.nan * np.ones(profile.n_bins)
        desired[:5] = scale
        desired[0] = scale * 0.63

        assert_allclose(actual, desired, rtol=1e-2)

    def test_multigroup(self, u, params):
        """Test analysis for a list of atomgroups."""
        params.update(atomgroups=[u.atoms, u.atoms])
        profile = ProfileSphereBase(**params).run()

        actual = profile.results.profile_mean
        desired = np.nan * np.ones([profile.n_bins, 2])
        desired[:5, :] = 1
        desired[0, :] = 0.63

        assert_allclose(actual, desired, rtol=1e-2)

    def test_output_name(self, params):
        """Test output name of save method."""
        params.update(output="foo.dat")
        profile = ProfileSphereBase(**params).run(stop=1)
        profile.save()
        assert os.path.exists(params["output"])

    def test_output(self, params, tmpdir):
        """Test output."""
        with tmpdir.as_cwd():
            profile = ProfileSphereBase(**params).run(stop=1)
            profile.save()
            res_dens = np.loadtxt(profile.output)

        assert_allclose(profile.results.r,
                        res_dens[:, 0],
                        rtol=2)

        assert_allclose(profile.results.profile_mean[:, 0],
                        res_dens[:, 1],
                        rtol=2)

        assert_allclose(profile.results.profile_err[:, 0],
                        res_dens[:, 2],
                        rtol=2)
