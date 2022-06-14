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

import MDAnalysis as mda
import numpy as np
import pytest
from datafiles import (
    AIRWATER_TPR,
    AIRWATER_TRR,
    WATER_GRO,
    WATER_TPR,
    WATER_TRR,
    )
from MDAnalysis.analysis.base import Results
from MDAnalysis.core._get_readers import get_reader_for
from numpy.testing import assert_allclose, assert_equal

import maicos
from maicos.modules import base
from maicos.modules.density import _density_weights


class Test_AnalysisBase(object):
    """Tests for the Analysis base class."""

    @pytest.fixture()
    def ag(self):
        """Import MDA universe."""
        return mda.Universe(WATER_TPR, WATER_TRR).atoms

    def test_AnalysisBase(self, ag):
        """Test AnalysisBase."""
        a = base.AnalysisBase(ag, verbose=True, save_results=True)

        assert a.atomgroup.n_atoms == ag.n_atoms
        assert a._trajectory == ag.universe.trajectory
        assert a._universe == ag.universe
        assert a._verbose is True
        assert type(a.results) is Results
        assert not hasattr(a, "atomgroups")

    def test_multigroups(self, ag):
        """Test multiple groups."""
        a = base.AnalysisBase([ag[:10], ag[10:]], multi_group=True)

        assert a.n_atomgroups == 2
        assert a._universe == ag.universe

    def test_different_universes(self, ag):
        """Test different universes."""
        with pytest.raises(ValueError, match="Atomgroups belong"):
            base.AnalysisBase([ag, mda.Universe(WATER_TPR)], multi_group=True)


class PlanarClass(base.PlanarBase):
    """Tests for the Planar Base class."""

    def __init__(self,
                 atomgroups,
                 pos_arg,
                 opt_arg="foo",
                 dim=2,
                 zmin=0,
                 zmax=None,
                 binwidth=1,
                 comgroup=None,
                 **kwargs):
        super(PlanarClass, self).__init__(atomgroups=atomgroups,
                                          dim=dim,
                                          zmin=zmin,
                                          zmax=zmax,
                                          binwidth=binwidth,
                                          comgroup=comgroup,
                                          multi_group=True,
                                          **kwargs)
        self.pos_arg = pos_arg
        self.opt_arg = opt_arg

    def _prepare(self):
        super(PlanarClass, self)._prepare()
        self.prepared = True

    def _single_frame(self):
        super(PlanarClass, self)._single_frame()
        self.ran = True

    def _conclude(self):
        super(PlanarClass, self)._conclude()
        self.calculated_results = True


class TestPlanarBase(object):
    """Tests for the TestPlanarBase class."""

    @pytest.fixture()
    def ag(self):
        """Import MDA universe."""
        u = mda.Universe(AIRWATER_TPR, AIRWATER_TRR)
        return u.atoms

    @pytest.fixture()
    def empty_ag(self):
        """Define an empty atomgroup."""
        u = mda.Universe.empty(0)
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

    @pytest.mark.parametrize('dim', (0, 1, 2))
    def test_dim(self, ag, dim):
        """Test dim."""
        planar_class_obj = PlanarClass(ag, pos_arg=42, dim=dim)
        assert planar_class_obj.dim == dim

    @pytest.mark.parametrize('dim', (3, 'x', -1))
    def test_wrong_dim(self, ag, dim):
        """Test dim."""
        with pytest.raises(ValueError,
                           match='Dimension can only be x=0, y=1 or z=2.'):
            planar_class_obj = PlanarClass(ag, pos_arg=42, dim=dim)
            planar_class_obj._prepare()

    def test_empty_comgroup(self, ag, empty_ag):
        """Test behaviour for empty comgroup."""
        with pytest.raises(ValueError,
                           match='Comgroup does not contain any atoms.'):
            planar_class_obj = PlanarClass(ag, pos_arg=42, comgroup=empty_ag)
            planar_class_obj._prepare()

    def test_binwidth(self, ag):
        """Test binwidth."""
        binwidth = 2
        planar_class_obj = PlanarClass(ag, pos_arg=42, binwidth=binwidth)
        planar_class_obj._prepare()

        assert planar_class_obj.binwidth == binwidth
        assert planar_class_obj.n_bins == 60 / (binwidth)

    def test_n_bins(self, planar_class_obj, caplog):
        """Test n bins."""
        planar_class_obj._verbose = True
        caplog.set_level(logging.INFO)
        planar_class_obj._prepare()

        assert planar_class_obj.n_bins == 60
        assert "Using 60 bins" in [rec.message for rec in caplog.records]

    def test_zmin(self, ag):
        """Test zmin."""
        binwidth = 2
        planar_class_obj = PlanarClass(ag, pos_arg=42, binwidth=binwidth)
        planar_class_obj.zmin = 10
        planar_class_obj._prepare()

        assert planar_class_obj.n_bins == (60 - 10) / (binwidth)

    def test_zmax(self, ag):
        """Test zmax."""
        binwidth = 2
        planar_class_obj = PlanarClass(ag, pos_arg=42, binwidth=binwidth)
        planar_class_obj._zmax = 50
        planar_class_obj._prepare()

        assert planar_class_obj._zmax == planar_class_obj.zmax
        assert planar_class_obj.n_bins == 50 / (binwidth)

    def test_zmax_dim(self, ag):
        """Test zmax dim."""
        planar_class_obj = PlanarClass(ag, pos_arg=42)
        planar_class_obj._zmax = None
        planar_class_obj._prepare()

        assert planar_class_obj.zmax == ag.universe.dimensions[2]

    def test_L_cum(self, ag):
        """Test cummulative box length L_cum."""
        planar_class_obj = PlanarClass(ag, pos_arg=42)
        planar_class_obj._zmax = None
        planar_class_obj._trajectory = ag.universe.trajectory
        planar_class_obj.run()

        L_cum = ag.universe.trajectory.n_frames * ag.universe.dimensions[2]
        assert planar_class_obj.L_cum == L_cum

    def test_zmin_zmax(self, ag):
        """Test zmin zmax."""
        binwidth = 2
        planar_class_obj = PlanarClass(ag, pos_arg=42, binwidth=binwidth)
        planar_class_obj.zmin = 10
        planar_class_obj._zmax = 50
        planar_class_obj._prepare()

        assert planar_class_obj.n_bins == (50 - 10) / (binwidth)

    def test_results_z(self, ag):
        """Test results z."""
        planar_class_obj = PlanarClass(ag, pos_arg=42)
        planar_class_obj._prepare()
        planar_class_obj.L_cum = 60
        planar_class_obj._frame_index = 0
        planar_class_obj._conclude()

        assert_allclose(planar_class_obj.results["z"],
                        np.linspace(0.5, 60 - 0.5, 60, endpoint=False))

    @pytest.mark.parametrize('dim', (0, 1, 2))
    def test_comgroup_z(self, ag, dim):
        """Test z list."""
        planar_class_obj = PlanarClass(ag,
                                       pos_arg=42,
                                       dim=dim,
                                       comgroup=ag.select_atoms("name OW"))
        planar_class_obj.run(stop=1)

        z = [-10 + 0.5, -10 + 0.5, -30 + 0.5]
        assert (planar_class_obj.results["z"]).min() == z[dim]

    @pytest.mark.parametrize('dim', (0, 1, 2))
    def test_comgroup(self, ag, dim):
        """Test comgroup."""
        planar_class_obj = PlanarClass(ag,
                                       pos_arg=42,
                                       dim=dim,
                                       comgroup=ag.select_atoms("name OW"))
        planar_class_obj._prepare()
        planar_class_obj._ts = planar_class_obj._universe
        planar_class_obj._single_frame()

        assert_allclose(ag.atoms.positions[0, dim],
                        [28.41, 6.36, 37.62][dim],
                        rtol=1e-01)


class TestPlanarBaseChilds:
    """Tests for the PlanarBaseChilds class."""

    @pytest.fixture()
    def ag_single_frame(self):
        """Import MDA univers."""
        u = mda.Universe(WATER_TPR, WATER_GRO)
        return u.atoms

    members = []
    for _, member in inspect.getmembers(maicos):
        if inspect.isclass(member) and issubclass(member, base.PlanarBase) \
                and member is not base.PlanarBase:
            members.append(member)

    @pytest.mark.parametrize("Member", members)
    def test_check_attr_change(self, Member, ag_single_frame):
        """Test check attr change."""
        params = dict(dim=2,
                      zmin=0,
                      zmax=None,
                      binwidth=1,
                      comgroup=None,
                      center=False)
        ana_obj = Member(ag_single_frame, **params).run()
        pb_obj = base.PlanarBase(ag_single_frame, **params).run()

        assert_equal(ana_obj.results.z, pb_obj.results.z)
        assert_equal(ana_obj.n_bins, pb_obj.n_bins)

        assert ana_obj.zmax == pb_obj.zmax
        assert ana_obj.L_cum == pb_obj.L_cum


class TestProfilePlanarBase:
    """Test the ProfilePlanarBase class."""

    @pytest.fixture()
    def u(self):
        """Generate a universe containing 125 atoms at random positions.

        Atom positions are drawn from a uniform distribution from 0 to 1 A in
        each direction. Masses and charges of all particles are 1.
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
        shape = (n_frames, n_atoms, 3)
        coords = rng.random(shape)

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

        positions = np.array([[0, 0, 0], [0, 1, 0],
                              [2, 1, 0], [2, 2, 0]])

        universe.trajectory = get_reader_for(positions)(positions,
                                                        order='fac',
                                                        n_atoms=4)

        for ts in universe.trajectory:
            ts.dimensions = np.array([2, 3, 3, 90, 90, 90])

        return universe

    def weights(self, ag, grouping, dim, scale=1):
        """Scalable weights for profile calculations."""
        return scale * _density_weights(ag, grouping, dim, dens="number")

    @pytest.fixture()
    def params(self, u):
        """Fixture for PlanarBase class atributes."""
        p = dict(function=self.weights,
                 atomgroups=u.atoms,
                 normalization="number",
                 dim=2,
                 zmin=0,
                 zmax=None,
                 binwidth=0.1,
                 center=False,
                 comgroup=None,
                 sym=False,
                 grouping="atoms",
                 make_whole=False,
                 binmethod="com",
                 concfreq=0,
                 output="profile.dat")
        return p

    @pytest.mark.parametrize("normalization", ["volume", "number", "None"])
    def test_profile(self, u, normalization, params):
        """Test profile with different normalizations."""
        params.update(normalization=normalization)
        profile_planar = base.ProfilePlanarBase(**params).run()

        if normalization == "volume":
            # Divide by 2 since only half of the box is filled with atoms.
            profile_vals = u.atoms.n_atoms / (u.trajectory.ts.volume / 2)
        elif normalization == "number":
            profile_vals = 1
        else:  # == None
            # Divide by 2: Half of the box is filled with atoms.
            profile_vals = u.atoms.n_atoms * profile_planar.n_bins / 2 / 100

        actual = profile_planar.results.profile_mean.flatten()
        desired = np.zeros(profile_planar.n_bins)
        desired[:10] = profile_vals

        assert_allclose(actual, desired, atol=2, rtol=1e-2)

        # TODO: Add test for error and standard deviation.
        # Needs analytical estimaton of the error

    def test_sym(self, u, params):
        """Test profile symmetrization."""
        params.update(comgroup=u.atoms, sym=True)
        profile_planar = base.ProfilePlanarBase(**params).run()

        actual = profile_planar.results.profile_mean.flatten()
        desired = [0, 0, 0, 0, 0.4, 1, 1, 1, 1, 1]
        desired += desired[::-1]

        assert_allclose(actual, desired, atol=1e-2)

    @pytest.mark.parametrize("grouping", ["atoms", "segments", "residues",
                                          "molecules", "fragments"])
    def test_grouping(self, u_dimers, grouping, params):
        """Test profile grouping."""
        params.update(atomgroups=u_dimers.atoms,
                      dim=1,
                      binwidth=1,
                      grouping=grouping)
        profile_planar = base.ProfilePlanarBase(**params).run()
        actual = profile_planar.results.profile_mean.flatten()

        if grouping == "atoms":
            desired = [1, 1, 1]
        else:
            desired = [1, 0, 1]

        assert_equal(actual, desired)

    @pytest.mark.parametrize("binmethod, desired",
                             [("cog", [1, 1, 0]),
                              ("com", [1, 0, 1]),
                              ("coc", [1, 1, 0])])
    def test_binmethod(self, u_dimers, binmethod, desired, params):
        """Test different bin methods."""
        params.update(atomgroups=u_dimers.atoms,
                      dim=1,
                      binwidth=1,
                      binmethod=binmethod,
                      grouping="molecules")
        profile_planar = base.ProfilePlanarBase(**params).run()
        actual = profile_planar.results.profile_mean.flatten()
        assert_equal(actual, desired)

    @pytest.mark.parametrize("make_whole, desired",
                             [(False, [1, 1]), (True, [2, 0])])
    def test_make_whole(self, u_dimers, make_whole, desired, params):
        """Test making molecules whole."""
        params.update(atomgroups=u_dimers.atoms,
                      dim=0,
                      binwidth=1,
                      make_whole=make_whole,
                      binmethod='com',
                      normalization="none",
                      grouping="molecules")

        profile_planar = base.ProfilePlanarBase(**params).run()
        actual = profile_planar.results.profile_mean.flatten()
        assert_equal(actual, desired)

    def test_raise_sym_no_comgroup(self, params):
        """Test error raise for symmetrization without provided comgroup."""
        params.update(sym=True)
        with pytest.raises(ValueError, match="For symmetrization the"):
            base.ProfilePlanarBase(**params).run()

    def test_wrong_normalization(self, params):
        """Test profile for a non existing normalization."""
        params.update(normalization="foo")

        with pytest.raises(ValueError, match="not supported"):
            base.ProfilePlanarBase(**params).run()

    def test_wrong_grouping(self, params):
        """Test grouping for a non existing type."""
        params.update(grouping="foo")

        with pytest.raises(ValueError, match="is not a valid option for"):
            base.ProfilePlanarBase(**params).run()

    def test_wrong_binmethod(self, params):
        """Test error raise for a non existing binmethod."""
        params.update(binmethod="foo")

        with pytest.raises(ValueError, match="is an unknown binning"):
            base.ProfilePlanarBase(**params).run()

    def test_make_whole_atoms(self, params):
        """Test that make_whole is always False for grouping wrt to atoms."""
        params.update(make_whole=True, grouping="atoms")
        profile_planar = base.ProfilePlanarBase(**params).run(stop=1)
        assert profile_planar.make_whole is False

    def test_f_kwargs(self, params):
        """Test an extra keyword argument."""
        scale = 3
        params.update(f_kwargs={"scale": scale})
        profile_planar = base.ProfilePlanarBase(**params).run()

        actual = profile_planar.results.profile_mean.flatten()
        desired = np.zeros(profile_planar.n_bins)
        desired[:10] = scale

        assert_allclose(actual, desired, atol=2, rtol=1e-2)

    def test_multigroup(self, u, params):
        """Test analysis for a list of atomgroups."""
        params.update(atomgroups=[u.atoms, u.atoms])
        profile_planar = base.ProfilePlanarBase(**params).run()

        actual = profile_planar.results.profile_mean
        desired = np.zeros([profile_planar.n_bins, 2])
        desired[:10, :] = 1

        assert_allclose(actual, desired, atol=2, rtol=1e-2)

    def test_output_name(self, params):
        """Test output name of save method."""
        params.update(output="foo.dat")
        profile_planar = base.ProfilePlanarBase(**params).run(stop=1)
        profile_planar.save()
        assert os.path.exists(params["output"])

    def test_output(self, params, tmpdir):
        """Test output."""
        with tmpdir.as_cwd():
            profile_planar = base.ProfilePlanarBase(**params).run(stop=1)
            profile_planar.save()
            res_dens = np.loadtxt(profile_planar.output)

        assert_allclose(profile_planar.results.z,
                        res_dens[:, 0],
                        rtol=2)

        assert_allclose(profile_planar.results.profile_mean[:, 0],
                        res_dens[:, 1],
                        rtol=2)

        assert_allclose(profile_planar.results.profile_err[:, 0],
                        res_dens[:, 2],
                        rtol=2)
