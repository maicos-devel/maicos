#!/usr/bin/env python3
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
from MDAnalysis.analysis import base
from numpy.testing import assert_allclose, assert_equal

import maicos
from maicos.modules.base import AnalysisBase, PlanarBase


class Test_AnalysisBase(object):
    """Tests for the Analysis base class."""

    @pytest.fixture()
    def ag(self):
        """Import MDA universe."""
        return mda.Universe(WATER_TPR, WATER_TRR).atoms

    def test_AnalysisBase(self, ag):
        """Test AnalysisBase."""
        a = AnalysisBase(ag, verbose=True, save_results=True)

        assert a.atomgroup.n_atoms == ag.n_atoms
        assert a._trajectory == ag.universe.trajectory
        assert a._universe == ag.universe
        assert a._verbose is True
        assert type(a.results) is base.Results
        assert not hasattr(a, "atomgroups")

    def test_multigroups(self, ag):
        """Test multiple groups."""
        a = AnalysisBase([ag[:10], ag[10:]], multi_group=True)

        assert a.n_atomgroups == 2
        assert a._universe == ag.universe

    def test_different_universes(self, ag):
        """Test different universes."""
        with pytest.raises(ValueError, match="Atomgroups belong"):
            AnalysisBase([ag, mda.Universe(WATER_TPR)], multi_group=True)


class PlanarClass(PlanarBase):
    """Tests for the Planar Base class."""

    def __init__(self,
                 atomgroups,
                 pos_arg,
                 opt_arg="foo",
                 dim=2,
                 binwidth=0.1,
                 center=False,
                 comgroup=None,
                 **kwargs):
        super(PlanarClass, self).__init__(atomgroups=atomgroups,
                                          dim=dim,
                                          binwidth=binwidth,
                                          center=center,
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
    def planar_class_obj(self, ag):
        """Planar class object."""
        return PlanarClass(ag, pos_arg=42)

    def test_origi_init(self, ag):
        """Test origin  init."""
        planar_class_obj = PlanarClass(ag, pos_arg=42, opt_arg="bar")
        assert planar_class_obj.pos_arg == 42
        assert planar_class_obj.opt_arg == "bar"

    @pytest.mark.parametrize(('dim', 'bins'),
                             ((0, [1, 4]), (1, [1, 7]), (2, [0, 4])))
    def test_get_bins(self, ag, dim, bins):
        """Test get bins."""
        planar_class_obj = PlanarClass(ag, pos_arg=42, binwidth=0.5)
        planar_class_obj._prepare()
        planar_class_obj._ts = planar_class_obj._universe

        # Universe dimensions are [2 nm, 2 nm, 6 nm]
        positions = np.array([[.1, .1, .1], [.6, 1.2, 2.2]])  # in nm
        assert_equal(planar_class_obj.get_bins(10 * positions, dim), bins)

    @pytest.mark.parametrize('dim', (0, 1, 2))
    def test_dim(self, ag, dim):
        """Test dim."""
        planar_class_obj = PlanarClass(ag, pos_arg=42, dim=dim)
        assert planar_class_obj.dim == dim

    def test_binwidth(self, ag):
        """Test binwidth."""
        binwidth = 0.2
        planar_class_obj = PlanarClass(ag, pos_arg=42, binwidth=binwidth)
        planar_class_obj._prepare()

        assert planar_class_obj.binwidth == 10 * binwidth
        assert planar_class_obj.n_bins == 60 / (10 * binwidth)

    def test_n_bins(self, planar_class_obj, caplog):
        """Test n bins."""
        planar_class_obj._verbose = True
        caplog.set_level(logging.INFO)
        planar_class_obj._prepare()

        assert planar_class_obj.n_bins == 60
        assert "Using 60 bins" in [rec.message for rec in caplog.records]

    def test_zmin(self, ag):
        """Test zmin."""
        binwidth = 0.2
        planar_class_obj = PlanarClass(ag, pos_arg=42, binwidth=binwidth)
        planar_class_obj.zmin = 1
        planar_class_obj._prepare()

        assert planar_class_obj.n_bins == (60 - 10) / (10 * binwidth)

    def test_zmax(self, ag):
        """Test zmax."""
        binwidth = 0.2
        planar_class_obj = PlanarClass(ag, pos_arg=42, binwidth=binwidth)
        planar_class_obj._zmax = 5
        planar_class_obj._prepare()

        assert planar_class_obj._zmax == planar_class_obj.zmax / 10
        assert planar_class_obj.n_bins == 50 / (10 * binwidth)

    def test_zmax_dim(self, ag):
        """Test zmax dim."""
        planar_class_obj = PlanarClass(ag, pos_arg=42)
        planar_class_obj._zmax = None
        planar_class_obj._prepare()

        assert planar_class_obj.zmax == ag.universe.dimensions[2]

    def test_Lz(self, ag):
        """Test Lz."""
        planar_class_obj = PlanarClass(ag, pos_arg=42)
        planar_class_obj._zmax = None
        planar_class_obj._trajectory = ag.universe.trajectory
        planar_class_obj.run()

        Lz = ag.universe.trajectory.n_frames * ag.universe.dimensions[2]
        assert planar_class_obj.Lz == Lz

    def test_zmin_zmax(self, ag):
        """Test zmin zmax."""
        binwidth = 0.2
        planar_class_obj = PlanarClass(ag, pos_arg=42, binwidth=binwidth)
        planar_class_obj.zmin = 1
        planar_class_obj._zmax = 5
        planar_class_obj._prepare()

        assert planar_class_obj.n_bins == (50 - 10) / (10 * binwidth)

    def test_results_z(self, ag):
        """Test results z."""
        planar_class_obj = PlanarClass(ag, pos_arg=42, center=False)
        planar_class_obj._prepare()
        planar_class_obj.Lz = 60
        planar_class_obj._frame_index = 0
        planar_class_obj._conclude()

        assert_allclose(planar_class_obj.results["z"],
                        np.linspace(0.05, 6 - 0.05, 60, endpoint=False))

    def test_center(self, ag):
        """Test center."""
        planar_class_obj = PlanarClass(ag, pos_arg=42, center=True)
        planar_class_obj._prepare()
        planar_class_obj._frame_index = 0
        planar_class_obj.Lz = 60
        planar_class_obj._conclude()

        assert (planar_class_obj.results["z"]).min() == -3 + 0.05

    def test_comgroup(self, ag):
        """Test comgroup."""
        planar_class_obj = PlanarClass(ag,
                                       pos_arg=42,
                                       comgroup=ag.select_atoms("name OW"))
        planar_class_obj._prepare()

        assert planar_class_obj.center

        planar_class_obj._ts = planar_class_obj._universe
        planar_class_obj._single_frame()

        assert_allclose(ag.atoms.positions[0, :],
                        [19.01, 8.14, 37.62], rtol=1e-01)


class TestPlanarBaseChilds:
    """Tests for the PlanarBaseChilds class."""

    @pytest.fixture()
    def ag_single_frame(self):
        """Import MDA univers."""
        u = mda.Universe(WATER_TPR, WATER_GRO)
        return u.atoms

    members = []
    for _, member in inspect.getmembers(maicos):
        if inspect.isclass(member) and issubclass(member, PlanarBase) \
                and member is not PlanarBase:
            members.append(member)

    @pytest.mark.parametrize("Member", members)
    def test_check_attr_change(self, Member, ag_single_frame):
        """Test check attr change."""
        params = dict(dim=2, binwidth=0.1, comgroup=None, center=False)
        ana_obj = Member(ag_single_frame, **params).run()
        pb_obj = PlanarBase(ag_single_frame, **params).run()

        assert_equal(ana_obj.results.z, pb_obj.results.z)
        assert_equal(ana_obj.n_bins, pb_obj.n_bins)

        assert ana_obj.zmax == pb_obj.zmax
        assert ana_obj.Lz == pb_obj.Lz
