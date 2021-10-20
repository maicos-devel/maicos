#!/usr/bin/env python3
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
#
# Copyright (c) 2020 Authors and contributors
# (see the file AUTHORS for the full list of names)
#
# Released under the GNU Public Licence, v2 or any higher version
# SPDX-License-Identifier: GPL-2.0-or-later

import argparse

import numpy as np
import MDAnalysis as mda
import pytest

from numpy.testing import assert_equal, assert_almost_equal

from maicos.modules.base import SingleGroupAnalysisBase, MultiGroupAnalysisBase
from maicos.decorators import planar_base, charge_neutral

from modules.datafiles import WATER_GRO, WATER_TPR, AIRWATER_TRR, AIRWATER_TPR


@planar_base()
class PlanarClass(SingleGroupAnalysisBase):
    """Single line doc.

    :param pos_arg (int): positional integer argument
    :param argument (str): optional str argument
    """
    def __init__(self,
                    atomgroup,
                    pos_arg,
                    opt_arg="foo",
                    # Planar base arguments are necessary for buidling CLI
                    dim=2,
                    binwidth=0.1,
                    center=False,
                    comgroup=None):

        self.atomgroup = atomgroup
        self.pos_arg = pos_arg
        self.opt_arg = opt_arg

        self.prepared = False
        self.ran = False
        self.calculated_results = False

        # Necessary arguments for run etc
        self._universe = self.atomgroup.universe
        self._verbose = False
        self.results = {}

    def _configure_parser(self, parser):
        parser.add_argument('-p1', dest='pos_arg')
        parser.add_argument('-p2', dest='opt_arg')

    def _prepare(self):
        self.prepared = True

    def _single_frame(self):
        self.ran = True

    def _calculate_results(self):
        self.calculated_results = True

def single_class(atomgroup, filter):
    @charge_neutral(filter)
    class SingleCharged(SingleGroupAnalysisBase):
        def __init__(self, atomgroup):
            self.atomgroup = atomgroup
            self.filter = filter

        def _prepare(self):
            def inner_func(self):
                pass

            inner_func(self)

    return SingleCharged(atomgroup)


def multi_class(atomgroup, filter):
    @charge_neutral(filter)
    class MultiCharged(MultiGroupAnalysisBase):
        def __init__(self, atomgroups):
            self.atomgroups = atomgroups
            self.filter = filter

        def _prepare(self):
            def inner_func(self):
                pass

            inner_func(self)

    return MultiCharged(atomgroup)


class TestPlanarBase(object):

    @pytest.fixture()
    def ag(self):
        u = mda.Universe(AIRWATER_TPR, AIRWATER_TRR)
        return u.atoms

    @pytest.fixture()
    def planar_class_obj(self, ag):
        return PlanarClass(ag, pos_arg=42)

    @pytest.fixture()
    def parser(self):
        return argparse.ArgumentParser()

    def test_origi_init(self, ag):
        planar_class_obj = PlanarClass(ag, pos_arg=42, opt_arg="bar")
        assert planar_class_obj.pos_arg is 42
        assert planar_class_obj.opt_arg == "bar"

    @pytest.mark.parametrize('dest', ["dim", "binwidth", "center", "comgroup"])
    def test_orig_configure_parser(self,
                                   planar_class_obj,
                                   parser,
                                   dest):
        planar_class_obj._configure_parser(parser)
        args = parser.parse_known_args()[0]

        # Check if new keys exists
        args.__dict__[dest]

        # Check if old keys exists
        args.__dict__["pos_arg"]
        args.__dict__["opt_arg"]

    @pytest.mark.parametrize('dest', ["dim", "binwidth", "center", "comgroup"])
    def tes_doc(self, planar_class_obj, dest):
        assert dest in planar_class_obj.__doc__

    def test_orig_prepare(self, planar_class_obj):
        planar_class_obj._prepare()
        assert planar_class_obj.prepared

    def test_orig_single_frame(self, planar_class_obj):
        planar_class_obj._prepare()
        planar_class_obj._ts = planar_class_obj._universe
        planar_class_obj._single_frame()

        assert planar_class_obj.ran

    def test_orig_calculate_results(self, planar_class_obj):
        planar_class_obj._prepare()
        planar_class_obj._frame_index = 1
        planar_class_obj._calculate_results()

        assert planar_class_obj.calculated_results

    @pytest.mark.parametrize(('dim', 'bins'),
        ((0, [1, 4]), (1, [1, 7]), (2, [0, 4])))
    def test_get_bins(self, ag, dim, bins):
        planar_class_obj = PlanarClass(ag, pos_arg=42, binwidth=0.5)
        planar_class_obj._prepare()
        planar_class_obj._ts = planar_class_obj._universe

        # Universe dimensions are [2 nm, 2 nm, 6 nm]

        positions = np.array([[.1, .1, .1], [.6, 1.2, 2.2]]) # in nm

        assert_equal(planar_class_obj.get_bins(10 * positions, dim), bins)

    @pytest.mark.parametrize('dim', (0, 1, 2))
    def test_dim(self, ag, dim):
        planar_class_obj = PlanarClass(ag, pos_arg=42, dim=dim)
        assert planar_class_obj.dim == dim

    def test_binwidth(self, planar_class_obj):
        pass

    def test_center(self, planar_class_obj):
        pass

    def test_comgroup(self, planar_class_obj):
        pass

    def test_comgroup_verbose(self, planar_class_obj):
        pass


class TestChargedDecorator(object):
    @pytest.fixture()
    def ag(self):
        u = mda.Universe(WATER_TPR, WATER_GRO)
        return u.atoms

    def test_charged_single(self, ag):
        with pytest.raises(UserWarning):
            single_class(ag.select_atoms("name OW*"),
                         filter="error")._prepare()

    def test_charged_Multi(self, ag):
        with pytest.raises(UserWarning):
            multi_class([ag.select_atoms("name OW*"), ag],
                        filter="error")._prepare()

    def test_charged_single_warn(self, ag):
        with pytest.warns(UserWarning):
            single_class(ag.select_atoms("name OW*"),
                         filter="default")._prepare()

    def test_charged_Multi_warn(self, ag):
        with pytest.warns(UserWarning):
            multi_class([ag.select_atoms("name OW*")],
                        filter="default")._prepare()

    def test_universe_charged_single(self, ag):
        ag[0].charge += 1
        with pytest.raises(UserWarning):
            single_class(ag.select_atoms("name OW*"),
                         filter="error")._prepare()

    def test_universe_slihghtly_charged_single(self, ag):
        ag[0].charge += 1E-5
        single_class(ag, filter="error")._prepare()