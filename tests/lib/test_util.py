#!/usr/bin/env python3
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
#
# Copyright (c) 2022 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the GNU Public Licence, v3 or any higher version
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for the utilities."""
import os
import sys
import warnings
from unittest.mock import patch

import MDAnalysis as mda
import numpy as np
import pytest
from MDAnalysisTests.core.util import UnWrapUniverse
from numpy.testing import assert_equal

import maicos.lib.util
from maicos.core.base import AnalysisBase


sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from data import WATER_GRO, WATER_TPR  # noqa: E402


@pytest.mark.parametrize('u, compound, index', (
    (UnWrapUniverse(),
     "molecules",
     "molnums"),
    (UnWrapUniverse(have_molnums=False, have_bonds=True),
     "fragments",
     "fragindices",),
    (UnWrapUniverse(have_molnums=False, have_bonds=False),
     "residues",
     "resindices"),))
def test_get_compound(u, compound, index):
    """Tests check compound."""
    comp = maicos.lib.util.get_compound(u.atoms)
    assert compound == comp
    comp, ix = maicos.lib.util.get_compound(u.atoms, return_index=True)
    assert compound == comp
    assert_equal(ix, getattr(u.atoms, index))


def test_get_cli_input():
    """Tests get cli input."""
    testargs = ['maicos', 'foo', "foo bar"]
    with patch.object(sys, 'argv', testargs):
        assert maicos.lib.util.get_cli_input() == 'maicos foo "foo bar"'


@pytest.mark.parametrize("doc, new_doc", [("${TEST}", "test"),
                                          (None, None), ("", ""),
                                          ("foo", "foo"),
                                          ("${TEST} ${BLA}", "test blu")])
def test_render_docs(doc, new_doc):
    """Test decorator for setting of phrase in documentation."""

    def func():
        pass

    doc_dict = dict(TEST="test", BLA="blu")

    func.__doc__ = doc
    func_decorated = maicos.lib.util._render_docs(func, doc_dict=doc_dict)
    assert func_decorated.__doc__ == new_doc


def single_class(atomgroup, filter):
    """Single class."""

    @maicos.lib.util.charge_neutral(filter)
    class SingleCharged(AnalysisBase):

        def __init__(self, atomgroup):
            self.atomgroup = atomgroup
            self.filter = filter

        def _prepare(self):

            def inner_func(self):
                pass

            inner_func(self)

    return SingleCharged(atomgroup)


def multi_class(atomgroup, filter):
    """Multi class."""

    @maicos.lib.util.charge_neutral(filter)
    class MultiCharged(AnalysisBase):

        def __init__(self, atomgroups):
            self.atomgroups = atomgroups
            self.filter = filter

        def _prepare(self):

            def inner_func(self):
                pass

            inner_func(self)

    return MultiCharged(atomgroup)


class TestChargedDecorator(object):
    """Test charged decorator."""

    @pytest.fixture()
    def ag(self):
        """Import MDA universe."""
        u = mda.Universe(WATER_TPR, WATER_GRO)
        return u.atoms

    def test_charged_single(self, ag):
        """Test charged single."""
        with pytest.raises(UserWarning):
            single_class(ag.select_atoms("name OW*"),
                         filter="error")._prepare()

    def test_charged_Multi(self, ag):
        """Test charged multi."""
        with pytest.raises(UserWarning):
            multi_class([ag.select_atoms("name OW*"), ag],
                        filter="error")._prepare()

    def test_charged_single_warn(self, ag):
        """Test charged single warn."""
        with pytest.warns(UserWarning):
            single_class(ag.select_atoms("name OW*"),
                         filter="default")._prepare()

    def test_charged_Multi_warn(self, ag):
        """Test charged multi warn."""
        with pytest.warns(UserWarning):
            multi_class([ag.select_atoms("name OW*")],
                        filter="default")._prepare()

    def test_universe_charged_single(self, ag):
        """Test universe charged single."""
        ag[0].charge += 1
        with pytest.raises(UserWarning):
            single_class(ag.select_atoms("name OW*"),
                         filter="error")._prepare()

    def test_universe_slightly_charged_single(self, ag):
        """Test universe slightly charged single."""
        ag[0].charge += 1E-5
        single_class(ag, filter="error")._prepare()


def unwrap_refgroup_class(**kwargs):
    """Simple class setting kywword arguments as attrubutes."""

    @maicos.lib.util.unwrap_refgroup
    class UnwrapRefgroup(AnalysisBase):

        def __init__(self, **kwargs):
            for key, val in kwargs.items():
                setattr(self, key, val)

        def _prepare(self):

            def inner_func(self):
                pass

            inner_func(self)

    return UnwrapRefgroup(**kwargs)


class TestWrapRefgroup:
    """Test the `unwrap_refgroup` decorator."""

    def test_unwrap_refgroup(self):
        """Test to raise an error if unwrap and refgroup."""
        with pytest.raises(ValueError, match="`unwrap=False` and `refgroup"):
            unwrap_refgroup_class(unwrap=False, refgroup="foo")._prepare()

    @pytest.mark.parametrize("kwargs", (
        {},
        {"unwrap": True, "refgroup": None},
        {"unwrap": False, "refgroup": None},
        {"unwrap": True, "refgroup": "foo"},))
    def test_noerror(self, kwargs):
        """Decorator should raise an error otherwise."""
        unwrap_refgroup_class(**kwargs)._prepare()


class TestTrajectoryPrecision(object):
    """Test the detection of the trajectory precision."""

    @pytest.fixture()
    def trj(self):
        """Import MDA universe trajectory."""
        return mda.Universe(WATER_TPR, WATER_GRO).trajectory

    def test_gro_trajectory(self, trj):
        """Test detect gro traj."""
        assert_equal(maicos.lib.util.trajectory_precision(trj),
                     np.float32(0.01))


class TestCitationReminder(object):
    """Test the detection of the trajectory precision."""

    def test_single_citation(self):
        """Test if a single citation will get printed correctly."""
        doi = "10.1103/PhysRevE.92.032718"

        assert doi in maicos.lib.util.citation_reminder(doi)
        assert "please read" in maicos.lib.util.citation_reminder(doi)
        assert "Schaaf" in maicos.lib.util.citation_reminder(doi)

    def test_mutliple_citation(self):
        """Test if a two citations will get printed at the same time."""
        dois = ["10.1103/PhysRevE.92.032718",
                "10.1103/PhysRevLett.117.048001"]

        assert 'Schlaich' in maicos.lib.util.citation_reminder(*dois)
        assert 'Schaaf' in maicos.lib.util.citation_reminder(*dois)
        assert dois[0] in maicos.lib.util.citation_reminder(*dois)
        assert dois[1] in maicos.lib.util.citation_reminder(*dois)


class TestCorrelationAnalysis(object):
    """Test the calculation of the correlation of the data."""

    def test_short_data(self, mocker, caplog):
        """Test if a warning is raised if the data is too short."""
        warning = "Your trajectory is too short to estimate a correlation "
        with pytest.warns(match=warning):
            corrtime = maicos.lib.util.correlation_analysis(np.arange(4))
        assert corrtime == -1

    def test_insufficient_data(self, mocker, caplog):
        """Test if a warning is raised if the data is insufficient."""
        warning = "Your trajectory does not provide sufficient statistics to "
        mocker.patch('maicos.lib.util.correlation_time', return_value=-1)
        with pytest.warns(match=warning):
            corrtime = maicos.lib.util.correlation_analysis(np.arange(10))
        assert corrtime == -1

    def test_correlated_data(self, mocker, caplog):
        """Test if a warning is issued if the data is correlated."""
        corrtime = 10
        warnings = ("Your data seems to be correlated with a ",
                    f"correlation time which is {corrtime + 1:.2f} ",
                    f"of {int(np.ceil(2 * corrtime + 1)):d} to get a ")
        mocker.patch('maicos.lib.util.correlation_time', return_value=corrtime)
        for warning in warnings:
            with pytest.warns(match=warning):
                returned_corrtime = \
                    maicos.lib.util.correlation_analysis(np.arange(10))
        assert returned_corrtime == corrtime

    def test_uncorrelated_data(self, mocker, caplog):
        """Test that no warning is issued if the data is uncorrelated."""
        corrtime = 0.25
        mocker.patch('maicos.lib.util.correlation_time', return_value=corrtime)
        with warnings.catch_warnings():  # no warning should be issued
            warnings.simplefilter("error")
            returned_corrtime = \
                maicos.lib.util.correlation_analysis(np.arange(10))

        assert returned_corrtime == corrtime

    def test_no_data(self, mocker, caplog):
        """Test that no warning is issued if no data exists."""
        with warnings.catch_warnings():  # no warning should be issued
            warnings.simplefilter("error")
            returned_corrtime = \
                maicos.lib.util.correlation_analysis(np.nan * np.arange(10))
        assert returned_corrtime == -1
