#!/usr/bin/env python3
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
#
# Copyright (c) 2022 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the GNU Public Licence, v3 or any higher version
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for the decorators."""

import MDAnalysis as mda
import pytest
from modules.datafiles import WATER_GRO, WATER_TPR

from maicos import decorators
from maicos.modules.base import AnalysisBase


@pytest.mark.parametrize(
    "docstring_variable, substring_list",
    [(decorators.verbose_parameter_doc, ["verbose : bool"]),
     (decorators.planar_class_parameters_doc,
      ["dim : int", "zmin : float", "zmax : float", "binwidth"]),
     (decorators.profile_planar_class_parameters_doc,
      ["atomgroups : list[AtomGroup]", "output : str", "concfreq : int"]),
     (decorators.planar_class_attributes_doc, ["results.z : list"]),
     (decorators.profile_planar_class_attributes_doc, [
         "results.profile_mean : np.ndarray",
         "results.profile_err : np.ndarray"]),
     (decorators.make_whole_parameter_doc, ["make_whole : bool"])])
def test_docstring(docstring_variable, substring_list):
    """Test if docstring variable contains substrings."""
    for s in substring_list:
        assert s in docstring_variable


@pytest.mark.parametrize("doc, new_doc", [("${TEST}", "test"), (None, None),
                                          ("", ""), ("foo", "foo")])
def test_set_doc(doc, new_doc):
    """Test decorator for setting of phrase in documentation."""

    def func():
        pass

    func.__doc__ = doc
    func_decorated = decorators.set_doc(func, doc, new_doc)
    assert func_decorated.__doc__ == new_doc


@pytest.mark.parametrize(
    "new, old, decorator",
    [(decorators.verbose_parameter_doc,
     "${VERBOSE_PARAMETER}",
      decorators.set_verbose_doc),
     (decorators.planar_class_parameters_doc,
     "${PLANAR_CLASS_PARAMETERS}",
      decorators.set_planar_class_doc),
     (decorators.planar_class_attributes_doc,
     "${PLANAR_CLASS_ATTRIBUTES}",
      decorators.set_planar_class_doc),
     (decorators.profile_planar_class_parameters_doc,
      "${PLANAR_PROFILE_CLASS_PARAMETERS}",
      decorators.set_profile_planar_class_doc),
     (decorators.profile_planar_class_attributes_doc,
      "${PLANAR_PROFILE_CLASS_ATTRIBUTES}",
      decorators.set_profile_planar_class_doc)])
def test_explicit_doc(new, old, decorator):
    """Test if old phrase is replace by the correct new phrase."""

    def func():
        pass

    func.__doc__ = old
    func_decorated = decorator(func)
    assert new in func_decorated.__doc__


def single_class(atomgroup, filter):
    """Single class."""

    @decorators.charge_neutral(filter)
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

    @decorators.charge_neutral(filter)
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
