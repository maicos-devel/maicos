#!/usr/bin/env python3
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
#
# Copyright (c) 2022 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the GNU Public Licence, v3 or any higher version
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for the utilities."""

import sys
from unittest.mock import patch

import MDAnalysis as mda
import numpy as np
import pytest
from MDAnalysisTests.core.util import UnWrapUniverse

import maicos.lib.util
from maicos.core.base import AnalysisBase


sys.path.append("..")
from data import LAMMPS10WATER, WATER_GRO, WATER_TPR  # noqa: E402


def test_check_compound():
    """Tests check compound."""
    u = UnWrapUniverse()
    assert maicos.lib.util.check_compound(u.atoms) == "molecules"

    u = UnWrapUniverse(have_molnums=False, have_bonds=True)
    assert maicos.lib.util.check_compound(u.atoms) == "fragments"

    u = UnWrapUniverse(have_molnums=False, have_bonds=False)
    assert maicos.lib.util.check_compound(u.atoms) == "residues"


def test_sort_atomsgroup_lammps():
    """Tests sort atoms group LAMMPS."""
    u = mda.Universe(LAMMPS10WATER)
    atoms = maicos.lib.util.sort_atomgroup(u.atoms)

    assert np.all(np.diff(atoms.fragindices) >= 0)


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
    func_decorated = maicos.lib.util.render_docs(func, doc_dict=doc_dict)
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
