#!/usr/bin/env python
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
#
# Copyright (c) 2023 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the GNU Public Licence, v3 or any higher version
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for the utilities."""
import sys
import warnings
from pathlib import Path
from unittest.mock import patch

import MDAnalysis as mda
import numpy as np
import pytest
from MDAnalysisTests.core.util import UnWrapUniverse
from numpy.testing import assert_allclose, assert_equal

import maicos.lib.util
from maicos.core.base import AnalysisBase


sys.path.append(str(Path(__file__).parents[1]))
from data import WATER_GRO, WATER_TPR  # noqa: E402
from modules.create_mda_universe import circle_of_water_molecules  # noqa: E402


@pytest.mark.parametrize(
    "u, compound",
    (
        (UnWrapUniverse(), "molecules"),
        (
            UnWrapUniverse(have_molnums=False, have_bonds=True),
            "fragments",
        ),
        (
            UnWrapUniverse(have_molnums=False, have_bonds=False),
            "residues",
        ),
    ),
)
def test_get_compound(u, compound):
    """Tests check compound."""
    comp = maicos.lib.util.get_compound(u.atoms)
    assert compound == comp


def test_get_cli_input():
    """Tests get cli input."""
    testargs = ["maicos", "foo", "foo bar"]
    with patch.object(sys, "argv", testargs):
        assert maicos.lib.util.get_cli_input() == 'maicos foo "foo bar"'


def test_banner():
    """Test banner string by checking some necesarry features.

    The banner is not tested for exact string equality. We just check the necessary
    features. Everything else is up to the developers to get creative.
    """
    # Test the character replacement
    assert maicos.lib.util.maicos_banner(frame_char="%")[1] == "%"
    # Check for correct number of lines as a sanity check
    assert maicos.lib.util.maicos_banner().count("\n") == 10
    # Check that newlines are added top and bottom
    assert maicos.lib.util.maicos_banner().startswith("\n")
    assert maicos.lib.util.maicos_banner().endswith("\n")
    # Check for correct length of lines (80 characters excluding top and bottom)
    # Also add in a long version string to check that it doesn't overflow
    for line in maicos.lib.util.maicos_banner(version="v1.10.11").split("\n")[1:-1]:
        assert len(line) == 80
    # Check that the version is correctly inserted
    assert "v0.0.1" in maicos.lib.util.maicos_banner(version="v0.0.1")


@pytest.mark.parametrize(
    "doc, new_doc",
    [
        ("${TEST}", "test"),
        (None, None),
        ("", ""),
        ("foo", "foo"),
        ("${TEST} ${BLA}", "test blu"),
        ("${OUTER}", "desc with inner"),
    ],
)
def test_render_docs(doc, new_doc):
    """Test decorator for replacing patterns in docstrings."""

    def func():
        pass

    DOC_DICT = dict(
        TEST="test",
        BLA="blu",
        INNER="inner",
        OUTER="desc with ${INNER}",
    )

    func.__doc__ = doc
    func_decorated = maicos.lib.util._render_docs(func, doc_dict=DOC_DICT)
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
            single_class(ag.select_atoms("name OW*"), filter="error")._prepare()

    def test_charged_Multi(self, ag):
        """Test charged multi."""
        with pytest.raises(UserWarning):
            multi_class([ag.select_atoms("name OW*"), ag], filter="error")._prepare()

    def test_charged_single_warn(self, ag):
        """Test charged single warn."""
        with pytest.warns(UserWarning):
            single_class(ag.select_atoms("name OW*"), filter="default")._prepare()

    def test_charged_Multi_warn(self, ag):
        """Test charged multi warn."""
        with pytest.warns(UserWarning):
            multi_class([ag.select_atoms("name OW*")], filter="default")._prepare()

    def test_universe_charged_single(self, ag):
        """Test universe charged single."""
        ag[0].charge += 1
        with pytest.raises(UserWarning):
            single_class(ag.select_atoms("name OW*"), filter="error")._prepare()

    def test_universe_slightly_charged_single(self, ag):
        """Test universe slightly charged single."""
        ag[0].charge += 1e-5
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

    @pytest.mark.parametrize(
        "kwargs",
        (
            {},
            {"unwrap": True, "refgroup": None},
            {"unwrap": False, "refgroup": None},
            {"unwrap": True, "refgroup": "foo"},
        ),
    )
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
        assert_equal(maicos.lib.util.trajectory_precision(trj), np.float32(0.01))


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
        dois = ["10.1103/PhysRevE.92.032718", "10.1103/PhysRevLett.117.048001"]

        assert "Schlaich" in maicos.lib.util.citation_reminder(*dois)
        assert "Schaaf" in maicos.lib.util.citation_reminder(*dois)
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
        mocker.patch("maicos.lib.util.correlation_time", return_value=-1)
        with pytest.warns(match=warning):
            corrtime = maicos.lib.util.correlation_analysis(np.arange(10))
        assert corrtime == -1

    def test_correlated_data(self, mocker, caplog):
        """Test if a warning is issued if the data is correlated."""
        corrtime = 10
        warnings = (
            "Your data seems to be correlated with a ",
            f"correlation time which is {corrtime + 1:.2f} ",
            f"of {int(np.ceil(2 * corrtime + 1)):d} to get a ",
        )
        mocker.patch("maicos.lib.util.correlation_time", return_value=corrtime)
        for warning in warnings:
            with pytest.warns(match=warning):
                returned_corrtime = maicos.lib.util.correlation_analysis(np.arange(10))
        assert returned_corrtime == corrtime

    def test_uncorrelated_data(self, mocker, caplog):
        """Test that no warning is issued if the data is uncorrelated."""
        corrtime = 0.25
        mocker.patch("maicos.lib.util.correlation_time", return_value=corrtime)
        with warnings.catch_warnings():  # no warning should be issued
            warnings.simplefilter("error")
            returned_corrtime = maicos.lib.util.correlation_analysis(np.arange(10))

        assert returned_corrtime == corrtime

    def test_no_data(self, mocker, caplog):
        """Test that no warning is issued if no data exists."""
        with warnings.catch_warnings():  # no warning should be issued
            warnings.simplefilter("error")
            returned_corrtime = maicos.lib.util.correlation_analysis(
                np.nan * np.arange(10)
            )
        assert returned_corrtime == -1


class Testget_center:
    """Test the `get_center` function."""

    compounds = ["group", "segments", "residues", "molecules", "fragments"]

    @pytest.fixture
    def ag(self):
        """An AtomGroup made from water molecules."""
        return mda.Universe(WATER_TPR, WATER_GRO)

    @pytest.mark.parametrize("compound", compounds)
    def cog(self, ag, compound):
        """Test same center of geometry."""
        assert_equal(
            maicos.lib.util.get_center(
                atomgroup=ag, bin_method="cog", compound=compound
            ),
            ag.center_of_geometry(compound=compound),
        )

    @pytest.mark.parametrize("compound", compounds)
    def com(self, ag, compound):
        """Test same center of mass."""
        assert_equal(
            maicos.lib.util.get_center(
                atomgroup=ag, bin_method="com", compound=compound
            ),
            ag.center_of_mass(compound=compound),
        )

    @pytest.mark.parametrize("compound", compounds)
    def coc(self, ag, compound):
        """Test same center of charge."""
        assert_equal(
            maicos.lib.util.get_center(
                atomgroup=ag, bin_method="cog", compound=compound
            ),
            ag.center_of_charge(compound=compound),
        )

    def test_get_center_unknown(self):
        """Test a wrong bin_method."""
        with pytest.raises(ValueError, match="'foo' is an unknown binning"):
            maicos.lib.util.get_center(atomgroup=None, bin_method="foo", compound=None)


class TestUnitVectors:
    """Test the `unit_vectors` functions."""

    @pytest.mark.parametrize("pdim", [0, 1, 2])
    def test_unit_vectors_planar(self, pdim):
        """Test calculation of planar unit vectors."""
        unit_vectors = np.zeros(3)
        unit_vectors[pdim] += 1

        assert_equal(
            maicos.lib.util.unit_vectors_planar(
                atomgroup=None, grouping=None, pdim=pdim
            ),
            unit_vectors,
        )

    @pytest.mark.parametrize("dim", [0, 1, 2])
    def test_unit_vectors_cylinder_r(self, dim):
        """Test calculation of cylindrical unit vectors in the radial direction."""
        ag, _ = circle_of_water_molecules(4, 90, radius=5)

        unit_vectors = maicos.lib.util.unit_vectors_cylinder(
            atomgroup=ag, grouping="residues", bin_method="com", dim=dim, pdim="r"
        )

        # Test that the length of the vectors is 1.
        assert_allclose(
            np.linalg.norm(unit_vectors, axis=1), np.ones(len(unit_vectors))
        )

        transform = maicos.lib.util.get_center(
            atomgroup=ag, bin_method="com", compound="residues"
        )

        transform -= ag.universe.dimensions[:3] / 2

        # set z direction to zero. r in cylindrical coordinates contains only x and y.
        transform[:, dim] = 0
        transform /= np.linalg.norm(transform, axis=1)[:, np.newaxis]

        assert_allclose(transform, unit_vectors)

    @pytest.mark.parametrize("dim", [0, 1, 2])
    def test_unit_vectors_cylinder_z(self, dim):
        """Test calculation of cylindrical unit vectors in the axial direction."""
        ag, _ = circle_of_water_molecules(4, 90, radius=5)

        unit_vectors = maicos.lib.util.unit_vectors_cylinder(
            atomgroup=ag, grouping="residues", bin_method="com", dim=dim, pdim="z"
        )

        test_unit_vectors = np.zeros(3)
        test_unit_vectors[dim] += 1

        assert_equal(
            test_unit_vectors,
            unit_vectors,
        )

    def test_unit_vectors_sphere(self):
        """Test calculation of spherical unit vectors."""
        ag, _ = circle_of_water_molecules(4, 90, radius=5)

        unit_vectors = maicos.lib.util.unit_vectors_sphere(
            atomgroup=ag, grouping="residues", bin_method="com"
        )

        # Test that the length of the vectors is 1.
        assert_allclose(
            np.linalg.norm(unit_vectors, axis=1), np.ones(len(unit_vectors))
        )

        transform = maicos.lib.util.get_center(
            atomgroup=ag, bin_method="com", compound="residues"
        )

        # shift origin to box center and afterwards normalize
        transform -= ag.universe.dimensions[:3] / 2
        transform /= np.linalg.norm(transform, axis=1)[:, np.newaxis]

        assert_allclose(transform, unit_vectors)
