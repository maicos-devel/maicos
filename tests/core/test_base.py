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
from mdacli.libcli import find_cls_members
from MDAnalysis.analysis.base import Results
from numpy.testing import assert_allclose
from scipy.signal import find_peaks

from maicos import DensityPlanar, _version
from maicos.core import AnalysisBase, ProfileBase


sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from data import WATER_GRO, WATER_TPR, WATER_TRR  # noqa: E402


class Output(AnalysisBase):
    """Class creating a file to check the output."""

    OUTPUT = "This is the output message of an analysis class."


class SubOutput(Output):
    """Class creating a file to check the output, but as a child class."""

    OUTPUT = "This is another output message from an inheriting class."


class FileModuleInput(AnalysisBase):
    """Class creating an output file to check the module input reporting."""

    def _single_frame(self):
        # Do nothing, but the run() methods needs to be called
        pass

    def __init__(self, atomgroups, test_input="some_default", refgroup=None):
        self._locals = locals()
        super().__init__(atomgroups, refgroup=refgroup)


class Series(AnalysisBase):
    """Class creating a random time series to check observables."""

    def _prepare(self):
        self.series = np.random.rand(self.n_frames)

    def _single_frame(self):
        self._obs.observable = self.series[self._frame_index]


class Frame_types(AnalysisBase):
    """Class setting a frame Dict key to specific types.

    The frame Dict should be able to consume the following types:
    - int
    - float
    - np.ndarray
    - list
    - np.float
    - np.int
    """

    def _single_frame(self):
        self._obs.observable = self.data[self._frame_index]


class Conclude(AnalysisBase):
    """Class to test the _conclude method.

    A new file with a file name of the current analysis frame number is created every
    time the `_conclude` method is called.
    """

    def _prepare(self):
        self.conclude_count = 0

    def _single_frame(self):
        pass

    def _conclude(self):
        self.conclude_count += 1

    def save(self):
        """Save a file named after the current number of frames."""
        open(f"out_{self._index}", "w").close()


class Test_AnalysisBase(object):
    """Tests for the Analysis base class."""

    @pytest.fixture()
    def ag(self):
        """Import MDA universe."""
        return mda.Universe(WATER_TPR, WATER_TRR, in_memory=True).atoms

    @pytest.fixture()
    def ag_single_frame(self):
        """Import MDA universe of single frame."""
        return mda.Universe(WATER_TPR, WATER_GRO, in_memory=True).atoms

    @pytest.fixture()
    def empty_ag(self):
        """Define an empty atomgroup."""
        u = mda.Universe.empty(0)
        return u.atoms

    def test_AnalysisBase(self, ag):
        """Test AnalysisBase."""
        a = AnalysisBase(ag)

        assert a.atomgroup.n_atoms == ag.n_atoms
        assert a._trajectory == ag.universe.trajectory
        assert a._universe == ag.universe
        assert isinstance(a.results, Results)
        assert not hasattr(a, "atomgroups")

    def test_empty_atomgroup(self, ag):
        """Test behaviour for empty atomgroup."""
        with pytest.raises(ValueError, match="not contain any atoms."):
            class_obj = AnalysisBase(ag.select_atoms("name foo"))
            class_obj._prepare()

    def test_empty_atomgroups(self, ag):
        """Test behaviour for empty atomgroups."""
        with pytest.raises(ValueError, match="not contain any atoms."):
            class_obj = AnalysisBase(
                [ag, ag.select_atoms("name foo")], multi_group=True
            )
            class_obj._prepare()

    def test_multigroups(self, ag):
        """Test multiple groups."""
        a = AnalysisBase([ag[:10], ag[10:]], multi_group=True)

        assert a.n_atomgroups == 2
        assert a._universe == ag.universe

    def test_different_universes(self, ag):
        """Test different universes."""
        with pytest.raises(ValueError, match="Atomgroups belong"):
            AnalysisBase([ag, mda.Universe(WATER_TPR)], multi_group=True)

    def test_frame_data(self, ag):
        """Test the calculation of the frame, mean and vars results dicts."""
        ana = Series(ag)
        ana.run()

        assert_allclose(ana.means.observable, np.mean(ana.series))
        assert_allclose(ana.sems.observable, np.std(ana.series) / np.sqrt(ana.n_frames))

    def test_output_message(self, ag, tmpdir):
        """Test the output message of modules."""
        data = np.random.rand(100, 2)
        ana = Output(ag)
        ana._index = 1
        sub_ana = SubOutput(ag)
        sub_ana._index = 1

        with tmpdir.as_cwd():
            # Simple check if a single message gets written to the output file
            ana.savetxt("foo", data, columns=["First", "Second"])
            assert ana.OUTPUT in open("foo.dat").read()

            # More elaborate check to find out if output messages of subclasses
            # get written to the file in the right order.
            sub_ana.savetxt("foo2", data, columns=["First", "Second"])
            foo = open("foo2.dat", "r").readlines()
            for i, line in enumerate(foo):
                if ana.OUTPUT in line:
                    assert sub_ana.OUTPUT in foo[i + 1]
                    break
            else:
                # Fail if the loop finished without finding the first
                raise AssertionError()

    def test_module_input(self, ag, tmpdir):
        """Test the module input reporting."""
        with tmpdir.as_cwd():
            # Test if the module name is written correctly
            ana = FileModuleInput(ag)
            ana.run()
            ana.savetxt("test.dat", np.random.rand(10, 2))
            assert "Module input:    FileModuleInput(" in open("test.dat").read()

            # Test if the refgroup name is written correctly
            ana = FileModuleInput(ag, refgroup=ag)
            ana.run()
            ana.savetxt("test_refgroup.dat", np.random.rand(10, 2))
            assert "refgroup=<AtomGroup>" in open("test_refgroup.dat").read()
            assert "atomgroups=<AtomGroup>" in open("test_refgroup.dat").read()

            # Test if the default value of the test_input parameter is written
            ana = FileModuleInput(ag)
            ana.run()
            ana.savetxt("test_default.dat", np.random.rand(10, 2))
            assert "test_input='some_default'" in open("test_default.dat").read()
            assert "refgroup=None" in open("test_default.dat").read()
            assert (
                ".run(start=None, stop=None, step=None, verbose=None)"
                in open("test_default.dat").read()
            )

            # Test if the set test_input parameter is written correctly
            ana = FileModuleInput(ag, test_input="some_other_value")
            ana.run()
            ana.savetxt("test_nondefault.dat", np.random.rand(10, 2))
            assert "test_input='some_other_value'" in open("test_nondefault.dat").read()

            ana.run(step=2, stop=7, start=5, verbose=True)
            ana.savetxt("test_run.dat", np.random.rand(10, 2))
            assert (
                ".run(start=5, stop=7, step=2, verbose=True)"
                in open("test_run.dat").read()
            )

    @pytest.mark.parametrize(
        "concfreq, files",
        [(0, []), (40, ["out_40", "out_80", "out_101"]), (100, ["out_100", "out_101"])],
    )
    def test_conclude_multi_frame(self, ag, tmpdir, concfreq, files):
        """Test the conclude and save methods for multi frame trajectories."""
        with tmpdir.as_cwd():
            conclude = Conclude(ag, concfreq=concfreq)
            conclude.run()
            # check that all expected files have been written
            if concfreq != 0:
                for file in files:
                    assert os.path.exists(file)
            else:
                assert len(os.listdir(tmpdir)) == 0
            # check that the _conclude method is running
            # the expected number of times
            if concfreq != 0:
                conclude_count = np.ceil(conclude.n_frames / concfreq)
            else:
                conclude_count = 1
            assert conclude.conclude_count == conclude_count
            # check that no more files than the expected
            # ones have been written
            assert len(files) == len(os.listdir(tmpdir))

    @pytest.mark.parametrize("concfreq, file", [(0, []), (50, ["out_1"])])
    def test_conclude_single_frame(self, ag_single_frame, tmpdir, concfreq, file):
        """Test the conclude and save methods for single frame trajectories."""
        with tmpdir.as_cwd():
            conclude = Conclude(ag_single_frame, concfreq=concfreq)
            conclude.run()
            if concfreq != 0:
                assert os.path.exists(file[0])
            # check that no extra files are written
            if concfreq != 0:
                assert len(os.listdir(tmpdir)) == 1
            else:
                assert len(os.listdir(tmpdir)) == 0
            # check that no double execution of the _conclude method happens
            assert conclude.conclude_count == 1

    @pytest.mark.parametrize("indices", [[0], [0, 1, 2], [3, 4, 5]])
    def test_refgroup(self, ag, indices):
        """Test refgroup.

        We test a single atom, a broken water molecule and a whole water molecule. The
        broken molecule requires the unwrap option to be set Otherwise, the broken
        water's center of mass is not correct. See next test below.
        """
        refgroup = ag.atoms[indices]
        class_obj = Conclude(ag, refgroup=refgroup, unwrap=True)
        class_obj.run(stop=1)

        assert_allclose(
            refgroup.center_of_mass(), ag.universe.dimensions[:3] / 2, rtol=1e-01
        )

    def test_empty_refgroup(self, ag, empty_ag):
        """Test behaviour for empty refgroup."""
        with pytest.raises(ValueError, match="not contain any atoms."):
            class_obj = AnalysisBase(ag, refgroup=empty_ag)
            class_obj._prepare()

    def test_unwrap(self, ag):
        """Unwrap test for logic only; Actual test in TestProfilePlanarBase."""
        class_obj = Conclude(ag, unwrap=True)
        class_obj.run(stop=1)

    def test_unwrap_multi_one(self, ag):
        """Unwrap test for multi_group."""
        Conclude(ag, unwrap=True, multi_group=True).run(stop=1)

    def test_unwrap_multi(self, ag):
        """Unwrap test for multi_group."""
        Conclude((ag[:10], ag[10:]), unwrap=True, multi_group=True).run(stop=1)

    @pytest.mark.parametrize(
        "data, result", [([1, 2], 1.5), ([float(1), float(2)], 1.5), ([[1], [2]], 1.5)]
    )
    def test_frame_dict_types(self, ag, data, result):
        """Check supported types for the frame Dict."""
        class_obj = Frame_types(ag)
        class_obj.data = data
        class_obj.run(stop=2)
        assert class_obj.means.observable == result

    @pytest.mark.parametrize("data,", [(["1", "2"]), ([{"1": 1}, {"1": 1}])])
    def test_frame_dict_wrong_types(self, ag, data):
        """Check that unsupported types for the frame Dict throw an error."""
        class_obj = Frame_types(ag)
        class_obj.data = data
        error_msg = "Obervable observable has uncompatible type."
        with pytest.raises(TypeError, match=error_msg):
            class_obj.run(stop=2)

    def test_banner(self, ag, caplog):
        """Test whether AnalysisBase prints the MAICoS banner."""
        ana_obj = AnalysisBase(ag)

        # Create empty methods for allowing the run method to succeed.
        ana_obj._prepare = lambda: None
        ana_obj._single_frame = lambda: None
        ana_obj._conclude = lambda: None

        caplog.set_level(logging.INFO)
        ana_obj.run(stop=1)

        assert (
            r"#   \ |||||_ /    | |  | |  / ____ \   _| |_  | |____  | (_) |  ____)"
            in "".join([rec.message for rec in caplog.records])
        )
        assert _version.get_versions()["version"] in "".join(
            [rec.message for rec in caplog.records]
        )

    def test_n_bins(self, ag, caplog):
        """Test `n_bins` logger info."""
        ana_obj = AnalysisBase(ag)

        # Create empty methods for allowing the run method to succeed.
        ana_obj._prepare = lambda: None
        ana_obj._single_frame = lambda: None
        ana_obj._conclude = lambda: None

        ana_obj.n_bins = 10

        caplog.set_level(logging.INFO)
        ana_obj.run(stop=1)

        assert "Using 10 bins." in [rec.message for rec in caplog.records]

    def test_unwrap_atoms(self, ag, caplog):
        """Test that unwrap is always False for `wrap_compound="atoms"`."""
        caplog.set_level(logging.WARN)
        profile = AnalysisBase(ag, unwrap=True, wrap_compound="atoms")

        msgs = [rec.message for rec in caplog.records]
        # Assume wrap warning is first warning recorded
        assert "'atoms` is superfluous." in msgs[0]

        assert profile.unwrap is False

    def test_jitter(self, ag_single_frame):
        """Test the jitter option.

        Call the DensityPlanar module with a jitter of 0.01, and make sure that the
        density profile has no peak at a position of 100 (which would be the case
        without jitter).
        """
        dens = DensityPlanar(ag_single_frame, bin_width=1e-4, jitter=0.01).run()
        (
            hist,
            _,
        ) = np.histogram(
            np.diff(dens.results.bin_pos[np.where(dens.results.profile.T[0])]),
            bins=1000,
            range=(0, 0.1),
        )
        assert find_peaks(hist)[0][0] < 100


class Test_ProfileBase:
    """Test class for the ProfileBase Class.

    The single_frame is for now extensivley tested in the child `ProfilePlanarBase`,
    `ProfileCylinderBase` and `ProfileSphereBase` for simple physical system.
    """

    @pytest.fixture()
    def u(self):
        """Simple empty Universe."""
        universe = mda.Universe.empty(
            n_atoms=10,
            n_residues=10,
            n_segments=10,
            atom_resindex=np.arange(10),
            residue_segindex=np.arange(10),
        )

        return universe

    @pytest.fixture()
    def params(self, u):
        """Fixture for PlanarBase class atributes."""
        p = dict(
            weighting_function=lambda x, grouping, a=1: a * x,
            atomgroups=[u.atoms],
            normalization="number",
            grouping="atoms",
            bin_method="com",
            output="profile.dat",
        )
        return p

    def test_wrong_normalization(self, params):
        """Test a wrong normalization string."""
        with pytest.raises(ValueError, match="'foo' not supported"):
            params.update(normalization="foo")
            ProfileBase(**params)._prepare()

    def test_wrong_grouping(self, params):
        """Test a wrong grouping."""
        with pytest.raises(ValueError, match="'foo' is not a valid option"):
            params.update(grouping="foo")
            ProfileBase(**params)._prepare()

    def test_f_kwargs(self, params):
        """Test an extra keyword argument."""
        profile = ProfileBase(**params)
        params.update(f_kwargs={"a": 2})
        profile_scaled = ProfileBase(**params)

        assert 2 * profile.weighting_function(1) == profile_scaled.weighting_function(1)

    def test_output_name(self, params, tmpdir):
        """Test output name of save method."""
        params.update(output="foo.dat")
        profile = ProfileBase(**params)
        profile.results.bin_pos = np.zeros(10)
        profile.results.profile = np.zeros((10, 1))
        profile.results.dprofile = np.zeros((10, 1))
        profile.run = lambda x: x
        profile._index = 0

        with tmpdir.as_cwd():
            profile.save()
            assert os.path.exists(params["output"])

    def test_output(self, params, tmpdir):
        """Test output."""
        profile = ProfileBase(**params)
        profile.results.bin_pos = np.random.random(10)
        profile.results.profile = np.random.random((10, 1))
        profile.results.dprofile = np.random.random((10, 1))
        profile.run = lambda x: x
        profile._index = 0

        with tmpdir.as_cwd():
            profile.save()
            res_dens = np.loadtxt(profile.output)

        assert_allclose(profile.results.bin_pos, res_dens[:, 0], rtol=2)

        assert_allclose(profile.results.profile[:, 0], res_dens[:, 1], rtol=2)

        assert_allclose(profile.results.dprofile[:, 0], res_dens[:, 2], rtol=2)


class TestPlanarBaseChilds:
    """Tests for the AnalayseBase child classes."""

    ignored_parameters = ["multi_group", "atomgroups", "atomgroup", "wrap_compound"]

    @pytest.mark.parametrize("Member", find_cls_members(AnalysisBase, ["maicos"]))
    def test_parameters(self, Member):
        """Test if AnalysisBase paramaters exist in all modules."""
        base_sig = inspect.signature(AnalysisBase)
        mod_sig = inspect.signature(Member)

        for param in base_sig.parameters.values():
            if param.name in self.ignored_parameters:
                continue

            try:
                mod_sig.parameters[param.name]
            except KeyError:
                raise KeyError(f"{param.name} is not a parameter of {Member}!")
