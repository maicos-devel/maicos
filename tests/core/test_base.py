#!/usr/bin/env python
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
#
# Copyright (c) 2024 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the GNU Public Licence, v3 or any higher version
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for the base modules."""
import inspect
import logging
import sys
from pathlib import Path

import MDAnalysis as mda
import numpy as np
import pytest
from mdacli.libcli import find_cls_members
from MDAnalysis.analysis.base import Results
from MDAnalysis.core._get_readers import get_reader_for
from MDAnalysisTests.datafiles import DCD, PSF, TPR, XTC
from numpy.testing import assert_allclose, assert_equal

from maicos import DensityPlanar, _version
from maicos.core import AnalysisBase, AnalysisCollection, ProfileBase


sys.path.append(str(Path(__file__).parents[1]))

from data import WATER_GRO_NPT, WATER_TPR_NPT, WATER_TRR_NPT  # noqa: E402


class Output(AnalysisBase):
    """Class creating a file to check the output."""

    def __init__(self, atomgroup):
        super().__init__(
            atomgroup=atomgroup,
            unwrap=False,
            refgroup=None,
            jitter=0.0,
            wrap_compound="atoms",
            concfreq=0,
        )

    OUTPUT = "This is the output message of an analysis class."


class SubOutput(Output):
    """Class creating a file to check the output, but as a child class."""

    OUTPUT = "This is another output message from an inheriting class."


class FileModuleInput(AnalysisBase):
    """Class creating an output file to check the module input reporting."""

    def _single_frame(self):
        # Do nothing, but the run() methods needs to be called
        pass

    def __init__(self, atomgroup, test_input="some_default", refgroup=None):
        self._locals = locals()
        super().__init__(
            atomgroup=atomgroup,
            unwrap=False,
            refgroup=refgroup,
            jitter=0.0,
            wrap_compound="atoms",
            concfreq=0,
        )


class Series(AnalysisBase):
    """Class creating a random time series to check observables."""

    def __init__(self, atomgroup):
        super().__init__(
            atomgroup=atomgroup,
            unwrap=False,
            refgroup=None,
            jitter=0.0,
            wrap_compound="atoms",
            concfreq=0,
        )

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

    def __init__(self, atomgroup):
        super().__init__(
            atomgroup=atomgroup,
            unwrap=False,
            refgroup=None,
            jitter=0.0,
            wrap_compound="atoms",
            concfreq=0,
        )

    def _single_frame(self):
        self._obs.observable = self.data[self._frame_index]


class Conclude(AnalysisBase):
    """Class to test the _conclude method.

    A new file with a file name of the current analysis frame number is created every
    time the `_conclude` method is called.
    """

    def __init__(
        self,
        atomgroup,
        unwrap=False,
        refgroup=None,
        jitter=0.0,
        wrap_compound="atoms",
        concfreq=0,
        output_prefix="",
    ):
        super().__init__(
            atomgroup=atomgroup,
            unwrap=unwrap,
            refgroup=refgroup,
            jitter=jitter,
            wrap_compound=wrap_compound,
            concfreq=concfreq,
        )

        self.output_prefix = output_prefix

    def _prepare(self):
        self.conclude_count = 0

    def _single_frame(self):
        pass

    def _conclude(self):
        self.conclude_count += 1

    def save(self) -> None:
        """Save a file named after the current number of frames."""
        open(f"{self.output_prefix}out_{self._index}", "w").close()


class Test_AnalysisBase:
    """Tests for the Analysis base class."""

    @pytest.fixture()
    def ag(self):
        """Import MDA universe."""
        return mda.Universe(WATER_TPR_NPT, WATER_TRR_NPT, in_memory=True).atoms

    @pytest.fixture()
    def ag_single_frame(self):
        """Import MDA universe of single frame."""
        return mda.Universe(WATER_TPR_NPT, WATER_GRO_NPT, in_memory=True).atoms

    @pytest.fixture()
    def empty_ag(self):
        """Define an empty atomgroup."""
        u = mda.Universe.empty(0)
        return u.atoms

    @pytest.fixture()
    def u_no_cell(self):
        """An MDAnalysis universe without where the `dimensions` attribute is `None`."""
        return mda.Universe(PSF, DCD)

    def test_AnalysisBase(self, ag):
        """Test AnalysisBase."""
        a = AnalysisBase(
            atomgroup=ag,
            unwrap=False,
            refgroup=None,
            jitter=0.0,
            wrap_compound="atoms",
            concfreq=0,
        )

        assert a.atomgroup.n_atoms == ag.n_atoms
        assert a._trajectory == ag.universe.trajectory
        assert a._universe == ag.universe
        assert isinstance(a.results, Results)

    def test_empty_atomgroup(self, ag):
        """Test behaviour for empty atomgroup."""
        with pytest.raises(ValueError, match="not contain any atoms."):
            class_obj = AnalysisBase(
                atomgroup=ag.select_atoms("name foo"),
                unwrap=False,
                refgroup=None,
                jitter=0.0,
                wrap_compound="atoms",
                concfreq=0,
            )
            class_obj._prepare()

    def test_frame_data(self, ag):
        """Test the calculation of the frame, sums, mean and sems results dicts."""
        ana = Series(atomgroup=ag)
        ana.run()

        assert_allclose(ana.sums.observable, np.sum(ana.series))
        assert_allclose(ana.means.observable, np.mean(ana.series))
        assert_allclose(ana.sems.observable, np.std(ana.series) / np.sqrt(ana.n_frames))

    def test_output_message(self, ag, monkeypatch, tmp_path):
        """Test the output message of modules."""
        monkeypatch.chdir(tmp_path)

        data = np.random.rand(100, 2)
        ana = Output(ag)
        ana._index = 1
        sub_ana = SubOutput(ag)
        sub_ana._index = 1

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

    def test_module_input(self, ag, monkeypatch, tmp_path):
        """Test the module input reporting."""
        monkeypatch.chdir(tmp_path)

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
        assert "atomgroup=<AtomGroup>" in open("test_refgroup.dat").read()

        # Test if the default value of the test_input parameter is written
        ana = FileModuleInput(ag)
        ana.run()
        ana.savetxt("test_default.dat", np.random.rand(10, 2))
        assert "test_input='some_default'" in open("test_default.dat").read()
        assert "refgroup=None" in open("test_default.dat").read()
        print(open("test_default.dat").read())
        assert (
            ".run(start=None, stop=None, step=None, frames=None, verbose=None, "
            "progressbar_kwargs=None)" in open("test_default.dat").read()
        )

        # Test if the set test_input parameter is written correctly
        ana = FileModuleInput(ag, test_input="some_other_value")
        ana.run()
        ana.savetxt("test_nondefault.dat", np.random.rand(10, 2))
        assert "test_input='some_other_value'" in open("test_nondefault.dat").read()

        ana.run(step=2, stop=7, start=5, verbose=True)
        ana.savetxt("test_run.dat", np.random.rand(10, 2))
        assert (
            ".run(start=5, stop=7, step=2, frames=None, verbose=True, "
            "progressbar_kwargs=None)" in open("test_run.dat").read()
        )

    @pytest.mark.parametrize(
        "concfreq, files",
        [(0, []), (40, ["out_40", "out_80", "out_101"]), (100, ["out_100", "out_101"])],
    )
    def test_conclude_multi_frame(self, ag, monkeypatch, tmp_path, concfreq, files):
        """Test the conclude and save methods for multi frame trajectories."""
        monkeypatch.chdir(tmp_path)

        conclude = Conclude(ag, concfreq=concfreq)
        conclude.run()
        # check that all expected files have been written
        if concfreq != 0:
            for file in files:
                assert Path(file).exists()
        else:
            assert len(list(tmp_path.iterdir())) == 0
        # check that the _conclude method is running
        # the expected number of times
        if concfreq != 0:
            conclude_count = np.ceil(conclude.n_frames / concfreq)
        else:
            conclude_count = 1
        assert conclude.conclude_count == conclude_count
        # check that no more files than the expected
        # ones have been written
        assert len(files) == len(list(tmp_path.iterdir()))

    @pytest.mark.parametrize("concfreq, file", [(0, []), (50, ["out_1"])])
    def test_conclude_single_frame(
        self, ag_single_frame, monkeypatch, tmp_path, concfreq, file
    ):
        """Test the conclude and save methods for single frame trajectories."""
        monkeypatch.chdir(tmp_path)

        conclude = Conclude(ag_single_frame, concfreq=concfreq)
        conclude.run()
        if concfreq != 0:
            assert Path(file[0]).exists()
        # check that no extra files are written
        if concfreq != 0:
            assert len(list(tmp_path.iterdir())) == 1
        else:
            assert len(list(tmp_path.iterdir())) == 0
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

    def test_refgroup_nomass(self, caplog):
        """Test warning and succesful ref_weights"""
        u = mda.Universe.empty(2)
        positions = np.array([[1, 2, 3], [3, 2, 1]])
        u.trajectory = get_reader_for(positions)(positions, order="fac", n_atoms=2)

        for ts in u.trajectory:
            ts.dimensions = np.array([4, 4, 1, 90, 90, 90])
        ana_obj = AnalysisBase(
            atomgroup=u.atoms,
            jitter=0.0,
            wrap_compound="atoms",
            concfreq=0,
            refgroup=u.atoms,
            unwrap=True,
        )
        ana_obj._setup_frames(ana_obj._trajectory)
        ana_obj._call_prepare()

        assert_equal(ana_obj.ref_weights, np.ones_like(u.atoms))
        assert (
            "No masses available in refgroup, falling back to center of geometry"
            in caplog.text
        )

    def test_empty_refgroup(self, ag, empty_ag):
        """Test behaviour for empty refgroup."""
        with pytest.raises(ValueError, match="not contain any atoms."):
            class_obj = AnalysisBase(
                atomgroup=ag,
                refgroup=empty_ag,
                unwrap=False,
                jitter=0.0,
                wrap_compound="atoms",
                concfreq=0,
            )
            class_obj._prepare()

    def test_unwrap(self, ag):
        """Unwrap test for logic only; Actual test in TestProfilePlanarBase."""
        class_obj = Conclude(ag, unwrap=True)
        class_obj.run(stop=1)

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
        ana_obj = AnalysisBase(
            atomgroup=ag,
            unwrap=False,
            refgroup=None,
            jitter=0.0,
            wrap_compound="atoms",
            concfreq=0,
        )

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
        ana_obj = AnalysisBase(
            atomgroup=ag,
            unwrap=False,
            refgroup=None,
            jitter=0.0,
            wrap_compound="atoms",
            concfreq=0,
        )

        # Create empty methods for allowing the run method to succeed.
        ana_obj._prepare = lambda: None
        ana_obj._single_frame = lambda: None
        ana_obj._conclude = lambda: None

        ana_obj.n_bins = 10

        caplog.set_level(logging.INFO)
        ana_obj.run(stop=1)

        assert "Using 10 bins." in [rec.message for rec in caplog.records]

    def test_info_log(self, ag, caplog):
        """Test that logger infos are printed."""
        ana_obj = AnalysisBase(
            atomgroup=ag,
            unwrap=False,
            refgroup=None,
            jitter=0.0,
            wrap_compound="atoms",
            concfreq=0,
        )

        # Create empty methods for allowing the run method to succeed.
        ana_obj._prepare = lambda: None
        ana_obj._single_frame = lambda: None
        ana_obj._conclude = lambda: None

        caplog.set_level(logging.INFO)
        ana_obj.run(stop=1)

        messages = [rec.message for rec in caplog.records]
        assert "Starting preparation" in messages
        assert "Starting analysis loop over 1 trajectory frames." in messages

    def test_unwrap_atoms(self, ag, caplog):
        """Test that unwrap is always False for `wrap_compound="atoms"`."""
        caplog.set_level(logging.WARN)
        profile = AnalysisBase(
            atomgroup=ag,
            unwrap=True,
            wrap_compound="atoms",
            refgroup=None,
            jitter=0.0,
            concfreq=0,
        )

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
        dens = DensityPlanar(ag_single_frame, bin_width=1e-6, jitter=0.0).run()

        dens_jitter = DensityPlanar(ag_single_frame, bin_width=1e-6, jitter=0.01).run()

        # Make sure that the integral over the jittered profile is the same as
        # the non-jittered one (up to numerical precision)
        assert_allclose(dens_jitter.results.profile.sum(), dens.results.profile.sum())

        # Check that unjittered profile has peaks due to rounding (multiple
        # atoms per bin) and jittered one does not.
        assert dens.results.profile.nonzero()[0].size != ag_single_frame.n_atoms
        assert dens_jitter.results.profile.nonzero()[0].size == ag_single_frame.n_atoms

    def test_no_dimensions_unwrap_error(self, u_no_cell):
        """Test that an error is raised if `unwrap=True` but no cell is present."""
        match = "Universe does not have `dimensions` and can't be unwrapped!"
        with pytest.raises(ValueError, match=match):
            class_obj = AnalysisBase(
                u_no_cell.atoms,
                unwrap=True,
                refgroup=None,
                jitter=0.0,
                wrap_compound="atoms",
                concfreq=0,
            )
            class_obj._prepare()

    def test_no_dimensions_run(self, u_no_cell):
        """Test that an analysis can be run for a universe without cell information."""
        class_obj = Conclude(u_no_cell.atoms)
        class_obj.run(stop=1)


class TestAnalysisCollection:
    """Test functions for the AnalysisCollection class."""

    @pytest.fixture
    def u(self):
        """An MDAnalysis universe."""
        return mda.Universe(PSF, DCD)

    def test_experimental_warning(self, u):
        """Test that the experimental warning is displayed."""
        ana_1 = Conclude(u.atoms)

        with pytest.warns(UserWarning, match="still experimental"):
            AnalysisCollection(ana_1)

    def test_run(self, u):
        """Smoke test if the class can be run."""
        ana_1 = Conclude(u.atoms)
        ana_2 = Conclude(u.atoms)

        collection = AnalysisCollection(ana_1, ana_2)
        collection.run()

        assert ana_1.results is not None
        assert ana_2.results is not None

    def test_trajectory_manipulation(self, u):
        """Test that the timestep is the same for each analysis class."""

        class CustomAnalysis(AnalysisBase):
            """Custom class that is shifting positions in every step by 10."""

            def __init__(self, atomgroup):
                super().__init__(
                    atomgroup=atomgroup,
                    unwrap=False,
                    refgroup=None,
                    jitter=0.0,
                    wrap_compound="atoms",
                    concfreq=0,
                )

            def _prepare(self):
                pass

            def _single_frame(self):
                self._ts.positions += 10
                self.ref_pos = self._ts.positions.copy()[0, 0]

            def _conlude(self):
                pass

        ana_1 = CustomAnalysis(u.atoms)
        ana_2 = CustomAnalysis(u.atoms)

        collection = AnalysisCollection(ana_1, ana_2)
        collection.run(frames=[0])

        assert ana_2.ref_pos == ana_1.ref_pos

    def test_inconsistent_trajectory(self, u):
        """Test error raise if two analysis objects have a different trajectory."""
        v = mda.Universe(TPR, XTC)

        with pytest.raises(ValueError, match="`analysis_instances` do not have the"):
            AnalysisCollection(Conclude(u.atoms), Conclude(v.atoms))

    def test_no_base_child(self, u):
        """Test error raise if an object is not a AnalyisBase child."""

        class CustomAnalysis:
            def __init__(self, trajectory):
                self._trajectory = trajectory

        # Create collection for common trajectory loop with inconsistent trajectory
        with pytest.raises(AttributeError, match="not a child of `AnalysisBase`"):
            AnalysisCollection(CustomAnalysis(u.trajectory))

    def test_save(self, u, monkeypatch, tmp_path):
        """Test that all results can be written to disk with one command."""
        monkeypatch.chdir(tmp_path)

        ana_1 = Conclude(u.atoms, output_prefix="ana1")
        ana_2 = Conclude(u.atoms, output_prefix="ana2")

        collection = AnalysisCollection(ana_1, ana_2)
        collection.run(stop=1)
        collection.save()

        assert Path(f"{ana_1.output_prefix}out_{ana_1._index}").exists()
        assert Path(f"{ana_2.output_prefix}out_{ana_2._index}").exists()

    def test_save_warning(self, u, monkeypatch, tmp_path):
        """Test that a warning is issued in an instance has no `save` method."""
        monkeypatch.chdir(tmp_path)

        ana_1 = Conclude(u.atoms, output_prefix="ana1")
        ana_2 = AnalysisBase(
            atomgroup=u.atoms,
            unwrap=False,
            refgroup=None,
            jitter=0.0,
            wrap_compound="atoms",
            concfreq=0,
        )
        # Create empty methods for allowing the run method to succeed.
        ana_2._prepare = lambda: None
        ana_2._single_frame = lambda: None
        ana_2._conclude = lambda: None

        collection = AnalysisCollection(ana_1, ana_2)
        collection.run(stop=1)
        with pytest.warns(UserWarning, match=r"has no save\(\) method"):
            collection.save()

        assert Path(f"{ana_1.output_prefix}out_{ana_1._index}").exists()


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
            weighting_function_kwargs=None,
            atomgroup=u.atoms,
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

    def test_weighting_function_kwargs(self, params):
        """Test an extra keyword argument."""
        profile = ProfileBase(**params)
        params.update(weighting_function_kwargs={"a": 2})
        profile_scaled = ProfileBase(**params)

        assert 2 * profile.weighting_function(1) == profile_scaled.weighting_function(1)

    def test_output_name(self, params, monkeypatch, tmp_path):
        """Test output name of save method."""
        monkeypatch.chdir(tmp_path)

        params.update(output="foo.dat")
        profile = ProfileBase(**params)
        profile.results.bin_pos = np.zeros(10)
        profile.results.profile = np.zeros(10)
        profile.results.dprofile = np.zeros(10)
        profile.run = lambda x: x
        profile._index = 0

        profile.save()
        assert Path(params["output"]).exists()

    def test_output(self, params, monkeypatch, tmp_path):
        """Test output."""
        monkeypatch.chdir(tmp_path)

        """Test output."""
        profile = ProfileBase(**params)
        profile.results.bin_pos = np.random.random(10)
        profile.results.profile = np.random.random(10)
        profile.results.dprofile = np.random.random(10)
        profile.run = lambda x: x
        profile._index = 0

        profile.save()
        res_dens = np.loadtxt(profile.output)

        assert_allclose(profile.results.bin_pos, res_dens[:, 0], rtol=2)

        assert_allclose(profile.results.profile, res_dens[:, 1], rtol=2)

        assert_allclose(profile.results.dprofile, res_dens[:, 2], rtol=2)


class TestPlanarBaseChilds:
    """Tests for the AnalayseBase child classes."""

    ignored_parameters = ["atomgroup", "wrap_compound"]

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
