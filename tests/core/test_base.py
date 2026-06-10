#!/usr/bin/env python
#
# Copyright (c) 2026 Authors and contributors
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
from MDAnalysisTests.core.util import UnWrapUniverse
from MDAnalysisTests.datafiles import DCD, PSF, TPR, XTC
from numpy.testing import assert_allclose, assert_equal

from maicos import DensityPlanar, __version__
from maicos.core import AnalysisBase, AnalysisCollection, ProfileBase
from maicos.lib.util import make_pair_key, joint_pop

sys.path.append(str(Path(__file__).parents[1]))

from data import (  # noqa: E402
    DIPOLE_GRO,
    DIPOLE_ITP,
    WATER_GRO_NPT,
    WATER_TPR_NPT,
    WATER_TRR_NPT,
)


class Output(AnalysisBase):
    """Class creating a file to check the output."""

    def __init__(self, atomgroup):
        super().__init__(
            atomgroup=atomgroup,
            unwrap=False,
            pack=True,
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
            pack=True,
            refgroup=refgroup,
            jitter=0.0,
            wrap_compound="atoms",
            concfreq=0,
        )


class SingularSeries(AnalysisBase):
    """Class creating a time series with one observable per frame."""

    def __init__(self, atomgroup):
        super().__init__(
            atomgroup=atomgroup,
            unwrap=False,
            pack=True,
            refgroup=None,
            jitter=0.0,
            wrap_compound="atoms",
            concfreq=0,
        )

    def _prepare(self):
        self.series = np.random.rand(self.n_frames)

    def _single_frame(self):
        self._obs.observable = self.series[self._frame_index]

class DebugSeries(AnalysisBase):
    """Class creating a time series with one observable per frame."""

    def __init__(self, atomgroup):
        super().__init__(
            atomgroup=atomgroup,
            unwrap=False,
            pack=True,
            refgroup=None,
            jitter=0.0,
            wrap_compound="atoms",
            concfreq=0,
        )

    def _prepare(self):
        self.series = np.arange(self.n_frames, dtype=float)

    def _single_frame(self):
        self._obs.observable = self.series[self._frame_index]



class MultipleSeries(AnalysisBase):
    """Class creating a time series with multiple observables per frame."""

    def __init__(self, atomgroup):
        super().__init__(
            atomgroup=atomgroup,
            unwrap=False,
            pack=True,
            refgroup=None,
            jitter=0.0,
            wrap_compound="atoms",
            concfreq=0,
        )

    def _prepare(self):
        self.population = np.random.randint(1, 10, self.n_frames)
        self.series = []
        for i in range(len(self.population)):
            self.series.append(np.random.rand(self.population[i]))

    def _single_frame(self):
        self._obs.observable = np.mean(self.series[self._frame_index])
        self._var.observable = np.var(self.series[self._frame_index])
        self._pop.observable = self.population[self._frame_index]


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
            pack=True,
            refgroup=None,
            jitter=0.0,
            wrap_compound="atoms",
            concfreq=0,
        )

    def _single_frame(self):
        self._obs.observable = self.data[self._frame_index]


class CorrelatedSeries(AnalysisBase):
    # TODO @quewakira: The name can be misleading (could be time correlation)
    """Class emitting several correlated observables per frame.

    Observables (one sample per frame):
    - ``x``     : scalar, drawn at random
    - ``y``     : scalar, linearly correlated with ``x``
    - ``prof``  : shape (3,) array, correlated with ``x``
    - ``other`` : shape (2,) array (does not broadcast against ``prof``)
    """

    _compute_covariance = [
        {"x", "y"},
        {"x", "prof"},
        {"x", "other"},
        {"y", "prof"},
        {"y", "other"},
        {"prof", "other"},
    ]

    def __init__(self, atomgroup):
        super().__init__(
            atomgroup=atomgroup,
            unwrap=False,
            pack=True,
            refgroup=None,
            jitter=0.0,
            wrap_compound="atoms",
            concfreq=0,
        )

    def _prepare(self):
        n = self.n_frames
        self.x = np.random.rand(n) * 10 - 5
        self.y = 2.5 * self.x + np.random.rand(n)
        self.prof = self.x[:, None] * np.array([1.0, -2.0, 0.5]) + np.random.rand(n, 3)
        self.other = np.random.rand(n, 2)

    def _single_frame(self):
        i = self._frame_index
        self._obs.x = self.x[i]
        self._obs.y = self.y[i]
        self._obs.prof = self.prof[i]
        self._obs.other = self.other[i]


class WeightedSeries(AnalysisBase):
    """Class with a single-sample and a weighted-population observable.

    ``single`` (shape (3,), one sample per frame) and ``weighted`` (shape (3,)
    with a per-bin sample count) broadcast in shape but are *not* co-sampled, so
    their covariance must not be tracked.
    """

    _compute_covariance = [{"single", "weighted"}]

    def __init__(self, atomgroup):
        super().__init__(
            atomgroup=atomgroup,
            unwrap=False,
            pack=True,
            refgroup=None,
            jitter=0.0,
            wrap_compound="atoms",
            concfreq=0,
        )

    def _prepare(self):
        self.rng = np.random.default_rng(0)

    def _single_frame(self):
        self._obs.single = self.rng.random(3)
        self._obs.weighted = self.rng.random(3)
        self._var.weighted = self.rng.random(3)
        self._pop.weighted = self.rng.integers(2, 10, 3)


class MultiSampleSeries(AnalysisBase):
    """Two co-sampled observables carrying multiple samples per frame.

    Each frame draws a variable number of correlated ``(x, y)`` samples per bin
    (some frames empty) and reports the per-frame mean, within-frame variance,
    within-frame covariance, and population. The streamed co-moment must equal
    the batch co-moment over all individual samples. With ``n_bins == 1`` the
    observables are scalars (fallback path); otherwise they are arrays of shape
    ``(n_bins,)`` (vectorized block path).
    """

    _compute_covariance = [{"x", "y"}]

    def __init__(self, atomgroup, n_bins=3, seed=0):
        self._n_bins = n_bins
        self._seed = seed
        super().__init__(
            atomgroup=atomgroup,
            unwrap=False,
            pack=True,
            refgroup=None,
            jitter=0.0,
            wrap_compound="atoms",
            concfreq=0,
        )

    def _prepare(self):
        rng = np.random.default_rng(self._seed)
        chol = np.array([[1.0, 0.0], [0.7, 0.6]])
        # samples[frame][bin] -> (k, 2) array; batch[bin] -> list of arrays.
        self._samples = []
        self.batch = [[] for _ in range(self._n_bins)]
        for _ in range(self.n_frames):
            per_bin = []
            for m in range(self._n_bins):
                k = int(rng.integers(0, 6))
                s = rng.standard_normal((k, 2)) @ chol.T + np.array([m, -m])
                per_bin.append(s)
                if k:
                    self.batch[m].append(s)
            self._samples.append(per_bin)

    def _single_frame(self):
        per_bin = self._samples[self._frame_index]
        scalar = self._n_bins == 1
        xm = np.full(self._n_bins, np.nan)
        ym = np.full(self._n_bins, np.nan)
        vx = np.zeros(self._n_bins)
        vy = np.zeros(self._n_bins)
        cxy = np.zeros(self._n_bins)
        pop = np.zeros(self._n_bins, dtype=int)
        for m, s in enumerate(per_bin):
            pop[m] = len(s)
            if len(s) == 0:
                continue
            x, y = s[:, 0], s[:, 1]
            xm[m], ym[m] = x.mean(), y.mean()
            vx[m], vy[m] = x.var(), y.var()
            cxy[m] = ((x - x.mean()) * (y - y.mean())).mean()

        def maybe_scalar(arr):
            return arr[0] if scalar else arr

        self._obs.x = maybe_scalar(xm)
        self._obs.y = maybe_scalar(ym)
        self._var.x = maybe_scalar(vx)
        self._var.y = maybe_scalar(vy)
        self._pop.x = int(pop[0]) if scalar else pop.copy()
        self._pop.y = int(pop[0]) if scalar else pop.copy()
        self._cov[make_pair_key("x", "y")] = maybe_scalar(cxy)
        print("cov in single frame")
        print(self._cov)

    def batch_comoment(self):
        """Co-moment ``sum((x - x.mean())(y - y.mean()))`` over all samples."""
        out = np.zeros(self._n_bins)
        for m in range(self._n_bins):
            alls = np.concatenate(self.batch[m], axis=0)
            x, y = alls[:, 0], alls[:, 1]
            out[m] = ((x - x.mean()) * (y - y.mean())).sum()
        return out[0] if self._n_bins == 1 else out


class Conclude(AnalysisBase):
    """Class to test the _conclude method.

    A new file with a file name of the current analysis frame number is created every
    time the ``_conclude`` method is called.
    """

    def __init__(
        self,
        atomgroup,
        unwrap=False,
        pack=False,
        refgroup=None,
        jitter=0.0,
        wrap_compound="atoms",
        concfreq=0,
        output_prefix="",
    ):
        super().__init__(
            atomgroup=atomgroup,
            unwrap=unwrap,
            pack=pack,
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
        Path(f"{self.output_prefix}out_{self._index}").open(mode="w").close()


class Test_AnalysisBase:
    """Tests for the Analysis base class."""

    @pytest.fixture
    def ag(self):
        """Import MDA universe."""
        return mda.Universe(WATER_TPR_NPT, WATER_TRR_NPT, in_memory=True).atoms

    @pytest.fixture
    def ag_single_frame(self):
        """Import MDA universe of single frame."""
        return mda.Universe(WATER_TPR_NPT, WATER_GRO_NPT, in_memory=True).atoms

    @pytest.fixture
    def empty_ag(self):
        """Define an empty atomgroup."""
        u = mda.Universe.empty(0)
        return u.atoms

    @pytest.fixture
    def u_no_cell(self):
        """An MDAnalysis universe without where the `dimensions` attribute is `None`."""
        return mda.Universe(PSF, DCD)

    def test_trajectory_starting_frame(self, ag):
        """Test that the trajectory is rewound to the first frame.

        The AnalysisBase should rewind the trajectory so that the _prepare method
        consistently sees the first frame.
        """
        # We select a frame from the middle, since we want to rewind to the first frame
        # of a sliced trajectory.
        params = dict(
            unwrap=False,
            pack=True,
            refgroup=None,
            jitter=0.0,
            wrap_compound="atoms",
            concfreq=0,
        )
        positions = ag.universe.trajectory[5].positions.copy()

        ana_obj = AnalysisBase(ag, **params)

        def _prepare(self):
            self.check_frame_data = self.atomgroup.positions

        # TODO(@hejamu): Create a Subclass of AnalysisBase
        # with stub methods to reuse in other tests
        ana_obj._prepare = lambda: _prepare(ana_obj)
        ana_obj._single_frame = lambda: None
        ana_obj._conclude = lambda: None

        ana_obj.run(start=5)

        # check_frame_data should contain the positions of the first frame _prepare saw
        assert np.all(ana_obj.check_frame_data == positions)

    def test_triclinic_warning(self, ag, caplog):
        """Test that the triclinic warning is displayed.

        Run this test first since warning will be only emmitted once.
        """
        assert len(ag.universe.trajectory) > 1  # ensure multi-frame trajectory
        for ts in ag.universe.trajectory:
            ts.dimensions = np.array([30, 30, 30, 70, 80, 100])
        conclude = Conclude(ag)
        with caplog.at_level(logging.WARNING):
            conclude.run()

        warnings = [rec.message for rec in caplog.records]
        assert len(warnings) == 1

        match = (
            "The trajectory contains box-dimensions that are not orthorhombic! "
            "Continue with caution."
        )
        assert match in warnings[0]

        caplog.clear()

        # Do the crosscheck that no warning is emitted for orthorhombic boxes
        for ts in ag.universe.trajectory:
            ts.dimensions = np.array([30, 30, 30, 90, 90, 90])
        conclude = Conclude(ag)
        with caplog.at_level(logging.WARNING):
            conclude.run()

        warnings = [rec.message for rec in caplog.records]
        assert len(warnings) == 0

    def test_triclinic_wrapping(self):
        """Test that atoms are wrapped into the orthorhombic bounding box.

        An atom placed at (2, 6, 5) in a triclinic box with gamma=45 degrees
        is inside the rectangular domain but outside the triclinic cell.
        Wrapping with the triclinic cell alone shifts it to x=12, outside
        the orthorhombic bounding box. Another wrap puts it again into the
        orthorhombic box so all atoms remain inside the analysis domain:

            Original (i.e. GROMACS)  After triclinic         After orthorhombic
            positions                wrap only (broken)      wrap

            y     b                  y     b                  y     b
            |    /       /           |    /       /           |    /       /
            |  */->     /            |   /      */->          |  */->     /
            |  /       /             |  /       /             |  /       /
            | /       /              | /       /              | /       /
            +---------x = a          +---------x = a          +---------x = a
        """
        from maicos.lib.util import triclinic_to_orthorhombic

        dimensions = [10, 10, 10, 90, 90, 45]
        ortho_box = triclinic_to_orthorhombic(np.array(dimensions, dtype=float))

        template = mda.Universe(DIPOLE_ITP, DIPOLE_GRO, topology_format="itp")
        dipole = template.copy()
        dipole.atoms.translate([2, 6, 5])
        dipole.atoms.residues.molnums = [0]
        u = mda.Merge(dipole.atoms)
        u.dimensions = dimensions

        conclude = Conclude(u.atoms, pack=True, wrap_compound="atoms")
        conclude.run()

        positions = u.atoms.positions
        assert np.all(positions >= 0)
        assert np.all(positions[:, 0] < ortho_box[0])
        assert np.all(positions[:, 1] < ortho_box[1])
        assert np.all(positions[:, 2] < ortho_box[2])

    def test_AnalysisBase(self, ag):
        """Test AnalysisBase."""
        a = AnalysisBase(
            atomgroup=ag,
            unwrap=False,
            pack=True,
            refgroup=None,
            jitter=0.0,
            wrap_compound="atoms",
            concfreq=0,
        )

        assert a.atomgroup.n_atoms == ag.n_atoms
        assert a._trajectory == ag.universe.trajectory
        assert a._universe == ag.universe
        assert isinstance(a.results, Results)

    def test_invalid_wrap_compound(self, ag):
        """Test that an invalid wrap_compound raises ValueError."""
        with pytest.raises(ValueError, match="Unrecognized `wrap_compound`"):
            AnalysisBase(
                atomgroup=ag,
                unwrap=False,
                pack=True,
                refgroup=None,
                jitter=0.0,
                wrap_compound="invalid_compound",
                concfreq=0,
            )

    def test_empty_atomgroup(self, ag):
        """Test behaviour for empty atomgroup."""
        with pytest.raises(ValueError, match="not contain any atoms."):
            AnalysisBase(
                atomgroup=ag.select_atoms("name foo"),
                unwrap=False,
                pack=True,
                refgroup=None,
                jitter=0.0,
                wrap_compound="atoms",
                concfreq=0,
            )

    def test_frame_data(self, ag):
        """Test the calculation of the frame, sums, mean and sems results dicts."""
        ana = DebugSeries(atomgroup=ag)
        ana.n_frames = 2
        ana.run()

        assert_allclose(ana.means.observable, np.mean(ana.series))
        assert_allclose(ana.sems.observable, np.std(ana.series) / np.sqrt(ana.n_frames))
        assert_allclose(ana.sums.observable, np.sum(ana.series))

        ana = MultipleSeries(atomgroup=ag)
        ana.run()
        raw_series = np.concatenate(ana.series)

        assert_allclose(ana.means.observable, np.mean(raw_series), rtol=1e-5)
        assert_allclose(ana.sums.observable, np.sum(raw_series), rtol=1e-5)
        assert_allclose(
            ana.sems.observable,
            np.std(raw_series) / np.sqrt(len(raw_series)),
            rtol=1e-5,
        )

    def test_output_message(self, ag, monkeypatch, tmp_path):
        """Test the output message of modules."""
        monkeypatch.chdir(tmp_path)

        data = np.random.rand(100, 2)
        ana = Output(ag)
        ana._index = 1
        sub_ana = SubOutput(ag)
        sub_ana._index = 1

        # Simple check if a single message gets written to the output file
        ana.savetxt("foo.dat", data, columns=["First", "Second"])

        with Path("foo.dat").open() as f:
            assert ana.OUTPUT in f.read()

        # More elaborate check to find out if output messages of subclasses
        # get written to the file in the right order.
        sub_ana.savetxt("foo2.dat", data, columns=["First", "Second"])

        with Path("foo2.dat").open() as f:
            foo = f.readlines()

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

        with Path("test.dat").open() as f:
            assert "Module input:    FileModuleInput(" in f.read()

        # Test if the refgroup name is written correctly
        ana = FileModuleInput(ag, refgroup=ag)
        ana.run()
        ana.savetxt("test_refgroup.dat", np.random.rand(10, 2))

        with Path("test_refgroup.dat").open() as f:
            assert "refgroup=<AtomGroup>" in f.read()
        with Path("test_refgroup.dat").open() as f:
            assert "atomgroup=<AtomGroup>" in f.read()

        # Test if the default value of the test_input parameter is written
        ana = FileModuleInput(ag)
        ana.run()
        ana.savetxt("test_default.dat", np.random.rand(10, 2))

        with Path("test_default.dat").open() as f:
            assert "test_input='some_default'" in f.read()

        with Path("test_default.dat").open() as f:
            assert "refgroup=None" in f.read()

        with Path("test_default.dat").open() as f:
            assert (
                ".run(start=None, stop=None, step=None, frames=None, verbose=None, "
                "progressbar_kwargs=None)" in f.read()
            )

        # Test if the set test_input parameter is written correctly
        ana = FileModuleInput(ag, test_input="some_other_value")
        ana.run()
        ana.savetxt("test_nondefault.dat", np.random.rand(10, 2))

        with Path("test_nondefault.dat").open() as f:
            assert "test_input='some_other_value'" in f.read()

        ana.run(step=2, stop=7, start=5, verbose=True)
        ana.savetxt("test_run.dat", np.random.rand(10, 2))
        with Path("test_run.dat").open() as f:
            assert (
                ".run(start=5, stop=7, step=2, frames=None, verbose=True, "
                "progressbar_kwargs=None)" in f.read()
            )

    def test_savetxt_warns_on_missing_extension(self, ag, monkeypatch, tmp_path):
        """Savetxt warns when fname lacks the .dat extension and appends it."""
        monkeypatch.chdir(tmp_path)
        ana = Output(ag)
        ana._index = 1
        with pytest.warns(UserWarning, match=r"\.dat"):
            ana.savetxt("missing_ext", np.random.rand(10, 2))
        assert Path("missing_ext.dat").exists()

    @pytest.mark.parametrize(
        ("concfreq", "files"),
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
        conclude_count = np.ceil(conclude.n_frames / concfreq) if concfreq != 0 else 1
        assert conclude.conclude_count == conclude_count
        # check that no more files than the expected
        # ones have been written
        assert len(files) == len(list(tmp_path.iterdir()))

    @pytest.mark.parametrize(("concfreq", "file"), [(0, []), (50, ["out_1"])])
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
        class_obj = Conclude(ag, refgroup=refgroup, unwrap=True, pack=True)
        class_obj.run(stop=1)

        assert_allclose(
            refgroup.center_of_mass(), ag.universe.dimensions[:3] / 2, rtol=1e-01
        )

    def test_refgroup_nomass(self, caplog):
        """Test warning and succesful ref_weights."""
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
            pack=True,
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
            AnalysisBase(
                atomgroup=ag,
                refgroup=empty_ag,
                unwrap=False,
                pack=True,
                jitter=0.0,
                wrap_compound="atoms",
                concfreq=0,
            )

    @pytest.mark.parametrize(
        ("unwrap", "pack"), [(True, True), (True, False), (False, True), (False, False)]
    )
    def test_unwrap_pack(self, unwrap, pack):
        """Test the pack and unwrap flag."""
        ag = UnWrapUniverse().atoms
        class_obj = Conclude(ag, unwrap=unwrap, pack=pack, wrap_compound="residues")
        class_obj.run(stop=1)

        ag_ref = UnWrapUniverse().atoms
        if unwrap:
            ag_ref.unwrap(compound="residues")
        if pack:
            ag_ref.wrap(compound="residues")

        assert_allclose(ag.positions, ag_ref.positions)

        ag_ref = UnWrapUniverse().atoms
        if unwrap:
            ag_ref.unwrap(compound="residues")
        if pack:
            ag_ref.wrap(compound="residues")

        assert_allclose(ag.positions, ag_ref.positions)

    @pytest.mark.parametrize(
        ("data", "result"),
        [([1, 2], 1.5), ([float(1), float(2)], 1.5), ([[1], [2]], 1.5)],
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
        error_msg = "Observable 'observable' has an incompatible type."
        with pytest.raises(TypeError, match=error_msg):
            class_obj.run(stop=2)

    def test_banner(self, ag, caplog):
        """Test whether AnalysisBase prints the MAICoS banner."""
        ana_obj = AnalysisBase(
            atomgroup=ag,
            unwrap=False,
            pack=True,
            refgroup=None,
            jitter=0.0,
            wrap_compound="atoms",
            concfreq=0,
        )

        # Create empty methods for allowing the run method to succeed.
        ana_obj._prepare = lambda: None
        ana_obj._single_frame = lambda: None
        ana_obj._conclude = lambda: None

        ana_obj.run(stop=1, verbose=True)

        assert (
            r"#    ()----()     |  \/  |     /\     |_   _|  / ____|          / ____|"
            in caplog.text
        )
        assert __version__ in caplog.text

    @pytest.mark.parametrize(
        "typefunc",
        [
            int,
            float,
            np.float32,
            np.float64,
            np.int32,
            np.int64,
        ],
    )
    def test_bin_width(self, ag, typefunc):
        """Test if various types for bin_wdith are supported."""
        ana_obj = AnalysisBase(
            atomgroup=ag,
            unwrap=False,
            pack=True,
            refgroup=None,
            jitter=0.0,
            wrap_compound="atoms",
            concfreq=0,
        )

        ana_obj._bin_width = typefunc(1.0)

        ana_obj._prepare = lambda: None
        ana_obj._single_frame = lambda: None
        ana_obj._conclude = lambda: None

        ana_obj.run(stop=1)
        assert ana_obj._bin_width == typefunc(1.0)

    def test_bin_width_not_a_number(self, ag):
        """Test error raise that bin_width is not a number."""
        ana_obj = AnalysisBase(
            atomgroup=ag,
            unwrap=False,
            pack=True,
            refgroup=None,
            jitter=0.0,
            wrap_compound="atoms",
            concfreq=0,
        )

        ana_obj._bin_width = "x"

        match = "Binwidth must be a real number but is of type 'str'."
        with pytest.raises(TypeError, match=match):
            ana_obj.run()

    @pytest.mark.parametrize("bin_width", [0, -0.5])
    def test_negative_bin_width(self, ag, bin_width):
        """Test error raise for negative bin_width."""
        ana_obj = AnalysisBase(
            atomgroup=ag,
            unwrap=False,
            pack=True,
            refgroup=None,
            jitter=0.0,
            wrap_compound="atoms",
            concfreq=0,
        )

        ana_obj._bin_width = bin_width

        match = rf"Binwidth must be a positive number but is {bin_width}."
        with pytest.raises(ValueError, match=match):
            ana_obj.run()

    def test_n_bins(self, ag, caplog):
        """Test `n_bins` logger info."""
        ana_obj = AnalysisBase(
            atomgroup=ag,
            unwrap=False,
            pack=True,
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

        ana_obj.run(stop=1, verbose=True)

        assert "Using 10 bins." in caplog.text

    def test_info_log_verbose(self, ag, caplog):
        """Test that logger infos are printed."""
        ana_obj = AnalysisBase(
            atomgroup=ag,
            unwrap=False,
            pack=True,
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
        ana_obj.run(stop=1, verbose=True)

        analysis_msg = "Analysing 1 trajectory frames."

        # INFO log messages should be in the logger when verbose=True
        assert analysis_msg in caplog.text

    def test_verbose_multiple_runs(self, ag, caplog):
        """Test that verbosity is set correctly across multiple runs.

        The run method can be executed multiple times with different values for
        verbose. This test ensures that the logging level is correctly adjusted
        each time.
        """
        ana_obj = AnalysisBase(
            atomgroup=ag,
            unwrap=False,
            pack=True,
            refgroup=None,
            jitter=0.0,
            wrap_compound="atoms",
            concfreq=0,
        )

        # Create empty methods for allowing the run method to succeed.
        ana_obj._prepare = lambda: None
        ana_obj._single_frame = lambda: None
        ana_obj._conclude = lambda: None

        parent_logger = logging.getLogger("maicos")

        # First run with verbose=True
        ana_obj.run(stop=1, verbose=True)
        assert parent_logger.level == logging.INFO
        # Verify that INFO messages are logged
        assert "Analysing 1 trajectory frames." in caplog.text
        caplog.clear()

        # Second run with verbose=False - level should change to WARNING
        ana_obj.run(stop=1, verbose=False)
        assert parent_logger.level == logging.WARNING
        # Verify that INFO messages are NOT logged
        assert "Analysing 1 trajectory frames." not in caplog.text
        caplog.clear()

        # Third run with verbose=True again - level should change back to INFO
        ana_obj.run(stop=1, verbose=True)
        assert parent_logger.level == logging.INFO
        # Verify that INFO messages are logged after switching back to verbose
        assert "Analysing 1 trajectory frames." in caplog.text

    def test_unwrap_atoms(self, ag, caplog):
        """Test that unwrap is always False for `wrap_compound="atoms"`."""
        with caplog.at_level(logging.DEBUG, logger="maicos.core.base"):
            profile = AnalysisBase(
                atomgroup=ag,
                unwrap=True,
                pack=True,
                wrap_compound="atoms",
                refgroup=None,
                jitter=0.0,
                concfreq=0,
            )

        assert "'atoms` is superfluous." in caplog.text

        assert profile.unwrap is False

    def test_jitter(self, ag_single_frame):
        """Test the jitter option.

        Call the DensityPlanar module with a jitter of 0.01, and make sure that the
        density profile has no peak at a position of 100 (which would be the case
        without jitter).
        """
        dims = ag_single_frame.universe.dimensions.copy()
        dims[2] = 2
        ag_single_frame.universe.dimensions = dims
        ag_single_frame = ag_single_frame[ag_single_frame.positions[:, 2] <= 2]

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
            AnalysisBase(
                u_no_cell.atoms,
                unwrap=True,
                pack=True,
                refgroup=None,
                jitter=0.0,
                wrap_compound="atoms",
                concfreq=0,
            )

    def test_no_dimensions_pack_error(self, u_no_cell):
        """Test that an error is raised if `unwrap=True` but no cell is present."""
        match = "Universe does not have `dimensions` and can't be packed!"
        with pytest.raises(ValueError, match=match):
            AnalysisBase(
                u_no_cell.atoms,
                unwrap=False,
                pack=True,
                refgroup=None,
                jitter=0.0,
                wrap_compound="atoms",
                concfreq=0,
            )

    def test_refgroup_pack_error(self, ag_single_frame):
        """Test that an error is raised if a refgroup is present an pack is disabled."""
        match = "Disabling `pack` with a `refgroup` is not allowed."
        with pytest.raises(ValueError, match=match):
            AnalysisBase(
                ag_single_frame,
                unwrap=False,
                pack=False,
                refgroup=ag_single_frame,
                jitter=0.0,
                wrap_compound="atoms",
                concfreq=0,
            )

    def test_no_dimensions_run(self, u_no_cell):
        """Test that an analysis can be run for a universe without cell information."""
        class_obj = Conclude(u_no_cell.atoms)
        class_obj.run(stop=1)

    def test_box_center(self, ag):
        """Test that the box center is calculated correctly."""
        actual = ag.universe.dimensions[:3] / 2
        class_obj = AnalysisBase(
            atomgroup=ag,
            unwrap=False,
            pack=True,
            refgroup=None,
            jitter=0.0,
            wrap_compound="atoms",
            concfreq=0,
        )
        assert_allclose(class_obj.box_center, actual)

    def box_prec(self, ag):
        """Test that the box precision is set correctly."""
        class_obj = AnalysisBase(
            atomgroup=ag,
            unwrap=False,
            pack=True,
            refgroup=None,
            jitter=0.0,
            wrap_compound="atoms",
            concfreq=0,
        )
        assert class_obj.box_dimensions.dtype == np.float64
        assert class_obj.box_center.dtype == np.float64


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

        with pytest.warns(UserWarning, match="still experimental"):
            collection = AnalysisCollection(ana_1, ana_2)
        collection.run()

        assert ana_1.results is not None
        assert ana_2.results is not None

    def test_positions_restored_between_analyses(self):
        """Test that atom positions are restored between analyses in a collection.

        This is important because not only should the timestep be restored after
        each analysis, but also the positions of the universe should be the same
        at the start of each analysis.
        """
        u = mda.Universe(WATER_TPR_NPT, WATER_TRR_NPT)

        class PositionShifter(AnalysisBase):
            """Shifts positions by an offset."""

            def __init__(self, atomgroup):
                super().__init__(  # Don't do anything to the trajectory
                    atomgroup=atomgroup,
                    unwrap=False,
                    pack=False,
                    refgroup=None,
                    jitter=0.0,
                    wrap_compound="atoms",
                    concfreq=0,
                )

            def _prepare(self):
                pass

            def _single_frame(self):
                # After this shift, the universe is left in a modified state.
                self._universe.atoms.positions += 5.0
                self.seen_positions = self._universe.atoms.positions.copy()

            def _conclude(self):
                pass

        ana_1 = PositionShifter(u.atoms)
        ana_2 = PositionShifter(u.atoms)

        with pytest.warns(UserWarning, match="still experimental"):
            collection = AnalysisCollection(ana_1, ana_2)
        collection.run(frames=[0])

        # If positions are properly restored between analyses, both should
        # see the same positions after their respective shift.
        assert_allclose(ana_1.seen_positions, ana_2.seen_positions)

    def test_inconsistent_trajectory(self, u):
        """Test error raise if two analysis objects have a different trajectory."""
        v = mda.Universe(TPR, XTC)

        with (
            pytest.warns(UserWarning, match="still experimental"),
            pytest.raises(ValueError, match="`analysis_instances` do not have the"),
        ):
            AnalysisCollection(Conclude(u.atoms), Conclude(v.atoms))

    def test_no_base_child(self, u):
        """Test error raise if an object is not a AnalyisBase child."""

        class CustomAnalysis:
            def __init__(self, trajectory):
                self._trajectory = trajectory

        # Create collection for common trajectory loop with inconsistent trajectory
        with (
            pytest.warns(UserWarning, match="still experimental"),
            pytest.raises(TypeError, match="not a child of `AnalysisBase`"),
        ):
            AnalysisCollection(CustomAnalysis(u.trajectory))

    def test_save(self, u, monkeypatch, tmp_path):
        """Test that all results can be written to disk with one command."""
        monkeypatch.chdir(tmp_path)

        ana_1 = Conclude(u.atoms, output_prefix="ana1")
        ana_2 = Conclude(u.atoms, output_prefix="ana2")

        with pytest.warns(UserWarning, match="still experimental"):
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
            pack=False,
            refgroup=None,
            jitter=0.0,
            wrap_compound="atoms",
            concfreq=0,
        )
        # Create empty methods for allowing the run method to succeed.
        ana_2._prepare = lambda: None
        ana_2._single_frame = lambda: None
        ana_2._conclude = lambda: None

        with pytest.warns(UserWarning, match="still experimental"):
            collection = AnalysisCollection(ana_1, ana_2)
        collection.run(stop=1)
        with pytest.warns(UserWarning, match=r"has no save\(\) method"):
            collection.save()

        assert Path(f"{ana_1.output_prefix}out_{ana_1._index}").exists()


class Test_ProfileBase:
    """Test class for the ProfileBase Class.

    The single_frame is for now extensivley tested in the child ``ProfilePlanarBase``,
    ``ProfileCylinderBase`` and ``ProfileSphereBase`` for simple physical system.
    """

    @pytest.fixture
    def u(self):
        """Simple empty Universe."""
        return mda.Universe.empty(
            n_atoms=10,
            n_residues=10,
            n_segments=10,
            atom_resindex=np.arange(10),
            residue_segindex=np.arange(10),
        )

    @pytest.fixture
    def params(self, u):
        """Fixture for PlanarBase class atributes."""
        return dict(
            weighting_function=lambda x, grouping, a=1: a * x,  # noqa: ARG005
            weighting_function_kwargs=None,
            atomgroup=u.atoms,
            normalization="number",
            grouping="atoms",
            bin_method="com",
            output="profile.dat",
        )

    def test_wrong_normalization(self, params):
        """Test a wrong normalization string."""
        params.update(normalization="foo")
        with pytest.raises(ValueError, match="'foo' not supported"):
            ProfileBase(**params)._prepare()

    def test_wrong_grouping(self, params):
        """Test a wrong grouping."""
        params.update(grouping="foo")
        with pytest.raises(ValueError, match="'foo' is not a valid option"):
            ProfileBase(**params)._prepare()

    def test_prepare_sets_unwrap_default(self, params):
        """Test that _prepare sets unwrap=True when not previously set."""
        profile = ProfileBase(**params)
        assert not hasattr(profile, "unwrap")
        profile.n_bins = 10
        profile._prepare()
        assert profile.unwrap is True

    def test_compute_histogram_not_implemented(self, params):
        """Test that _compute_histogram raises NotImplementedError on base class."""
        profile = ProfileBase(**params)
        with pytest.raises(NotImplementedError, match="Only implemented in child"):
            profile._compute_histogram(np.zeros((10, 3)))

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
            except KeyError as err:
                raise KeyError(f"{param.name} is not a parameter of {Member}!") from err


class TestDumpLoad:
    """Tests for the dump/load checkpoint methods."""

    @pytest.fixture
    def ag(self):
        """Import MDA universe."""
        return mda.Universe(WATER_TPR_NPT, WATER_TRR_NPT, in_memory=True).atoms

    @pytest.fixture
    def singular(self, ag):
        """Run a SingularSeries analysis."""
        np.random.seed(42)
        ana = SingularSeries(ag)
        ana.run()
        return ana

    @pytest.fixture
    def multiple(self, ag):
        """Run a MultipleSeries analysis."""
        np.random.seed(42)
        ana = MultipleSeries(ag)
        ana.run()
        return ana

    def test_dump_creates_file(self, singular, tmp_path):
        """Test that dump creates an .npz file."""
        fpath = tmp_path / "checkpoint.npz"
        singular.dump(str(fpath))
        assert fpath.exists()

    def test_dump_warns_on_missing_extension(self, singular, tmp_path):
        """Dump warns when filename lacks the .npz extension and appends it."""
        fpath = tmp_path / "missing_ext"
        with pytest.warns(UserWarning, match=r"\.npz"):
            singular.dump(str(fpath))
        assert fpath.with_suffix(".npz").exists()

    def test_dump_does_not_mutate_universe(self, singular, tmp_path):
        """Test that dump leaves the user's universe untouched."""
        fpath = tmp_path / "checkpoint.npz"
        traj_before = singular._universe.trajectory
        n_frames_before = traj_before.n_frames
        singular.dump(str(fpath))
        assert singular._universe.trajectory is traj_before
        assert singular._universe.trajectory.n_frames == n_frames_before
        assert singular._trajectory is traj_before

    def test_roundtrip_means(self, singular, tmp_path):
        """Test that means survive a dump/load roundtrip."""
        fpath = tmp_path / "checkpoint.npz"
        singular.dump(str(fpath))

        restored = SingularSeries.load(str(fpath))

        for key in singular.means:
            assert_allclose(restored.means[key], singular.means[key])

    def test_roundtrip_sems(self, singular, tmp_path):
        """Test that sems survive a dump/load roundtrip."""
        fpath = tmp_path / "checkpoint.npz"
        singular.dump(str(fpath))

        restored = SingularSeries.load(str(fpath))

        for key in singular.sems:
            assert_allclose(restored.sems[key], singular.sems[key])

    def test_roundtrip_results(self, singular, tmp_path):
        """Test that results survive a dump/load roundtrip."""
        fpath = tmp_path / "checkpoint.npz"
        singular.dump(str(fpath))

        restored = SingularSeries.load(str(fpath))

        for key in singular.results:
            assert_allclose(restored.results[key], singular.results[key])

    def test_roundtrip_accumulators(self, singular, tmp_path):
        """Test that sums, pop, M2 survive a dump/load roundtrip."""
        fpath = tmp_path / "checkpoint.npz"
        singular.dump(str(fpath))

        restored = SingularSeries.load(str(fpath))

        for attr in ("sums", "pop", "M2"):
            orig = getattr(singular, attr)
            rest = getattr(restored, attr)
            for key in orig:
                assert_allclose(rest[key], orig[key])

    def test_roundtrip_arrays(self, singular, tmp_path):
        """Test that timeseries, frames, times survive a dump/load roundtrip."""
        fpath = tmp_path / "checkpoint.npz"
        singular.dump(str(fpath))

        restored = SingularSeries.load(str(fpath))

        assert_allclose(restored.timeseries, singular.timeseries)
        assert_allclose(restored.frames, singular.frames)
        assert_allclose(restored.times, singular.times)

    def test_roundtrip_metadata(self, singular, tmp_path):
        """Test that _frame_index, _index, corrtime survive a roundtrip."""
        fpath = tmp_path / "checkpoint.npz"
        singular.dump(str(fpath))

        restored = SingularSeries.load(str(fpath))

        assert restored._frame_index == singular._frame_index
        assert restored._index == singular._index
        assert_allclose(restored.corrtime, singular.corrtime)

    def test_roundtrip_multiple_observables(self, multiple, tmp_path):
        """Test dump/load with variance and population tracking."""
        fpath = tmp_path / "checkpoint.npz"
        multiple.dump(str(fpath))

        restored = MultipleSeries.load(str(fpath))

        for key in multiple.means:
            assert_allclose(restored.means[key], multiple.means[key])
        for key in multiple.sems:
            assert_allclose(restored.sems[key], multiple.sems[key])

    def test_scalar_types_preserved(self, singular, tmp_path):
        """Test that scalar values remain scalars after roundtrip, not 0-d arrays."""
        fpath = tmp_path / "checkpoint.npz"
        singular.dump(str(fpath))

        restored = SingularSeries.load(str(fpath))

        for key in singular.means:
            orig = singular.means[key]
            rest = restored.means[key]
            if np.ndim(orig) == 0:
                assert not isinstance(rest, np.ndarray), (
                    f"means[{key!r}] should be scalar, got {type(rest)}"
                )

    def test_roundtrip_atomgroup(self, singular, tmp_path):
        """Test that the analysed atomgroup is restored on load."""
        fpath = tmp_path / "checkpoint.npz"
        singular.dump(str(fpath))

        restored = SingularSeries.load(str(fpath))

        assert restored.atomgroup.n_atoms == singular.atomgroup.n_atoms
        np.testing.assert_array_equal(
            restored.atomgroup.indices, singular.atomgroup.indices
        )

    def test_loaded_universe_has_no_trajectory(self, singular, tmp_path):
        """Test that the rebuilt universe carries only the topology."""
        fpath = tmp_path / "checkpoint.npz"
        singular.dump(str(fpath))

        restored = SingularSeries.load(str(fpath))

        assert restored._trajectory is None
        assert not hasattr(restored._universe, "trajectory")

    def test_load_run_raises_runtime_error(self, singular, tmp_path):
        """Test that calling run() on a loaded instance raises a useful error."""
        fpath = tmp_path / "checkpoint.npz"
        singular.dump(str(fpath))

        restored = SingularSeries.load(str(fpath))

        with pytest.raises(RuntimeError, match="restored via `load`"):
            restored.run()

    @pytest.mark.parametrize(
        ("level", "attr_name"),
        [
            ("atoms", "names"),
            ("atoms", "types"),
            ("atoms", "masses"),
            ("atoms", "charges"),
            ("residues", "resnames"),
            ("residues", "resids"),
            ("segments", "segids"),
        ],
    )
    def test_roundtrip_topology_attrs(self, singular, tmp_path, level, attr_name):
        """Common topology attrs must round-trip — canary for MDA dev changes."""
        fpath = tmp_path / "checkpoint.npz"
        singular.dump(str(fpath))

        restored = SingularSeries.load(str(fpath))

        orig = getattr(getattr(singular._universe, level), attr_name)
        rest = getattr(getattr(restored._universe, level), attr_name)
        np.testing.assert_array_equal(rest, orig)

    def test_load_rejects_version_mismatch(self, singular, tmp_path, monkeypatch):
        """Test that load refuses files written by a different MAICoS version."""
        fpath = tmp_path / "checkpoint.npz"
        singular.dump(str(fpath))

        monkeypatch.setattr("maicos.core.base.__version__", "0.0.0-other")
        with pytest.raises(ValueError, match="version-locked"):
            SingularSeries.load(str(fpath))

    def test_load_rejects_missing_version_tag(self, tmp_path):
        """Test that load refuses files without a MAICoS version tag."""
        fpath = tmp_path / "untagged.npz"
        np.savez(str(fpath), foo=np.zeros(3))
        with pytest.raises(ValueError, match="version tag"):
            SingularSeries.load(str(fpath))

    def test_no_new_mda_topology_attrs(self):
        """Alert when MDAnalysis registers new per-object TopologyAttr classes.

        ``dump`` serialises anything in ``_topology.attrs`` whose
        ``per_object`` is one of ``atom``/``residue``/``segment``, so new
        attrs round-trip automatically. We still want a deliberate review
        when MDAnalysis grows its registry: a new attr may carry data that
        needs explicit roundtrip coverage in
        ``test_roundtrip_topology_attrs``, or may use a dtype our
        ``object → str`` coercion does not handle. This test enumerates
        leaf TopologyAttr subclasses from MDAnalysis itself and fails
        loudly when the set changes (run under ``tox -e tests-dev`` to
        surface upstream additions early).
        """
        from MDAnalysis.core.topologyattrs import TopologyAttr

        def walk(cls):
            yield cls
            for sub in cls.__subclasses__():
                yield from walk(sub)

        # Concrete leaves only: skip abstract bases like ResidueAttr that
        # share an ``attrname`` with their subclasses.
        levels = {"atom", "residue", "segment"}
        seen: set[tuple[str, str]] = set()
        for cls in walk(TopologyAttr):
            per_object = getattr(cls, "per_object", None)
            if per_object not in levels or cls.__subclasses__():
                continue
            seen.add((str(per_object), str(cls.attrname)))
        known = {
            ("atom", "altLocs"),
            ("atom", "aromaticities"),
            ("atom", "atomiccharges"),
            ("atom", "chainIDs"),
            ("atom", "charges"),
            ("atom", "epsilon14s"),
            ("atom", "epsilons"),
            ("atom", "formalcharges"),
            ("atom", "gbscreens"),
            ("atom", "ids"),
            ("atom", "masses"),
            ("atom", "names"),
            ("atom", "nbindices"),
            ("atom", "occupancies"),
            ("atom", "radii"),
            ("atom", "record_types"),
            ("atom", "rmin14s"),
            ("atom", "rmins"),
            ("atom", "solventradii"),
            ("atom", "tempfactors"),
            ("atom", "types"),
            ("residue", "icodes"),
            ("residue", "molnums"),
            ("residue", "moltypes"),
            ("residue", "resids"),
            ("residue", "resnames"),
            ("residue", "resnums"),
            ("segment", "models"),
            ("segment", "segids"),
        }
        unexpected = seen - known
        missing = known - seen
        assert not unexpected, (
            f"MDAnalysis registered new per-object TopologyAttr classes: "
            f"{sorted(unexpected)}. Decide whether each needs explicit "
            f"`test_roundtrip_topology_attrs` coverage (esp. for non-string "
            f"object dtypes), then add them to `known`."
        )
        assert not missing, (
            f"TopologyAttr classes disappeared from MDAnalysis: "
            f"{sorted(missing)}. Update `known` and roundtrip coverage."
        )

    def test_no_unknown_topology_attrs(self, singular):
        """Alert when MDAnalysis exposes new topology attrs we don't yet test.

        ``dump`` serialises everything in ``_topology.attrs`` whose
        ``per_object`` is one of ``atom``/``residue``/``segment``. New attrs
        are picked up automatically, but a new attr can also signal that we
        should extend ``test_roundtrip_topology_attrs`` with explicit
        coverage. This test fails loudly when the set grows so the change
        gets a deliberate review (run under ``tox -e tests-dev``).
        """
        known = {
            ("atom", "chainIDs"),
            ("atom", "charges"),
            ("atom", "ids"),
            ("atom", "masses"),
            ("atom", "names"),
            ("atom", "types"),
            ("residue", "molnums"),
            ("residue", "moltypes"),
            ("residue", "resids"),
            ("residue", "resnames"),
            ("residue", "resnums"),
            ("segment", "segids"),
        }
        seen = {
            (attr.per_object, attr.attrname)
            for attr in singular._universe._topology.attrs
            if getattr(attr, "per_object", None) in {"atom", "residue", "segment"}
        }
        unexpected = seen - known
        missing = known - seen
        assert not unexpected, (
            f"MDAnalysis exposes new per-object topology attrs not covered "
            f"by dump/load tests: {sorted(unexpected)}. Update `known` and "
            f"add them to `test_roundtrip_topology_attrs` if they should "
            f"round-trip."
        )
        assert not missing, (
            f"Topology attrs disappeared from the test universe: "
            f"{sorted(missing)}. MDAnalysis may have renamed or removed "
            f"them; update `known` and roundtrip coverage."
        )


class Test_Covariance:
    """Tests for the iterative off-diagonal covariance accumulation."""

    @pytest.fixture
    def ag(self):
        """Import MDA universe."""
        return mda.Universe(WATER_TPR_NPT, WATER_TRR_NPT, in_memory=True).atoms

    @pytest.fixture
    def ana(self, ag):
        """Run a CorrelatedSeries analysis with a fixed seed."""
        np.random.seed(42)
        ana = CorrelatedSeries(ag)
        ana.run()
        return ana

    @staticmethod
    def _comoment(a, b):
        """Batch co-moment sum((a - a.mean())(b - b.mean())) along axis 0."""
        return ((a - a.mean(axis=0)) * (b - b.mean(axis=0))).sum(axis=0)

    def test_comoment_matches_batch(self, ana):
        """Streamed co-moment equals the batch reference for scalar pairs."""
        assert_allclose(
            ana.moments.C[make_pair_key("x", "y")], self._comoment(ana.x, ana.y), rtol=1e-9
        )

    def test_comoment_array_observable(self, ana):
        """Element-wise co-moment of a scalar with an array observable."""
        ref = self._comoment(ana.x[:, None], ana.prof)
        assert ana.moments.C[make_pair_key("prof", "x")].shape == (3,)
        assert_allclose(ana.moments.C[make_pair_key("prof", "x")], ref, rtol=1e-9)

    def test_diagonal_equals_variance(self, ana):
        """cov(key, key) reproduces the squared standard error of the mean."""
        assert_allclose(ana.moments.cov("x", "x"), ana.sems.x**2, rtol=1e-12)

    def test_cov_is_covariance_of_means(self, ana):
        """cov() divides the co-moment by the squared shared population."""
        n = ana.n_frames
        assert_allclose(ana.moments.cov("x", "y"), self._comoment(ana.x, ana.y) / n**2)
        assert_allclose(
            ana.moments.cov("x", "y"), np.cov(np.vstack([ana.x, ana.y]), bias=True)[0, 1] / n
        )

    def test_cov_symmetric(self, ana):
        """cov() is symmetric in its arguments."""
        assert_allclose(ana.moments.cov("x", "y"), ana.moments.cov("y", "x"))

    def test_incompatible_pair_not_tracked(self, ana):
        """Pairs whose shapes do not broadcast are never tracked."""
        assert make_pair_key("other", "prof") not in ana.moments.C
        with pytest.raises(KeyError, match="do not broadcast"):
            ana.moments.cov("prof", "other")

    def test_propagate_matches_manual(self, ana):
        """propagate() of f = x*y matches the explicit bilinear form."""
        grads = {"x": ana.means.y, "y": ana.means.x}
        expected = np.sqrt(
            grads["x"] ** 2 * ana.sems.x**2
            + grads["y"] ** 2 * ana.sems.y**2
            + 2 * grads["x"] * grads["y"] * ana.moments.cov("x", "y")
        )
        assert_allclose(ana.moments.propagate_error(grads), expected, rtol=1e-12)

    def test_propagate_differs_from_diagonal(self, ana):
        """Correlated variables: full propagation differs from the diagonal-only one."""
        grads = {"x": ana.means.y, "y": ana.means.x}
        diagonal_only = np.sqrt(
            grads["x"] ** 2 * ana.sems.x**2 + grads["y"] ** 2 * ana.sems.y**2
        )
        assert not np.isclose(ana.moments.propagate_error(grads), diagonal_only)

    def test_propagate_raises_on_untracked(self, ana):
        """propagate() raises when a requested pair has no tracked covariance."""
        with pytest.raises(KeyError, match="do not broadcast"):
            ana.moments.propagate_error({"prof": np.ones(3), "other": np.ones(2)})

    def test_uncorrelated_covariance_is_small(self, ag):
        """Independent observables have near-zero off-diagonal covariance of means."""
        np.random.seed(7)
        ana = CorrelatedSeries(ag)
        ana.run()
        # `other` is independent of `x`; covariance of the means -> 0 as 1/n.
        cov_xother = (
            ana.moments.C[make_pair_key("other", "x")] / joint_pop(ana.pop["other"], ana.pop["x"]) ** 2
        )
        assert np.all(np.abs(cov_xother) < np.abs(ana.moments.cov("x", "y")))

    def test_not_cosampled_pair_not_tracked(self, ag):
        """Shape-compatible but differently-populated observables are not tracked."""
        np.random.seed(1)
        ana = WeightedSeries(ag)
        ana.run()
        assert make_pair_key("single", "weighted") not in ana.moments.C
        with pytest.raises(KeyError, match="do not broadcast"):
            ana.moments.cov("single", "weighted")

    def test_roundtrip_covariance(self, ana, tmp_path):
        """The covariance container survives a dump/load roundtrip with tuple keys."""
        fpath = tmp_path / "checkpoint.npz"
        ana.dump(str(fpath))
        restored = CorrelatedSeries.load(str(fpath))

        assert set(restored.C) == set(ana.moments.C)
        for key in ana.moments.C:
            assert isinstance(key, tuple)
            assert_allclose(restored.C[key], ana.moments.C[key])

    def test_multisample_array_matches_batch(self, ag):
        """Multi-sample array observables (block path) match the batch co-moment."""
        ana = MultiSampleSeries(ag, n_bins=4, seed=0)
        ana.run()
        streamed = ana.moments.C[make_pair_key("x", "y")]
        assert streamed.shape == (4,)
        assert_allclose(streamed, ana.batch_comoment(), rtol=1e-9)

    def test_multisample_scalar_matches_batch(self, ag):
        """Multi-sample scalar observables (fallback path) match the batch co-moment."""
        ana = MultiSampleSeries(ag, n_bins=1, seed=3)
        ana.run()
        streamed = ana.moments.C[make_pair_key("x", "y")]
        assert_allclose(streamed, ana.batch_comoment(), rtol=1e-9)

    def test_multisample_cov_matches_numpy(self, ag):
        """cov() of multi-sample observables equals numpy's pooled covariance."""
        ana = MultiSampleSeries(ag, n_bins=1, seed=3)
        ana.run()
        allx = np.concatenate([s[:, 0] for s in ana.batch[0]])
        ally = np.concatenate([s[:, 1] for s in ana.batch[0]])
        n = allx.size
        assert_allclose(ana.moments.cov("x", "y"), np.cov(allx, ally, bias=True)[0, 1] / n)
