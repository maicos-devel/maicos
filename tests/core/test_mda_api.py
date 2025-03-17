"""Test that maicos.core.Analysis API is same as MDAnlysis.analysis.base.Analysis.

We run the original tests from ``MDAnlysis.analysis.base.Analysis`` using
``maicos.core.AnalysisBase`` which mimics the original API. This requires some
monkeypatching which explain in detail below in code.
"""

import MDAnalysis as mda
import pytest
from MDAnalysisTests.analysis.test_base import *  # noqa: F401, F403
from MDAnalysisTests.analysis.test_base import (
    FrameAnalysis,
    IncompleteAnalysis,
    test_analysis_class,
    test_AnalysisFromFunction,
    test_AnalysisFromFunction_args_content,
    test_frame_bool_fail,
    test_frame_slice_parallel,
    test_frames_times,
    test_incompatible_n_workers,
    test_instance_serial_backend,
    test_n_workers_conflict_raises_value_error,
    test_parallelizable_transformations,
    test_progressbar_multiprocessing,
    test_reset_n_parts_to_n_frames,
    test_rewind,
    test_start_stop_step_parallel,
    test_verbose,
    test_warn_nparts_nworkers,
)
from MDAnalysisTests.datafiles import DCD, PSF, TPR, XTC

from maicos.core import AnalysisBase

# Remove specific functions/classes we don't want or can test because of a different API
del test_verbose
del test_frame_bool_fail

# Remove parallel tests because we don't support parallelization, yet...
del test_analysis_class
del test_AnalysisFromFunction
del test_AnalysisFromFunction_args_content
del test_frame_slice_parallel
del test_frames_times
del test_incompatible_n_workers
del test_instance_serial_backend
del test_n_workers_conflict_raises_value_error
del test_parallelizable_transformations
del test_progressbar_multiprocessing
del test_reset_n_parts_to_n_frames
del test_rewind
del test_start_stop_step_parallel
del test_warn_nparts_nworkers


@pytest.fixture(autouse=True)
def override_analysis_base():
    """Monkeypatch test classes used by MDAnalysisTests.

    We overwrite the ``__bases__`` attribute as well as the ``__init__`` methods. For
    the ``__init__`` method we change from passing a ``reader`` as required by the
    MDAnalysis' classes to ``reader.universe.atoms`` which is an ``AtomGrouop`` required
    by our ``AnalysisBase``.
    """

    def frame_analysis_init(self, reader):
        super(FrameAnalysis, self).__init__(
            reader.universe.atoms,
            unwrap=False,
            pack=False,
            refgroup=None,
            jitter=0.0,
            wrap_compound="atoms",
            concfreq=0.0,
        )
        self.traj = reader
        self.found_frames = []

    FrameAnalysis.__bases__ = (AnalysisBase,)
    FrameAnalysis.__init__ = frame_analysis_init

    def incomplete_analysis_init(self, reader):
        super(IncompleteAnalysis, self).__init__(
            reader.universe.atoms,
            unwrap=False,
            pack=False,
            refgroup=None,
            jitter=0.0,
            wrap_compound="atoms",
            concfreq=0.0,
        )
        self.traj = reader
        self.found_frames = []

    IncompleteAnalysis.__bases__ = (AnalysisBase,)
    IncompleteAnalysis.__init__ = incomplete_analysis_init


@pytest.fixture(scope="module")
def u():
    """A universe without box dimensions."""
    universe = mda.Universe(PSF, DCD)
    # add custom universe reference to reader to allow backrefence
    universe.trajectory.universe = universe

    return universe


@pytest.fixture(scope="module")
def u_xtc():
    """A universe with box dimensions."""
    universe = mda.Universe(TPR, XTC)  # dt = 100
    # add custom universe reference to reader to allow backrefence
    universe.trajectory.universe = universe

    return universe


# Some tests have to be adjusted manually due to whatever reasons and are copied.


def test_verbose_progressbar_run(u, capsys):
    """Test that progtressbar prints to the terminal."""
    FrameAnalysis(u.trajectory).run(verbose=True)
    _, err = capsys.readouterr()
    actual = err.strip().split("\r")[-1]
    assert "100%" in actual


def test_verbose_progressbar_run_with_kwargs(u, capsys):
    """Test that progtressbar prints to the terminal with a custom argument."""
    FrameAnalysis(u.trajectory).run(verbose=True, progressbar_kwargs={"desc": "custom"})
    _, err = capsys.readouterr()
    actual = err.strip().split("\r")[-1]
    assert "custom: 100%" in actual


@pytest.mark.parametrize(
    "run_kwargs",
    [
        ({"start": 4, "frames": [4, 5, 6, 7, 8, 9]}),
        ({"stop": 6, "frames": [0, 1, 2, 3, 4, 5]}),
        ({"step": 2, "frames": [0, 2, 4, 6, 8]}),
        ({"start": 4, "stop": 7, "frames": [4, 5, 6]}),
        ({"stop": 6, "step": 2, "frames": [0, 2, 4, 6]}),
        ({"start": 4, "step": 2, "frames": [4, 6, 8]}),
        ({"start": 0, "stop": 0, "step": 0, "frames": [4, 6, 8]}),
    ],
)
def test_frame_fail(u, run_kwargs):
    """Test that frames cannot be combined with start/stop/step."""
    an = FrameAnalysis(u.trajectory)
    msg = "start/stop/step cannot be combined with frames"
    with pytest.raises(ValueError, match=msg):
        an.run(**run_kwargs)
