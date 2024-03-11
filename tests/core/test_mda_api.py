"""Test that maicos.core.Analysis API is same as MDAnlysis.analysis.base.Analysis.

We run the original tests from ``MDAnlysis.analysis.base.Analysis`` using
``maicos.core.AnalysisBase`` which mimics the original API. This requires some
monkeypatching which explain in detail below in code.
"""

import MDAnalysis as mda
import numpy as np
import pytest
from MDAnalysisTests.analysis.test_base import *  # noqa: F401, F403
from MDAnalysisTests.analysis.test_base import (
    FRAMES_ERR,
    TIMES_ERR,
    FrameAnalysis,
    IncompleteAnalysis,
)
from MDAnalysisTests.datafiles import DCD, PSF, TPR, XTC
from numpy.testing import assert_almost_equal, assert_equal

from maicos.core import AnalysisBase


# Remove specific functions/classes we don't want or can test because of a different API
del Test_Results  # noqa: F821
del test_verbose  # noqa: F821
del test_frame_bool_fail  # noqa: F821


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
            multi_group=False,
            unwrap=False,
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
            multi_group=False,
            unwrap=False,
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


# The original tests below define the universe directlty inside their functions. We can
# not overwrite this. We copied them and changed the logic manually for now.
#
# TODO: Remove these reimplementataions once PR #4459 in MDAnalysis is merged and a new
# version of MDAnalysisTests is released.


@pytest.mark.parametrize(
    "run_kwargs, frames",
    [
        ({"frames": [4, 5, 6, 7, 8, 9]}, np.arange(4, 10)),
        ({"frames": [0, 2, 4, 6, 8]}, np.arange(0, 10, 2)),
        ({"frames": [4, 6, 8]}, np.arange(4, 10, 2)),
        ({"frames": [0, 3, 4, 3, 5]}, [0, 3, 4, 3, 5]),
        (
            {
                "frames": [
                    True,
                    True,
                    False,
                    True,
                    False,
                    True,
                    True,
                    False,
                    True,
                    False,
                ]
            },
            (0, 1, 3, 5, 6, 8),
        ),
    ],
)
def test_frame_slice(run_kwargs, frames, u_xtc):
    """Test ``frame`` argument to ``run()`` method."""
    an = FrameAnalysis(u_xtc.trajectory).run(**run_kwargs)
    assert an.n_frames == len(frames)
    assert_equal(an.found_frames, frames)
    assert_equal(an.frames, frames, err_msg=FRAMES_ERR)


def test_frame_bool_fail(u_xtc):
    """Test failure when providing bolean frames argument."""
    an = FrameAnalysis(u_xtc.trajectory)
    frames = [True, True, False]
    msg = "boolean index did not match indexed array along (axis|dimension) 0"
    with pytest.raises(IndexError, match=msg):
        an.run(frames=frames)


def test_rewind(u_xtc):
    """Test that ``ts`` property is unchanged."""
    FrameAnalysis(u_xtc.trajectory).run(frames=[0, 2, 3, 5, 9])
    assert_equal(u_xtc.trajectory.ts.frame, 0)


def test_frames_times(u_xtc):
    """Tets that times are correct."""
    an = FrameAnalysis(u_xtc.trajectory).run(start=1, stop=8, step=2)
    frames = np.array([1, 3, 5, 7])
    assert an.n_frames == len(frames)
    assert_equal(an.found_frames, frames)
    assert_equal(an.frames, frames, err_msg=FRAMES_ERR)
    assert_almost_equal(an.times, frames * 100, decimal=4, err_msg=TIMES_ERR)
