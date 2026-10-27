# Copyright (c) 2026 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the GNU Public Licence, v3 or any higher version
# SPDX-License-Identifier: GPL-3.0-or-later
"""Benchmarks for :class:`maicos.core.AnalysisBase`."""

from typing import ClassVar

import MDAnalysis as mda
import numpy as np

from benchmarks.synthetic import make_universe
from maicos.core import AnalysisBase
from tests.data import WATER_TPR_NPT, WATER_TRR_NPT


class _RandomObs(AnalysisBase):
    """Minimal analysis that writes random observables — measures framework overhead."""

    def __init__(self, atomgroup, n_obs=10, **kwargs):
        self._n_obs = n_obs
        kwargs.setdefault("unwrap", False)
        kwargs.setdefault("pack", False)
        kwargs.setdefault("refgroup", None)
        kwargs.setdefault("jitter", 0.0)
        kwargs.setdefault("wrap_compound", "atoms")
        kwargs.setdefault("concfreq", 0)
        super().__init__(atomgroup=atomgroup, **kwargs)

    def _single_frame(self):
        for i in range(self._n_obs):
            self._obs[f"obs{i}"] = np.random.rand()
        return np.random.rand()


class AnalysisBaseBenchmark:
    """Direct framework overhead of :class:`AnalysisBase` with no real analysis."""

    timeout = 120

    def setup(self):
        """Build the synthetic atomgroup."""
        self.atomgroup = make_universe().atoms

    def time_run(self):
        """Time a bare run over the trajectory."""
        _RandomObs(self.atomgroup).run()

    def peakmem_run(self):
        """Peak memory of a bare run over the trajectory."""
        _RandomObs(self.atomgroup).run()


class ObsAccumulationBenchmark:
    """Observable-accumulation overhead as the number of ``_obs`` entries grows."""

    timeout = 180
    params: ClassVar[list[int]] = [1, 10, 100, 1000]
    param_names: ClassVar[list[str]] = ["n_obs"]

    def setup(self, _n_obs):
        """Build the synthetic atomgroup."""
        self.atomgroup = make_universe().atoms

    def time_run(self, n_obs):
        """Time a run accumulating ``n_obs`` observables per frame."""
        _RandomObs(self.atomgroup, n_obs=n_obs).run()


class SingleFrameBenchmark:
    """Cost of the per-frame transforms (pack, refgroup, unwrap)."""

    timeout = 180
    params: ClassVar[list[str]] = ["none", "pack", "pack+refgroup", "unwrap"]
    param_names: ClassVar[list[str]] = ["transform"]

    def setup(self, _transform):
        """Build the synthetic atomgroup."""
        self.atomgroup = make_universe().atoms

    def _kwargs(self, transform):
        if transform == "pack":
            return {"pack": True}
        if transform == "pack+refgroup":
            half = self.atomgroup[: len(self.atomgroup) // 2]
            return {"refgroup": half, "pack": True}
        if transform == "unwrap":
            return {"unwrap": True, "wrap_compound": "residues"}
        return {}

    def time_run(self, transform):
        """Time a run applying the selected per-frame transform."""
        _RandomObs(self.atomgroup, n_obs=1, **self._kwargs(transform)).run()


class CovarianceSeries(AnalysisBase):
    """Emits ``n_obs`` co-sampled array observables to drive covariance accumulation.

    Each frame writes ``n_obs`` observables of shape ``(n_bins,)``, so the base
    class tracks ``n_obs * (n_obs - 1) / 2`` off-diagonal co-moment pairs.

    Parameters
    ----------
    atomgroup : MDAnalysis.core.groups.AtomGroup
        Dummy atomgroup to satisfy the class.
    n_obs : int
        Number of observables written per frame, named o{i}.
    n_bins : int
        Length of each observable array.
    """

    def __init__(self, atomgroup, n_obs, n_bins):
        self._n_obs = n_obs
        self._n_bins = n_bins
        keys = [f"o{i}" for i in range(n_obs)]
        self._compute_covariance = [
            {a, b} for k, a in enumerate(keys) for b in keys[k + 1 :]
        ]
        super().__init__(
            atomgroup=atomgroup,
            unwrap=False,
            pack=False,
            refgroup=None,
            jitter=0.0,
            wrap_compound="atoms",
            concfreq=0,
        )

    def _single_frame(self):
        for i in range(self._n_obs):
            self._obs[f"o{i}"] = np.random.rand(self._n_bins)


class CovarianceBenchmark:
    """Benchmark the per-frame off-diagonal covariance accumulation."""

    timeout = 300
    params: ClassVar[list[int]] = [4, 8, 16]
    param_names: ClassVar[list[str]] = ["n_obs"]

    def setup(self, _n_obs):
        """Load a multi-frame universe shared across the parametrized runs."""
        self.atoms = mda.Universe(WATER_TPR_NPT, WATER_TRR_NPT, in_memory=True).atoms

    def time_covariance_run(self, n_obs):
        """Run an analysis emitting ``n_obs`` co-sampled array observables."""
        CovarianceSeries(self.atoms, n_obs=n_obs, n_bins=100).run()
