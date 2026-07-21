#!/usr/bin/env python
#
# Copyright (c) 2026 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the GNU Public Licence, v3 or any higher version
# SPDX-License-Identifier: GPL-3.0-or-later
"""Benchmarks for :class:`maicos.core.AnalysisBase`."""

import MDAnalysis as mda
import numpy as np

from maicos.core import AnalysisBase
from tests.data import WATER_TPR_NPT, WATER_TRR_NPT


class AnalysisBaseBenchmark(AnalysisBase):
    """Minimal analysis that writes random observables — measures framework overhead."""

    def __init__(self, atomgroup, n_obs=10):
        self._n_obs = n_obs
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
        self._obs.data = np.random.rand(self._n_obs)


class CovarianceSeries(AnalysisBase):
    """Emits ``n_obs`` co-sampled array observables to drive covariance accumulation.

    Each frame writes ``n_obs`` observables of shape ``(n_bins,)``, so the base
    class tracks ``n_obs * (n_obs - 1) / 2`` off-diagonal co-moment pairs. This
    is the workload that exercises the per-frame covariance update.
    """

    def __init__(self, atomgroup, n_obs, n_bins):
        self._n_obs = n_obs
        self._n_bins = n_bins
        # Request every pair so the full N*(N-1)/2 accumulation is benchmarked.
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
    params = [4, 8, 16]
    param_names = ["n_obs"]

    def setup(self, _n_obs):
        """Load a multi-frame universe shared across the parametrized runs."""
        self.atoms = mda.Universe(WATER_TPR_NPT, WATER_TRR_NPT, in_memory=True).atoms

    def time_covariance_run(self, n_obs):
        """Run an analysis emitting ``n_obs`` co-sampled array observables."""
        CovarianceSeries(self.atoms, n_obs=n_obs, n_bins=100).run()
