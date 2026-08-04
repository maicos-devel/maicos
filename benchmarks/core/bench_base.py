#!/usr/bin/env python
#
# Copyright (c) 2026 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the GNU Public Licence, v3 or any higher version
# SPDX-License-Identifier: GPL-3.0-or-later
"""Benchmarks for :class:`maicos.core.AnalysisBase`."""

import numpy as np

from benchmarks.synthetic import make_universe
from maicos.core import AnalysisBase


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


class AnalysisBaseBenchmark:
    """Direct framework overhead of :class:`AnalysisBase` with no real analysis."""

    timeout = 120

    def setup(self):
        """Build the synthetic atomgroup."""
        self.atomgroup = make_universe()

    def time_run(self):
        """Time a bare run over the trajectory."""
        _RandomObs(self.atomgroup).run()

    def peakmem_run(self):
        """Peak memory of a bare run over the trajectory."""
        _RandomObs(self.atomgroup).run()


class ObsAccumulationBenchmark:
    """Observable-accumulation overhead as the number of ``_obs`` entries grows."""

    timeout = 180
    params = [1, 10, 100, 1000]
    param_names = ["n_obs"]

    def setup(self, _n_obs):
        """Build the synthetic atomgroup."""
        self.atomgroup = make_universe()

    def time_run(self, n_obs):
        """Time a run accumulating ``n_obs`` observables per frame."""
        _RandomObs(self.atomgroup, n_obs=n_obs).run()


class SingleFrameBenchmark:
    """Marginal cost of the per-frame transforms (pack, unwrap, refgroup)."""

    timeout = 180
    params = ["none", "pack", "unwrap", "refgroup"]
    param_names = ["transform"]

    def setup(self, _transform):
        """Build the synthetic atomgroup."""
        self.atomgroup = make_universe()

    def _kwargs(self, transform):
        if transform == "pack":
            return {"pack": True}
        if transform == "unwrap":
            return {"unwrap": True, "wrap_compound": "residues"}
        if transform == "refgroup":
            # the framework requires pack when a refgroup is set
            half = self.atomgroup[: len(self.atomgroup) // 2]
            return {"refgroup": half, "pack": True}
        return {}

    def time_run(self, transform):
        """Time a run applying the selected per-frame transform."""
        _RandomObs(self.atomgroup, n_obs=1, **self._kwargs(transform)).run()
