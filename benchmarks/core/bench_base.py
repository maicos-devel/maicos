#!/usr/bin/env python
#
# Copyright (c) 2026 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the GNU Public Licence, v3 or any higher version
# SPDX-License-Identifier: GPL-3.0-or-later
"""Benchmarks for :class:`maicos.core.AnalysisBase`."""

import numpy as np

from maicos.core import AnalysisBase


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
