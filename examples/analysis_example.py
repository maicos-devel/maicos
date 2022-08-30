#!/usr/bin/env python3
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
#
# Copyright (c) 2022 Authors and contributors
# (see the file AUTHORS for the full list of names)
#
# Released under the GNU Public Licence, v2 or any higher version
# SPDX-License-Identifier: GPL-2.0-or-later
"""An examples Module."""

# Mandatory imports
import logging

import numpy as np

from maicos.core import AnalysisBase


logger = logging.getLogger(__name__)


class AnalysisExample(AnalysisBase):
    """Analysis class calcuting the average box volume.

    Parameters
    ----------
    atomgroup : AtomGroup
       Atomgroup on which the analysis is executed
    output : str
        Output filename
    temperature : str
        Reference temperature (K)

    Attributes
    ----------
    results.volume: float
        averaged box volume (Å³)
    """

    def __init__(self,
                 atomgroup,
                 temperature=300,
                 output="outfile.dat"):
        super().__init__(atomgroup)

        self.temperature = temperature
        self.output = output

    def _prepare(self):
        """Set things up before the analysis loop begins."""
        # self.atomgroup - given atomgroup
        # self._universe - full universe of given atomgroup
        self.volume = 0

    def _single_frame(self):
        """Calculate data from a single frame of trajectory.

        Don't worry about normalising, just deal with a single frame.
        """
        # Current frame index: self._frame_index
        # Current timestep object: self._ts

        self.volume += self._ts.volume

    def _conclude(self):
        """Finalise the results you've gathered.

        Called at the end of the run() method to finish everything up.
        """
        self.results.volume = self.volume / self.n_frames
        logger.info("Average volume of the simulation box "
                    f"{self.results.volume:.2f} Å³")

    def save(self):
        """Save results to a file.

        Called at the end of the run() method after _conclude.
        """
        self.savetxt(self.output,
                     np.array([self.results.volume]),
                     columns='volume / Å³')
