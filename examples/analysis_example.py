#!/usr/bin/env python3
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
#
# Copyright (c) 2019 Authors and contributors
# (see the file AUTHORS for the full list of names)
#
# Released under the GNU Public Licence, v2 or any higher version
# SPDX-License-Identifier: GPL-2.0-or-later

# Mandatory imports
import numpy as np

from maicos.modules.base import SingleGroupAnalysisBase
from maicos.utils import savetxt


class analysis_example(SingleGroupAnalysisBase):
    """Description for my awesome single group analysis script.

       **Inputs**

       :param output (str): Output filename
       :param temperature (str): Reference temperature (K)

       **Outputs**

       :returns (dict): * volume: averaged box volume (Å³)
    """

    def __init__(self, atomgroup, temperature=300, output="outfile.dat", **kwargs):
        super().__init__(atomgroup, **kwargs)

        self.temperature = temperature
        self.output = output

    def _configure_parser(self, parser):
        # Custom arguments
        # To generate the CLI help `dest` MUST be the same as the name in the docstring
        parser.add_argument('-o', dest='output')
        parser.add_argument('-temp', dest='temperature')

    def _prepare(self):
        """Set things up before the analysis loop begins"""
        # self.atomgroup - given atomgroup
        # self._universe - full universe of given atomgroup
        self.volume = 0

    def _single_frame(self):
        """Calculate data from a single frame of trajectory

        Don't worry about normalising, just deal with a single frame.
        """
        # Current frame index: self._frame_index
        # Current timestep object: self._ts

        self.volume += self._ts.volume

    def _calculate_results(self):
        """Calculate the results.

        Called at the end of the run() method to before the _conclude function.
        Can also called during a run to update the results during processing."""

        self.results["volume"] = self.volume / self.n_frames

    def _conclude(self):
        """Finalise the results you've gathered.

        Called at the end of the run() method after _calculate_results
        to finish everything up."""
        if self._verbose:
            print("Average volume of the simulation box {:.2f} Å³".format(
                self.results["volume"]))

    def _save_results(self):
        """Saves results to a file.

        Called at the end of the run() method after _calculate_results and
        _conclude"""

        savetxt(self.output,
                np.array([self.results["volume"]]),
                fmt='%1.2f',
                header='volume / Å³')
