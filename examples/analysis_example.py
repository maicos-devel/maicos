#!/usr/bin/env python
# coding: utf-8

# ========== DESCRIPTION ===========
# This is an example for an analysis script. To use this
# script do the following steps:
# 1. Copy it to the "mdtsools/ana" folder and add your code.
# 2. Choose an unique name and add <"analysis_example"> to the __all__ list
#    in "mdtools/ana/__init__.py".
# 3. OPTIONAL: Add bash completion commands to "mdtools/share/mdtools_completion.bash".
# ==================================

# Mandatory imports
from __future__ import absolute_import, division, print_function

import os
import sys

import numpy as np

from .base import AnalysisBase


def _configure_parser(parser):
    parser.description = analysis_example.__doc__

    # Custom arguments
    parser.add_argument(
        '-o',
        dest='output',
        type=str,
        default='outfile',
        help='Prefix for output filenames')
    parser.add_argument(
        '-temp',
        dest='temperature',
        type=float,
        default=300,
        help='Reference temperature')


class analysis_example(AnalysisBase):
    """Description for my awesome analysis script."""

    def __init__(self, atomgroup, temperature=300, output="output", **kwargs):
        # Inherit all classes from AnalysisBase
        super(analysis_example, self).__init__(atomgroup.universe.trajectory,
                                               **kwargs)

        self.atomgroup = atomgroup
        self.temperature = temperature
        self.output = output

    def _prepare(self):
        """Set things up before the analysis loop begins"""
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
            print("Average volume of the simulation box {:.2f} Å**3".format(
                self.results["volume"]))

    def _save_results(self):
        """Saves results to a file. 
        
        Called at the end of the run() method after _calculate_results and
        _conclude"""

        np.savetxt(
            self.output + '.dat',
            np.array([self.results["volume"]]),
            fmt='%1.2f',
            header='volume / Angstrom**3')
