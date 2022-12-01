#!/usr/bin/env python3
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
#
# Copyright (c) 2022 Authors and contributors
# (see the file AUTHORS for the full list of names)
#
# Released under the GNU Public Licence, v2 or any higher version
# SPDX-License-Identifier: GPL-2.0-or-later
"""
Writing your own analysis module
================================

To write your own analysis module you can use the example given below.
As all MAICoS modules this inherits from the
:class:`maicos.core.base.AnalysisBase` class.

The example will calculate the average box volume and stores the result
within the result object of the class.
"""

import logging

import numpy as np

from maicos.core import AnalysisBase


# %%
#
# Creating a logger makes debugging easier.

logger = logging.getLogger(__name__)

# %%
#
# In the following the example of an analysis class.


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
        # self.atomgroup refers to the provided `atomgroup`
        # self._universe refers to full universe of given `atomgroup`
        self.volume = 0

    def _single_frame(self):
        """Calculate data from a single frame of trajectory.

        Don't worry about normalising, just deal with a single frame.
        """
        # Current frame index: self._frame_index
        # Current timestep object: self._ts

        volume = self._ts.volume
        self.volume += volume

        # Eeach module should return a characteristic scalar which is used
        # by MAICoS to estimate correlations of an Analysis.
        return volume

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

# %%
#
# Now run your new module as all the other modules.
#
# Using your modules from the command line
# ----------------------------------------
#
# To add your custom module to the MAICoS CLI first
# create a ``.maicos`` folder in your home directory.
# Afterwards, copy your analysis module file to this folder. MAICoS will detect
# all modules within the ``.maicos`` folder based on a
# ``maicos_custom_modules.py`` file. Create this file based on the example below
# and adjust the name of your module accordingly.
#
# .. literalinclude:: ../../../examples/maicos_custom_modules.py
#    :language: python
