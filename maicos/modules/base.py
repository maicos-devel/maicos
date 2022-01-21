#!/usr/bin/env python3
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
#
# Copyright (c) 2021 Authors and contributors
# (see the file AUTHORS for the full list of names)
#
# Released under the GNU Public Licence, v2 or any higher version
# SPDX-License-Identifier: GPL-2.0-or-later

import logging
import warnings
from ..utils import sort_atomgroup

import numpy as np
from MDAnalysis.analysis import base
from MDAnalysis.lib.log import ProgressBar

logger = logging.getLogger(__name__)


class _AnalysisBase(base.AnalysisBase):
    """Extends the MDAnalysis base class for defining multi frame analysis."""

    def __init__(self, trajectory, verbose=False, save=False, **kwargs):
        """
        Parameters
        ----------
        trajectory : mda.Reader
            A trajectory Reader
        verbose : bool, optional
           Turn on more logging and debugging, default ``False``
        save : bool, optional
           Save results to a file, default ``False``
        """
        super().__init__(self, **kwargs)

        self._trajectory = trajectory
        self._verbose = verbose
        self._save = save
        self.results = base.Results()

    def _setup_frames(self, trajectory, begin=0, end=None, dt=0):
        """
        Pass a Reader object and define the desired iteration pattern
        through the trajectory

        Parameters
        ----------
        trajectory : mda.Reader
            A trajectory Reader
        begin : float, optional
            start time (ps) for evaluation
        end : float, optional
            end time (ps) for evaluation
        dt : float, optional
            time step (ps) to read analysis frame. If `0` take all frames
        """
        self._trajectory = trajectory

        self.begin = begin
        self.end = end
        self.dt = dt

        with warnings.catch_warnings():
            warnings.simplefilter('always')
            if begin > trajectory.totaltime:
                raise ValueError("Start ({:.2f} ps) is larer than total time "
                                 "({:.2f} ps)."
                                 "".format(begin, trajectory.totaltime))

            if end is not None and end < trajectory.dt:
                raise ValueError("End ({:.2f} ps) is smaller than a single "
                                 "trajectory timestep ({:.2f} ps)."
                                 "".format(end, trajectory.dt))

            if begin > 0:
                startframe = int(begin // trajectory.dt)
            else:
                startframe = 0
            if end is not None:
                stopframe = int(end // trajectory.dt)
            else:
                stopframe = trajectory.n_frames
            if self.dt > 0:
                step = int(dt // trajectory.dt)
            else:
                step = 1

        startframe, stopframe, step = trajectory.check_slice_indices(
            startframe, stopframe, step)
        self.startframe = startframe
        self.stopframe = stopframe
        self.step = step
        self.n_frames = len(range(startframe, stopframe, step))
        self.frames = np.zeros(self.n_frames, dtype=int)
        self.times = np.zeros(self.n_frames)

    def _configure_parser(self, parser):
        """Adds parser options using an argparser object"""
        parser.description = self.__doc__

    def _calculate_results(self):
        """Calculate the results"""
        pass

    def _save_results(self):
        """Saves the results you've gatherd to a file."""
        pass

    def run(self, begin=0, end=None, dt=0, verbose=None):
        """Perform the calculation

        Parameters
        ----------
        begin : float, optional
            start time (ps) for evaluation
        end : float, optional
            end time (ps) for evaluation
        dt : float, optional
            time step (ps) to read analysis frame
        verbose : bool, optional
            Turn on verbosity
        """
        logger.info("Choosing frames to analyze")
        # if verbose unchanged, use class default
        verbose = getattr(self, '_verbose',
                          False) if verbose is None else verbose

        self._setup_frames(self._trajectory, begin, end, dt)
        logger.info("Starting preparation")
        self._prepare()
        for i, ts in enumerate(ProgressBar(
                self._trajectory[self.startframe:self.stopframe:self.step],
                verbose=verbose)):
            self._frame_index = i
            self.frames[i] = ts.frame
            self.times[i] = ts.time
            self._ts = ts
            # logger.info("--> Doing frame {} of {}".format(i+1, self.n_frames))
            self._single_frame()
        logger.info("Finishing up")
        self._calculate_results()
        self._conclude()
        if self._save:
            self._save_results()
        return self


class SingleGroupAnalysisBase(_AnalysisBase):
    """The base class for analysing a single AtomGroup only."""

    _allow_multiple_atomgroups = False

    def __init__(self, atomgroup, **kwargs):
        super().__init__(atomgroup.universe.trajectory, **kwargs)
        self.atomgroup = sort_atomgroup(atomgroup)
        self._universe = atomgroup.universe


class MultiGroupAnalysisBase(_AnalysisBase):
    """The base class for analysing a single or multiple AtomGroups."""

    _allow_multiple_atomgroups = True

    def __init__(self, atomgroups, **kwargs):
        if type(atomgroups) not in [list, tuple, np.ndarray]:
            # Sort the atomgroups,
            # such that molecules are listed one after the other
            atomgroups = [sort_atomgroup(atomgroups)]
        else:
            atomgroups = list(map(sort_atomgroup, atomgroups))
            # Check that all atomgroups are from same universe
            if len(set([ag.universe for ag in atomgroups])) != 1:
                raise ValueError(
                    "Given Atomgroups are not from the same Universe.")
        super().__init__(atomgroups[0].universe.trajectory, **kwargs)

        self.atomgroups = atomgroups
        self.n_atomgroups = len(self.atomgroups)
        self._universe = atomgroups[0].universe
