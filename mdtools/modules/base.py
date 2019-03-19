#!/usr/bin/env python3
# coding: utf-8

import logging
import warnings

import numpy as np
from MDAnalysis.analysis import base
from MDAnalysis.lib.log import ProgressMeter

logger = logging.getLogger(__name__)


class AnalysisBase(base.AnalysisBase):
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
        super(AnalysisBase, self).__init__(self, **kwargs)

        self._trajectory = trajectory
        self._verbose = verbose
        self._save = save
        self.results = {}

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
            time step (ps) to read analysis frame
        """
        self._trajectory = trajectory

        self.begin = begin
        self.end = end
        self.dt = dt

        with warnings.catch_warnings():
            warnings.simplefilter('always')
            if self.begin > trajectory.totaltime:
                raise ValueError("Start ({:.2f} ps) is larer than total time "
                                 "({:.2f} ps).".format(self.begin,
                                                       trajectory.totaltime))
            elif self.begin > 0:
                startframe = int(begin // trajectory.dt)
            else:
                startframe = 0
            if self.end is not None:
                stopframe = int(end // trajectory.dt)
                self.end += 1  # catch also last frame in loops
            else:
                stopframe = None
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
        interval = int(self.n_frames // 100)
        if interval == 0:
            interval = 1

        verbose = getattr(self, '_verbose', False)
        self._pm = ProgressMeter(
            self.n_frames if self.n_frames else 1,
            interval=interval,
            verbose=verbose)

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
        stop : float, optional
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
        for i, ts in enumerate(
                self._trajectory[self.startframe:self.stopframe:self.step]):
            self._frame_index = i
            self._ts = ts
            # logger.info("--> Doing frame {} of {}".format(i+1, self.n_frames))
            self._single_frame()
            self._pm.echo(self._frame_index)
        logger.info("Finishing up")
        self._calculate_results()
        self._conclude()
        if self._save:
            self._save_results()
        return self


class SingleGroupAnalysisBase(AnalysisBase):
    """The base class for analysing a single AtomGroup only."""

    _allow_multiple_atomgroups = False

    def __init__(self, atomgroup, **kwargs):
        super(SingleGroupAnalysisBase, self).__init__(
            atomgroup.universe.trajectory, **kwargs)
        self.atomgroup = atomgroup
        self._universe = atomgroup.universe


class MultiGroupAnalysisBase(AnalysisBase):
    """The base class for analysing a single or multiple AtomGroups."""

    _allow_multiple_atomgroups = True

    def __init__(self, atomgroups, **kwargs):
        if type(atomgroups) not in [list, tuple, np.ndarray]:
            atomgroups = [atomgroups]
        else:
            #Check that all atomgroups are from same universe
            for ag in atomgroups[:1]:
                if ag.universe != atomgroups[0].universe:
                    raise ValueError(
                        "Given Atomgroups are not from the same Universe.")
        super(MultiGroupAnalysisBase, self).__init__(
            atomgroups[0].universe.trajectory, **kwargs)

        self.atomgroups = atomgroups
        self._universe = atomgroups[0].universe
