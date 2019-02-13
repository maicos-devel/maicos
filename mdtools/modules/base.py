#!/usr/bin/env python3
# coding: utf-8

import logging

from MDAnalysis.analysis import base

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

    def _configure_parser(self, parser):
        """Adds parser options using an argparser object"""
        parser.description = self.__doc__

    def _calculate_results(self):
        """Calculate the results"""
        pass

    def _save_results(self):
        """Saves the results you've gatherd to a file."""
        pass

    def run(self, start=None, stop=None, step=None, verbose=None):
        """Perform the calculation

        Parameters
        ----------
        start : int, optional
            start frame of analysis
        stop : int, optional
            stop frame of analysis
        step : int, optional
            number of frames to skip between each analysed frame
        verbose : bool, optional
            Turn on verbosity
        """
        logger.info("Choosing frames to analyze")
        # if verbose unchanged, use class default
        verbose = getattr(self, '_verbose',
                          False) if verbose is None else verbose

        self._setup_frames(self._trajectory, start, stop, step)
        logger.info("Starting preparation")
        self._prepare()
        for i, ts in enumerate(
                self._trajectory[self.start:self.stop:self.step]):
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
