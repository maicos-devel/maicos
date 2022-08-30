#!/usr/bin/env python3
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
#
# Copyright (c) 2022 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the GNU Public Licence, v3 or any higher version
# SPDX-License-Identifier: GPL-3.0-or-later
"""Base class for building Analysis classes."""

import inspect
import logging
import warnings
from datetime import datetime

import MDAnalysis.analysis.base
import numpy as np
from MDAnalysis.analysis.base import Results
from MDAnalysis.lib.log import ProgressBar

from ..lib.math import cluster_com, correlation_time, new_mean, new_variance
from ..lib.util import (
    atomgroup_header,
    check_compound,
    get_cli_input,
    render_docs,
    sort_atomgroup,
    )
from ..version import __version__


logger = logging.getLogger(__name__)


@render_docs
class AnalysisBase(MDAnalysis.analysis.base.AnalysisBase):
    """Base class derived from MDAnalysis for defining multi-frame analysis.

    The class is designed as a template for creating multi-frame analyses.
    This class will automatically take care of setting up the trajectory
    reader for iterating, and it offers to show a progress meter.
    Computed results are stored inside the :attr:`results` attribute.
    To define a new analysis, `AnalysisBase` needs to be subclassed
    and :meth:`_single_frame` must be defined. It is also possible to define
    :meth:`_prepare` and :meth:`_conclude` for pre- and post-processing.
    All results should be stored as attributes of the
    :class:`MDAnalysis.analysis.base.Results` container.

    Parameters
    ----------
    atomgroups : MDAnalysis.core.groups.AtomGroup or list[AtomGroup]
        Atomgroups taken for the Analysis
    ${BASE_CLASS_PARAMETERS}
    multi_group : bool
        Analysis is able to work with list of atomgroups

    Attributes
    ----------
    atomgroup : MDAnalysis.core.groups.AtomGroup
        Atomgroup taken for the Analysis (available if `multi_group = False`)
    atomgroups : list[MDAnalysis.core.groups.AtomGroup]
        Atomgroups taken for the Analysis (available if `multi_group = True`)
    n_atomgroups : int
        Number of atomngroups (available if `multi_group = True`)
    _universe : MDAnalysis.core.universe.Universe
        The Universe the atomgroups belong to
    _trajectory : MDAnalysis.coordinates.base.ReaderBase
        The trajetcory the atomgroups belong to
    times : numpy.ndarray
        array of Timestep times. Only exists after calling
        :meth:`AnalysisBase.run`
    frames : numpy.ndarray
        array of Timestep frame indices. Only exists after calling
        :meth:`AnalysisBase.run`
    _frame_index : int
        index of the frame currently analysed
    _index : int
        Number of frames already analysed (same as _frame_index + 1)
    results : :class:`Results`
        results of calculation are stored after call
        to :meth:`AnalysisBase.run`
    results.frame : :class:`Results`
        Observables of the current frame
    results.mean : :class:`Results`
        Means of the observables.
        Keys are the same as :attr:`results.frame`.
    results.vars : :class:`Results`
        Variances of the observables.
        Keys are the same as :attr:`results.frame`
    """

    def __init__(self,
                 atomgroups,
                 multi_group=False,
                 refgroup=None,
                 unwrap=False,
                 concfreq=0):
        if multi_group:
            if type(atomgroups) not in (list, tuple):
                atomgroups = [atomgroups]
            # Check that all atomgroups are from same universe
            if len(set([ag.universe for ag in atomgroups])) != 1:
                raise ValueError("Atomgroups belong to different Universes")

            # Sort the atomgroups,
            # such that molecules are listed one after the other
            self.atomgroups = list(map(sort_atomgroup, atomgroups))
            self.n_atomgroups = len(self.atomgroups)
            self._universe = atomgroups[0].universe
            self._allow_multiple_atomgroups = True
        else:
            self.atomgroup = sort_atomgroup(atomgroups)
            self._universe = atomgroups.universe
            self._allow_multiple_atomgroups = False

        self._trajectory = self._universe.trajectory
        self.refgroup = refgroup
        self.unwrap = unwrap
        self.concfreq = concfreq

        if self.refgroup is not None and self.refgroup.n_atoms == 0:
            raise ValueError("Refgroup does not contain any atoms.")

        super(AnalysisBase, self).__init__(trajectory=self._trajectory)

    @property
    def box_center(self):
        """Center of the simulation cell."""
        return self._universe.dimensions[:3] / 2

    def run(self, start=None, stop=None, step=None, verbose=None):
        """Iterate over the trajectory.

        Parameters
        ----------
        start : int
            start frame of analysis
        stop : int
            stop frame of analysis
        step : int
            number of frames to skip between each analysed frame
        verbose : bool
            Turn on verbosity
        """
        logger.info("Choosing frames to analyze")
        # if verbose unchanged, use class default
        verbose = getattr(self, '_verbose',
                          False) if verbose is None else verbose

        self._setup_frames(self._trajectory, start, stop, step)
        logger.info("Starting preparation")

        self.results.frame = Results()
        compatible_types = [np.ndarray, float, int, list, np.float_, np.int_]

        self._prepare()

        module_has_save = callable(getattr(self.__class__, 'save', None))

        timeseries = np.zeros(self.n_frames)

        for i, ts in enumerate(ProgressBar(
                self._trajectory[self.start:self.stop:self.step],
                verbose=verbose)):
            self._frame_index = i
            self._index = self._frame_index + 1

            self._ts = ts
            self.frames[i] = ts.frame
            self.times[i] = ts.time

            if self.refgroup is not None:
                com_refgroup = cluster_com(self.refgroup)
                t = self.box_center - com_refgroup
                self._universe.atoms.translate(t)
                self._universe.atoms.wrap()

            if self.unwrap:
                if hasattr(self, "atomgroup"):
                    groups = [self.atomgroup]
                else:
                    groups = self.atomgroups
                for group in groups:
                    group.unwrap(compound=check_compound(group))

            timeseries[i] = self._single_frame()

            try:
                for key in self.results.frame.keys():
                    if type(self.results.frame[key]) is list:
                        self.results.frame[key] = \
                            np.array(self.results.frame[key])
                    old_mean = self.results.means[key]
                    old_var = self.results.sems[key]**2 * (self._index - 1)
                    self.results.means[key] = \
                        new_mean(self.results.means[key],
                                 self.results.frame[key], self._index)
                    self.results.sems[key] = \
                        np.sqrt(new_variance(old_var, old_mean,
                                             self.results.means[key],
                                             self.results.frame[key],
                                             self._index) / self._index)
            except AttributeError:
                logger.info("Preparing error estimation.")
                # the results.means and results.sems are not yet defined.
                # We initialize the means with the data from the first frame
                # and set the sems to zero (with the correct shape).
                self.results.means = self.results.frame.copy()
                self.results.sems = Results()
                for key in self.results.frame.keys():
                    if type(self.results.frame[key]) not in compatible_types:
                        raise TypeError(
                            f"Obervable {key} has uncompatible type.")
                    self.results.sems[key] = \
                        np.zeros(np.shape(self.results.frame[key]))

            if self.concfreq and self._index % self.concfreq == 0 \
               and self._frame_index > 0:
                self._conclude()
                if module_has_save:
                    self.save()

        logger.info("Finishing up")

        if len(timeseries > 4) and (timeseries[0] is not None):
            corrtime = correlation_time(timeseries)
            if corrtime == -1:
                warnings.warn("Your trajectory does not provide sufficient"
                              "statistics to estimate a correlation time."
                              "Use the calculated error estimates with"
                              "caution.")
            if corrtime > 0.5:
                warnings.warn("Your data seems to be correlated with a "
                              f"correlation time which is {corrtime + 1:.2f} "
                              "times larger than your step size. "
                              "Consider increasing your step size by a factor "
                              f"of {int(np.ceil(2 * corrtime + 1)):d} to get a "
                              "reasonable error estimate.")

        self._conclude()
        if self.concfreq and module_has_save:
            self.save()
        return self

    def savetxt(self, fname, X, columns=None):
        """Save to text.

        An extension of the numpy savetxt function. Adds the command line
        input to the header and checks for a doubled defined filesuffix.

        Return a header for the text file to save the data to.
        This method builds a generic header that can be used by any MAICoS
        module. It is called by the save method of each module.

        The information it collects is:
          - timestamp of the analysis
          - name of the module
          - version of MAICoS that was used
          - command line arguments that were used to run the module
          - module call including the default arguments
          - number of frames that were analyzed
          - atomgroups that were analyzed
          - output messages from modules and base classes (if they exist)
        """
        # Get the required information first
        current_time = datetime.now().strftime("%a, %b %d %Y at %H:%M:%S ")
        module_name = self.__class__.__name__

        # Here the specific output messages of the modules are collected.
        # We only take into account maicos modules and start at the top of the
        # module tree. Submodules without an own OUTPUT inherit from the parent
        # class, so we want to remove those duplicates.
        messages = []
        for cls in self.__class__.mro()[-3::-1]:
            if hasattr(cls, 'OUTPUT'):
                if cls.OUTPUT not in messages:
                    messages.append(cls.OUTPUT)
        messages = '\n'.join(messages)

        # Get information on the analyzed atomgroup
        atomgroups = ''
        if self._allow_multiple_atomgroups:
            for i, ag in enumerate(self.atomgroups):
                atomgroups += f"  ({i + 1}) {atomgroup_header(ag)}\n"
        else:
            atomgroups += f"  (1) {atomgroup_header(self.atomgroup)}\n"

        header = (
            f"This file was generated by {module_name} on {current_time}\n\n"
            f"{module_name} is part of MAICoS v{__version__}\n\n"
            f"Command line:"
            f"    {get_cli_input()}\n"
            f"Module input:"
            f"    {module_name}{inspect.signature(self.__init__)}"
            f".run({inspect.signature(self.run)})\n\n"
            f"Statistics over {self._index} frames\n\n"
            f"Considered atomgroups:\n"
            f"{atomgroups}\n"
            f"{messages}\n\n"
            )

        if columns is not None:
            header += '|'.join([f"{i:^26}"for i in columns])[2:]

        fname = "{}{}".format(fname, (not fname.endswith('.dat')) * '.dat')
        np.savetxt(fname, X, header=header, fmt='% .18e ')
