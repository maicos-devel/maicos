#!/usr/bin/env python3
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
#
# Copyright (c) 2022 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the GNU Public Licence, v3 or any higher version
# SPDX-License-Identifier: GPL-3.0-or-later
"""Base class."""

import inspect
import logging
import warnings
from datetime import datetime

import MDAnalysis.analysis.base
import numpy as np
from MDAnalysis.analysis.base import Results
from MDAnalysis.lib.log import ProgressBar

from ..decorators import (
    make_whole,
    set_planar_class_doc,
    set_profile_planar_class_doc,
    set_verbose_doc,
    )
from ..utils import (
    atomgroup_header,
    cluster_com,
    correlation_time,
    get_cli_input,
    new_mean,
    new_variance,
    sort_atomgroup,
    symmetrize,
    )
from ..version import __version__


logger = logging.getLogger(__name__)


@set_verbose_doc
class AnalysisBase(MDAnalysis.analysis.base.AnalysisBase):
    """Base class derived from MDAnalysis for defining multi-frame analysis.

    The class is designed as a template for creating multi-frame analyses.
    This class will automatically take care of setting up the trajectory
    reader for iterating, and it offers to show a progress meter.
    Computed results are stored inside the :attr:`results` attribute.
    To define a new analysis, `AnalysisBase` needs to be subclassed
    and :meth:`_single_frame` must be defined. It is also possible to define
    :meth:`_prepare` and :meth:`_conclude` for pre- and post-processing.
    All results should be stored as attributes of the :class:`Results`
    container.

    Parameters
    ----------
    atomgroups : Atomgroup or list[Atomgroup]
        Atomgroups taken for the Analysis
    multi_group : bool
        Analysis is able to work with list of atomgroups
    concfreq : int
        Call the conclcude function and write the output files every n frames
    ${VERBOSE_PARAMETER}

    Attributes
    ----------
    atomgroup : mda.Atomgroup
        Atomgroup taken for the Analysis (available if `multi_group = False`)
    atomgroups : list[mda.Atomgroup]
        Atomgroups taken for the Analysis (available if `multi_group = True`)
    n_atomgroups : int
        Number of atomngroups (available if `multi_group = True`)
    _universe : mda.Universe
        The Universe the atomgroups belong to
    _trajectory : mda.trajectory
        The trajetcory the atomgroups belong to
    times : numpy.ndarray
        array of Timestep times. Only exists after calling
        :meth:`AnalysisBase.run`
    frames : numpy.ndarray
        array of Timestep frame indices. Only exists after calling
        :meth:`AnalysisBase.run`
    results : :class:`Results`
        results of calculation are stored after call
        to :meth:`AnalysisBase.run`
    """

    def __init__(self,
                 atomgroups,
                 multi_group=False,
                 verbose=False,
                 concfreq=0,
                 **kwargs):
        # Get the arguments to reproduce the analysis
        frame = inspect.currentframe()
        try:
            # We want the locals of the __init__ of the child class, so we
            # traverse the stack until we find the frame of the child class
            while not frame.f_locals['__class__'] is self.__class__:
                frame = frame.f_back
            self.locals = frame.f_locals
        except KeyError:
            # Could not find the corresponding __init__ method
            # This should never happen
            self.locals = None
        finally:
            del frame

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
        self.concfreq = concfreq

        super(AnalysisBase, self).__init__(trajectory=self._trajectory,
                                           verbose=verbose,
                                           **kwargs)

    def run(self, start=None, stop=None, step=None, verbose=None):
        """Iterate over the trajectory."""
        self.run_params = {'start': start, 'stop': stop, 'step': step}
        logger.info("Choosing frames to analyze")
        # if verbose unchanged, use class default
        verbose = getattr(self, '_verbose',
                          False) if verbose is None else verbose

        self._setup_frames(self._trajectory, start, stop, step)
        logger.info("Starting preparation")

        self.results.frame = Results()

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
            # logger.info("--> Doing frame {} of {}".format(i+1, self.n_frames))

            timeseries[i] = self._single_frame()

            try:
                for key in self.results.frame.keys():
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
                    self.results.sems[key] = \
                        np.zeros(self.results.frame[key].shape)

            if self.concfreq and self._index % self.concfreq == 0 \
               and self._frame_index > 0:
                self._conclude()
                if module_has_save:
                    self.save()

        logger.info("Finishing up")

        if len(timeseries > 4) and (timeseries[0] is not None):
            corrtime = correlation_time(timeseries)
            if corrtime == -1:
                warnings.warn("Your trajectory does not provide sufficient "
                              "statistics to estimate a correlation time. "
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
        if hasattr(self, 'OUTPUT'):
            messages.append(self.OUTPUT)
        messages = '\n'.join(messages)

        # Get information on the analyzed atomgroup
        atomgroups = ''
        if self._allow_multiple_atomgroups:
            for i, ag in enumerate(self.atomgroups):
                atomgroups += f"  ({i + 1}) {atomgroup_header(ag)}\n"
        else:
            atomgroups += f"  (1) {atomgroup_header(self.atomgroup)}\n"

        params = inspect.signature(self.__init__).parameters.copy()
        params.pop('kwargs')
        values = [f'{param}={self.locals[param]}' for param in params.keys()]
        run_params = ', '.join(
            [f'{key}={value}' for key, value in self.run_params.items()]
            )

        header = (
            f"This file was generated by {module_name} on {current_time}\n\n"
            f"{module_name} is part of MAICoS v{__version__}\n\n"
            f"Command line:"
            f"    {get_cli_input()}\n"
            f"Module input:"
            f"    {module_name}({', '.join(values)})"
            f".run({run_params})\n\n"
            f"Statistics over {self._index} frames\n\n"
            f"Considered atomgroups:\n"
            f"{atomgroups}\n"
            f"{messages}\n\n"
            )

        if columns is not None:
            header += '|'.join([f"{i:^26}"for i in columns])[2:]

        fname = "{}{}".format(fname, (not fname.endswith('.dat')) * '.dat')
        np.savetxt(fname, X, header=header, fmt='% .18e ')


@set_planar_class_doc
class PlanarBase(AnalysisBase):
    """Class to provide options and attributes for analysis in planar system.

    Provide the results attribute `z`.

    Parameters
    ----------
    trajectory : MDAnalysis.coordinates.base.ReaderBase
        A trajectory Reader
    ${PLANAR_CLASS_PARAMETERS}
    kwargs : dict
        Parameters parsed to `AnalysisBase`.

    Attributes
    ----------
    ${PLANAR_CLASS_ATTRIBUTES}
    zmax : float
        the maximal coordinate for evaluation. If provided `zmax` is
        `None` it will adjust to box length during analysis.
    n_bins : int
        Number of bins for analysis
    """

    def __init__(self,
                 atomgroups,
                 dim,
                 zmin,
                 zmax,
                 binwidth,
                 comgroup,
                 **kwargs):
        super(PlanarBase, self).__init__(atomgroups, **kwargs)
        self.dim = dim
        self.zmin = zmin

        # These values are requested by the user,
        # but the actual ones are calculated during runtime
        self._zmax = zmax
        self._binwidth = binwidth

        self.comgroup = comgroup

    def _prepare(self):
        """Prepare the planar analysis."""
        if self.dim not in [0, 1, 2]:
            raise ValueError("Dimension can only be x=0, y=1 or z=2.")

        if self._zmax is None:
            self.L_cum = 0
            self.zmax = self._universe.dimensions[self.dim]
        else:
            self.zmax = self._zmax
        try:
            if self._binwidth > 0:
                self.n_bins = int(np.ceil((self.zmax - self.zmin)
                                          / self._binwidth))
            else:
                raise ValueError("Binwidth must be a positive number.")
        except TypeError:
            raise ValueError("Binwidth must be a number.")

        logger.info(f"Using {self.n_bins} bins")

        if self.comgroup is not None and self.comgroup.n_atoms == 0:
            raise ValueError("Comgroup does not contain any atoms.")

    def _single_frame(self):
        """Single frame for the planar analysis."""
        if self._zmax is None:
            self.zmax = self._ts.dimensions[self.dim]
            self.L_cum += self.zmax
        if self.comgroup is not None:
            center_of_box = self._universe.dimensions[:3] / 2
            center_of_box[self.dim] = (self.zmax - self.zmin) / 2

            com_comgroup = cluster_com(self.comgroup)
            t = center_of_box - com_comgroup
            self._universe.atoms.translate(t)
            self._universe.atoms.wrap()

    def _conclude(self):
        """Results calculations for the planar analysis."""
        if self._zmax is None:
            zmax = self.L_cum / (self._frame_index + 1)
        else:
            zmax = self.zmax

        self.binwidth = (zmax - self.zmin) / self.n_bins

        self.results.z = np.linspace(
            self.zmin + self.binwidth / 2,
            zmax - self.binwidth / 2, self.n_bins,
            endpoint=True)

        if self.comgroup:
            self.results.z -= self.zmin + (zmax - self.zmin) / 2


@set_verbose_doc
@set_profile_planar_class_doc
@make_whole()
class ProfilePlanarBase(PlanarBase):
    """Base class for computing profiles in a cartesian geometry.

    Parameters
    ----------
    function : callable
        The function calculating the array for the analysis.
        It must take an `Atomgroup` as first argument,
        grouping ('atoms', 'residues', 'segments', 'molecules', 'fragments')
        as second and a dimension (0, 1, 2) as third. Additional parameters can
        be given as `f_kwargs`. The function must return a numpy.ndarry with
        the same length as the number of group members.
    normalization : str {'None', 'number', 'volume'}
        The normalization of the profile performed in every frame.
        If `None` no normalization is performed. If `number` the histogram
        is divided by the number of occurences in each bin. If `volume` the
        profile is divided by the volume of each bin.
    ${PLANAR_PROFILE_CLASS_PARAMETERS}
    f_kwargs : dict
        Additional parameters for `function`
    ${VERBOSE_PARAMETER}

    Attributes
    ----------
    ${PLANAR_PROFILE_CLASS_ATTRIBUTES}
    """

    def __init__(self,
                 function,
                 normalization,
                 atomgroups,
                 dim,
                 zmin,
                 zmax,
                 binwidth,
                 comgroup,
                 sym,
                 grouping,
                 make_whole,
                 binmethod,
                 output,
                 f_kwargs=None,
                 **kwargs):
        super(ProfilePlanarBase, self).__init__(atomgroups=atomgroups,
                                                dim=dim,
                                                zmin=zmin,
                                                zmax=zmax,
                                                binwidth=binwidth,
                                                comgroup=comgroup,
                                                multi_group=True,
                                                **kwargs)
        if f_kwargs is None:
            f_kwargs = {}

        self.function = lambda ag, grouping, dim: function(
            ag, grouping, dim, **f_kwargs)
        self.normalization = normalization.lower()
        self.sym = sym
        self.grouping = grouping.lower()
        self.make_whole = make_whole
        self.binmethod = binmethod.lower()
        self.output = output

    def _prepare(self):
        super(ProfilePlanarBase, self)._prepare()

        if self.normalization not in ["none", "volume", "number"]:
            raise ValueError(f"`{self.normalization}` not supported. "
                             "Use `None`, `volume` or `number`.")

        if self.sym and self.comgroup is None:
            raise ValueError("For symmetrization the `comgroup` argument is "
                             "required.")

        if self.grouping not in ["atoms", "segments", "residues", "molecules",
                                 "fragments"]:
            raise ValueError(f"{self.grouping} is not a valid option for "
                             "grouping. Use 'atoms', 'residues', "
                             "'segments', 'molecules' or 'fragments'.")

        if self.make_whole and self.grouping == "atoms":
            logger.warning("Making molecules whole in combination with atom "
                           "grouping is superfluous. `make_whole` will be set "
                           "to `False`.")
            self.make_whole = False

        if self.binmethod not in ["cog", "com", "coc"]:
            raise ValueError(f"{self.binmethod} is an unknown binning "
                             "method. Use `cog`, `com` or `coc`.")

        logger.info(f"Computing {self.grouping} profile along "
                    f"{'XYZ'[self.dim]}-axes.")

        # Arrays for accumulation
        self.results.frame.profile = np.zeros((self.n_bins,
                                               self.n_atomgroups))

    def _single_frame(self):
        super(ProfilePlanarBase, self)._single_frame()

        for index, selection in enumerate(self.atomgroups):
            if self.grouping == 'atoms':
                positions = selection.atoms.positions
            else:
                kwargs = dict(compound=self.grouping)
                if self.binmethod == "cog":
                    positions = selection.atoms.center_of_geometry(**kwargs)
                elif self.binmethod == "com":
                    positions = selection.atoms.center_of_mass(**kwargs)
                elif self.binmethod == "coc":
                    positions = selection.atoms.center_of_charge(**kwargs)

            positions = positions[:, self.dim]
            weights = self.function(selection, self.grouping, self.dim)

            profile_ts, _ = np.histogram(positions,
                                         bins=self.n_bins,
                                         range=(self.zmin, self.zmax),
                                         weights=weights)

            if self.normalization == 'number':
                bincount, _ = np.histogram(positions,
                                           bins=self.n_bins,
                                           range=(self.zmin, self.zmax))
                # If a bin does not contain any particles we divide by 0.
                with np.errstate(invalid='ignore'):
                    profile_ts /= bincount
                profile_ts = np.nan_to_num(profile_ts)
            elif self.normalization == "volume":
                profile_ts /= self._ts.volume / self.n_bins

            self.results.frame.profile[:, index] = profile_ts

    def _conclude(self):
        super(ProfilePlanarBase, self)._conclude()

        self.results.profile_mean = self.results.means.profile
        self.results.profile_err = self.results.sems.profile

        if self.sym:
            symmetrize(self.results.profile_mean, inplace=True)
            symmetrize(self.results.profile_err, inplace=True)

    def save(self):
        """Save results of analysis to file."""
        columns = ["positions [Å]"]

        for i, _ in enumerate(self.atomgroups):
            columns.append(f'({i + 1}) profile')
        for i, _ in enumerate(self.atomgroups):
            columns.append(f'({i + 1}) error')

        self.savetxt(self.output, np.hstack(
                     (self.results.z[:, np.newaxis],
                      self.results.profile_mean,
                      self.results.profile_err)),
                     columns=columns)
