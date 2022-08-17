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

from ..decorators import render_docs
from ..utils import (
    atomgroup_header,
    check_compound,
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
    All results should be stored as attributes of the :class:`Results`
    container.

    Parameters
    ----------
    atomgroups : Atomgroup or list[Atomgroup]
        Atomgroups taken for the Analysis
    ${BASE_CLASS_PARAMETERS}
    multi_group : bool
        Analysis is able to work with list of atomgroups

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
                 concfreq=0,
                 verbose=False,
                 **kwargs):
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

        super(AnalysisBase, self).__init__(trajectory=self._trajectory,
                                           verbose=verbose,
                                           **kwargs)

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


@render_docs
class PlanarBase(AnalysisBase):
    """Class to provide options and attributes for analysis in planar system.

    Parameters
    ----------
    atomgroups : Atomgroup or list[Atomgroup]
        Atomgroups taken for the Analysis
    ${PLANAR_CLASS_PARAMETERS}
    kwargs : dict
        Parameters parsed to `AnalysisBase`.

    Attributes
    ----------
    ${PLANAR_CLASS_ATTRIBUTES}
    zmin : float
         Minimal coordinate for evaluation (Å) with in the lab frame, where
         0 corresponds to the origin of the cell.
    zmax : float
         Maximal coordinate for evaluation (Å) with in the lab frame, where
         0 corresponds to the origin of the cell.
    binwidth : float
        The actual binwidth taking the length of the changing box into account.
    odims : np.ndarray
        other dimensions perpendicular to `dim` i.e. (0,2) if `dim = 1`
    results.L : float
        average length along the chosen dimension
    """

    def __init__(self,
                 atomgroups,
                 dim,
                 zmin,
                 zmax,
                 binwidth,
                 **kwargs):
        super(PlanarBase, self).__init__(atomgroups=atomgroups, **kwargs)

        if dim not in [0, 1, 2]:
            raise ValueError("Dimension can only be x=0, y=1 or z=2.")
        else:
            self.dim = dim

        # These values are requested by the user,
        # but the actual ones are calculated during runtime in the lab frame
        self._zmax = zmax
        self._zmin = zmin
        self._binwidth = binwidth

    @property
    def odims(self):
        """Directions perpendicular to dim."""
        return np.roll(np.arange(3), -self.dim)[1:]

    def _compute_lab_frame_planar(self):
        """Compute lab limits `zmin` and `zmax`."""
        if self._zmin is None:
            self.zmin = 0
        else:
            self.zmin = self.box_center[self.dim] + self._zmin

        if self._zmax is None:
            self.zmax = self._universe.dimensions[self.dim]
        else:
            self.zmax = self.box_center[self.dim] + self._zmax

    def _prepare(self):
        """Prepare the planar analysis."""
        self._compute_lab_frame_planar()

        # TODO: There are much more wrong combinations of zmin and zmax...
        if self._zmax is not None and self._zmin is not None \
           and self._zmax <= self._zmin:
            raise ValueError("`zmax` can not be smaller or equal than `zmin`!")

        try:
            if self._binwidth > 0:
                L = self.zmax - self.zmin
                self.n_bins = int(np.ceil(L / self._binwidth))
            else:
                raise ValueError("Binwidth must be a positive number.")
        except TypeError:
            raise ValueError("Binwidth must be a number.")

        logger.info(f"Using {self.n_bins} bins")

    def _single_frame(self):
        """Single frame for the planar analysis."""
        self._compute_lab_frame_planar()
        self.results.frame.L = self.zmax - self.zmin

    def _conclude(self):
        """Results calculations for the planar analysis."""
        self.L = self.results.means.L

        if self._zmin is None:
            zmin = -self.L / 2
        else:
            zmin = self._zmin

        if self._zmax is None:
            zmax = self.L / 2
        else:
            zmax = self._zmax

        self.binwidth = self.L / self.n_bins
        self.results.z = np.linspace(zmin, zmax, self.n_bins) \
            + self.binwidth / 2


@render_docs
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
    ${PROFILE_PLANAR_CLASS_PARAMETERS}
    f_kwargs : dict
        Additional parameters for `function`

    Attributes
    ----------
    ${PROFILE_PLANAR_CLASS_ATTRIBUTES}
    profile_cum : np.ndarray
        cumulative profile
    profile_cum_sq : np.ndarray
        cumulative squared profile
    """

    def __init__(self,
                 function,
                 normalization,
                 atomgroups,
                 sym,
                 grouping,
                 binmethod,
                 output,
                 f_kwargs=None,
                 **kwargs):
        super(ProfilePlanarBase, self).__init__(atomgroups=atomgroups,
                                                multi_group=True,
                                                **kwargs)
        if f_kwargs is None:
            f_kwargs = {}

        self.function = lambda ag, grouping, dim: function(
            ag, grouping, dim, **f_kwargs)
        self.normalization = normalization.lower()
        self.sym = sym
        self.grouping = grouping.lower()
        self.binmethod = binmethod.lower()
        self.output = output

    def _prepare(self):
        super(ProfilePlanarBase, self)._prepare()

        if self.normalization not in ["none", "volume", "number"]:
            raise ValueError(f"`{self.normalization}` not supported. "
                             "Use `None`, `volume` or `number`.")

        if self.sym and self.refgroup is None:
            raise ValueError("For symmetrization the `refgroup` argument is "
                             "required.")

        if self.grouping not in ["atoms", "segments", "residues", "molecules",
                                 "fragments"]:
            raise ValueError(f"{self.grouping} is not a valid option for "
                             "grouping. Use 'atoms', 'residues', "
                             "'segments', 'molecules' or 'fragments'.")

        if self.unwrap and self.grouping == "atoms":
            logger.warning("Unwrapping in combination with atom grouping "
                           "is superfluous. `unwrap` will be set to `False`.")
            self.unwrap = False

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


@render_docs
class CylinderBase(PlanarBase):
    """Class to provide options and attributes for analysis in cylinder system.

    Provide the results attribute `r`.

    Parameters
    ----------
    atomgroups : Atomgroup or list[Atomgroup]
        Atomgroups taken for the Analysis
    ${CYLINDER_CLASS_PARAMETERS}
    kwargs : dict
        Parameters parsed to `AnalysisBase`.

    Attributes
    ----------
    ${CYLINDER_CLASS_ATTRIBUTES}
    pos_cyl : np.ndarray
        positions in cylinder coordinats (r, phi, z)
    binwidth : float
        The actual binwidth taking the length of the changing box into account.
    results.R : float
        average length along the radial dimension
    results.binarea : np.ndarray
        area of the concentrtic radial bins. Calculated via
        :math:`r_{i+1}^2 - r_i^2)` where `i` is the index of the bin.
    """

    def __init__(self,
                 atomgroups,
                 rmin,
                 rmax,
                 **kwargs):
        super(CylinderBase, self).__init__(atomgroups, **kwargs)

        self.rmin = rmin
        self._rmax = rmax

    def _compute_lab_frame_cylinder(self):
        """Compute lab limit `rmax`."""
        if self._rmax is None:
            self.rmax = self._universe.dimensions[self.odims.min()] / 2
        else:
            self.rmax = self._rmax

        # Transform into cylinder coordinates
        self.pos_cyl = self.transform_positions(self._universe.atoms.positions)

    def _prepare(self):
        """Prepare the cylinder analysis."""
        super(CylinderBase, self)._prepare()

        self._compute_lab_frame_cylinder()

        if self.rmin < 0:
            raise ValueError("Only values for rmin largere 0 are allowed.")

        if self._rmax is not None and self._rmax <= self.rmin:
            raise ValueError("`rmax` can not be smaller or equal than `rmin`!")

        try:
            if self._binwidth > 0:
                R = self.rmax - self.rmin
                self.n_bins = int(np.ceil(R / self._binwidth))
            else:
                raise ValueError("Binwidth must be a positive number.")
        except TypeError:
            raise ValueError("Binwidth must be a number.")

        logger.info(f"Using {self.n_bins} bins")

    def transform_positions(self, positions):
        """Transform positions into cylinder coordinates.

        The origin of th coordinate system is at
        :attr:`AnalysisBase.box_center`. And the direction of the
        cylinder defined by :attr:`self.dim`.

        Parameters
        ----------
        positions : np.ndarray
            Cartesian coordinates (x,y,z)

        Returns
        -------
        trans_positions : np.ndarray
            Positions in cylinder coordinates (r, phi, z)
        """
        trans_positions = np.zeros(positions.shape)

        # z component
        trans_positions[:, 2] = np.copy(positions[:, self.dim])

        # shift origin to box center
        pos_xyz_center = positions - self.box_center

        # r component
        trans_positions[:, 0] = np.linalg.norm(pos_xyz_center[:, self.odims],
                                               axis=1)

        # phi component
        trans_positions[:, 1] = np.arctan2(*pos_xyz_center[:, self.odims].T)

        return trans_positions

    def _single_frame(self):
        """Single frame for the cylinder analysis."""
        super(CylinderBase, self)._single_frame()
        self._compute_lab_frame_cylinder()
        self.results.frame.R = self.rmax - self.rmin

        r = np.linspace(self.rmin, self.rmax, self.n_bins + 1, endpoint=True)
        self.results.frame.binarea = np.pi * np.diff(r**2)

    def _conclude(self):
        """Results calculations for the cylinder analysis."""
        super(CylinderBase, self)._conclude()

        # Remove not used z attribute from PlanarBase
        del self.results.z

        self.R = self.results.means.R
        self.binarea = self.results.means.binarea

        if self._rmax is None:
            rmax = self.R
        else:
            rmax = self._rmax

        self.binwidth = self.rmax / self.n_bins
        self.results.r = np.linspace(self.rmin, rmax, self.n_bins) \
            + self.binwidth / 2


@render_docs
class ProfileCylinderBase(CylinderBase):
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
    ${PROFILE_CYLINDER_CLASS_PARAMETERS}
    f_kwargs : dict
        Additional parameters for `function`

    Attributes
    ----------
    ${PROFILE_CYLINDER_CLASS_ATTRIBUTES}
    profile_cum : np.ndarray
        cumulative profile
    profile_cum_sq : np.ndarray
        cumulative squared profile
    """

    def __init__(self,
                 function,
                 normalization,
                 atomgroups,
                 grouping,
                 binmethod,
                 output,
                 f_kwargs=None,
                 **kwargs):
        super(ProfileCylinderBase, self).__init__(atomgroups=atomgroups,
                                                  multi_group=True,
                                                  **kwargs)
        if f_kwargs is None:
            f_kwargs = {}

        self.function = lambda ag, grouping, dim: function(
            ag, grouping, dim, **f_kwargs)
        self.normalization = normalization.lower()
        self.grouping = grouping.lower()
        self.binmethod = binmethod.lower()
        self.output = output

    def _prepare(self):
        super(ProfileCylinderBase, self)._prepare()

        if self.normalization not in ["none", "volume", "number"]:
            raise ValueError(f"`{self.normalization}` not supported. "
                             "Use `None`, `volume` or `number`.")

        if self.grouping not in ["atoms", "segments", "residues", "molecules",
                                 "fragments"]:
            raise ValueError(f"{self.grouping} is not a valid option for "
                             "grouping. Use 'atoms', 'residues', "
                             "'segments', 'molecules' or 'fragments'.")

        if self.unwrap and self.grouping == "atoms":
            logger.warning("Unwrapping in combination with atom grouping "
                           "is superfluous. `unwrap` will be set to `False`.")
            self.unwrap = False

        if self.binmethod not in ["cog", "com", "coc"]:
            raise ValueError(f"{self.binmethod} is an unknown binning "
                             "method. Use `cog`, `com` or `coc`.")

        logger.info(f"Computing {self.grouping} radial cylinder profile along "
                    f"{'XYZ'[self.dim]}-axes.")

        # Arrays for accumulation
        self.results.frame.profile = np.zeros((self.n_bins, self.n_atomgroups))

    def _single_frame(self):
        super(ProfileCylinderBase, self)._single_frame()

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

            positions = self.transform_positions(positions)
            weights = self.function(selection, self.grouping, self.dim)

            profile_ts, _, _ = np.histogram2d(positions[:, 0],
                                              positions[:, 2],
                                              bins=(self.n_bins, 1),
                                              range=((self.rmin, self.rmax),
                                                     (self.zmin, self.zmax)),
                                              weights=weights)

            # Reshapee into 1D array
            profile_ts = profile_ts[:, 0]

            if self.normalization == 'number':
                # Use the 2D histogram function to perform the selection in
                # the z dimension.
                bincount, _, _ = np.histogram2d(positions[:, 0],
                                                positions[:, 2],
                                                bins=(self.n_bins, 1),
                                                range=((self.rmin, self.rmax),
                                                       (self.zmin, self.zmax)))
                # Reshapee into 1D array
                bincount = bincount[:, 0]

                # If a bin does not contain any particles we divide by 0.
                with np.errstate(invalid='ignore'):
                    profile_ts /= bincount
                profile_ts = np.nan_to_num(profile_ts)
            elif self.normalization == "volume":
                profile_ts /= self.results.frame.binarea * self.results.frame.L

            self.results.frame.profile[:, index] = profile_ts

    def _conclude(self):
        super(ProfileCylinderBase, self)._conclude()

        self.results.profile_mean = self.results.means.profile
        self.results.profile_err = self.results.sems.profile

    def save(self):
        """Save results of analysis to file."""
        columns = ["radial positions [Å]"]

        for i, _ in enumerate(self.atomgroups):
            columns.append(f'({i + 1}) profile')
        for i, _ in enumerate(self.atomgroups):
            columns.append(f'({i + 1}) error')

        self.savetxt(self.output, np.hstack(
                     (self.results.r[:, np.newaxis],
                      self.results.profile_mean,
                      self.results.profile_err)),
                     columns=columns)
