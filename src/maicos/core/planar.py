#!/usr/bin/env python3
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
#
# Copyright (c) 2022 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the GNU Public Licence, v3 or any higher version
# SPDX-License-Identifier: GPL-3.0-or-later
"""Base class for planar analysis."""

import logging

import numpy as np

from ..lib.math import symmetrize
from ..lib.util import render_docs
from .base import AnalysisBase


logger = logging.getLogger(__name__)


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
        """Other dimensions perpendicular to dim i.e. (0,2) if dim = 1."""
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
        It must take an `Atomgroup` as first argument and a
        grouping ('atoms', 'residues', 'segments', 'molecules', 'fragments')
        as second. Additional parameters can
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
    profile_cum : numpy.ndarray
        cumulative profile
    profile_cum_sq : numpy.ndarray
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

        self.function = lambda ag: function(ag, grouping, **f_kwargs)
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
        self.results.frame.profile = np.zeros((self.n_bins, self.n_atomgroups))

        if self.normalization == 'number':
            self.tot_bincount = np.zeros((self.n_bins, self.n_atomgroups))

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
            weights = self.function(selection)

            profile, _ = np.histogram(positions,
                                      bins=self.n_bins,
                                      range=(self.zmin, self.zmax),
                                      weights=weights)

            if self.normalization == 'number':
                bincount, _ = np.histogram(positions,
                                           bins=self.n_bins,
                                           range=(self.zmin, self.zmax))

                self.tot_bincount[:, index] += bincount

                # If a bin does not contain any particles we divide by 0.
                with np.errstate(invalid='ignore'):
                    profile /= bincount
                profile = np.nan_to_num(profile)
            elif self.normalization == "volume":
                profile /= self._ts.volume / self.n_bins

            self.results.frame.profile[:, index] = profile

    def _conclude(self):
        super(ProfilePlanarBase, self)._conclude()

        self.results.profile_mean = self.results.means.profile
        self.results.profile_err = self.results.sems.profile

        if self.sym:
            symmetrize(self.results.profile_mean, inplace=True)
            symmetrize(self.results.profile_err, inplace=True)

            if self.normalization == 'number':
                symmetrize(self.tot_bincount, inplace=True)

        if self.normalization == 'number':
            no_occurences_idx = self.tot_bincount == 0
            self.results.profile_mean[no_occurences_idx] = np.nan
            self.results.profile_err[no_occurences_idx] = np.nan

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
