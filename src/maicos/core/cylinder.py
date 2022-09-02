#!/usr/bin/env python3
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
#
# Copyright (c) 2022 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the GNU Public Licence, v3 or any higher version
# SPDX-License-Identifier: GPL-3.0-or-later
"""Base class for cylindrical analysis."""

import logging

import numpy as np

from ..lib.util import render_docs
from .planar import PlanarBase


logger = logging.getLogger(__name__)


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
    pos_cyl : numpy.ndarray
        positions in cylinder coordinats (r, phi, z)
    binwidth : float
        The actual binwidth taking the length of the changing box into account.
    results.R : float
        average length along the radial dimension
    results.binarea : numpy.ndarray
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
        positions : numpy.ndarray
            Cartesian coordinates (x,y,z)

        Returns
        -------
        trans_positions : numpy.ndarray
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
    profile_cum : numpy.ndarray
        cumulative profile
    profile_cum_sq : numpy.ndarray
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

        if self.normalization == 'number':
            self.tot_bincount = np.zeros((self.n_bins, self.n_atomgroups))

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

            profile, _, _ = np.histogram2d(positions[:, 0],
                                           positions[:, 2],
                                           bins=(self.n_bins, 1),
                                           range=((self.rmin, self.rmax),
                                                  (self.zmin, self.zmax)),
                                           weights=weights)

            # Reshapee into 1D array
            profile = profile[:, 0]

            if self.normalization == 'number':
                # Use the 2D histogram function to perform the selection in
                # the z dimension.
                bincount, _, _ = np.histogram2d(positions[:, 0],
                                                positions[:, 2],
                                                bins=(self.n_bins, 1),
                                                range=((self.rmin, self.rmax),
                                                       (self.zmin, self.zmax)))
                # Reshape into 1D array
                bincount = bincount[:, 0]

                self.tot_bincount[:, index] += bincount

                # If a bin does not contain any particles we divide by 0.
                with np.errstate(invalid='ignore'):
                    profile /= bincount
                profile = np.nan_to_num(profile)
            elif self.normalization == "volume":
                profile /= self.results.frame.binarea * self.results.frame.L

            self.results.frame.profile[:, index] = profile

    def _conclude(self):
        super(ProfileCylinderBase, self)._conclude()

        self.results.profile_mean = self.results.means.profile
        self.results.profile_err = self.results.sems.profile

        if self.normalization == 'number':
            no_occurences_idx = self.tot_bincount == 0
            self.results.profile_mean[no_occurences_idx] = np.nan
            self.results.profile_err[no_occurences_idx] = np.nan

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
