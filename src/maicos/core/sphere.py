#!/usr/bin/env python3
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
#
# Copyright (c) 2022 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the GNU Public Licence, v3 or any higher version
# SPDX-License-Identifier: GPL-3.0-or-later
"""Base class for spherical analysis."""

import logging

import numpy as np

from ..lib.util import render_docs
from .planar import AnalysisBase


logger = logging.getLogger(__name__)


@render_docs
class SphereBase(AnalysisBase):
    r"""Analysis class providing options and attributes for spherical system.

    Provide the results attribute `r`.

    Parameters
    ----------
    atomgroups : Atomgroup or list[Atomgroup]
        Atomgroups taken for the Analysis
    ${SPHERE_CLASS_PARAMETERS}
    kwargs : dict
        Parameters parsed to `AnalysisBase`.

    Attributes
    ----------
    ${SPHERE_CLASS_ATTRIBUTES}
    pos_sph : numpy.ndarray
        positions in spherical coordinats (r, phi, theta)
    _obs.R : float
        Average length (in Å) along the radial dimension in the current frame.
    _obs.bin_pos : numpy.ndarray, (n_bins)
        Central bin position of each bin (in Å) in the current frame.
    _obs.bin_width : float
         Bin width (in Å) in the current frame
    _obs.bin_edges : numpy.ndarray, (n_bins + 1)
        Edges of the bins (in Å) in the current frame.
    _obs.bin_area : numpy.ndarray, (n_bins)
        Surface area (in Å^2) of the sphere of each bin with radius `bin_pos`
        in the current frame. Calculated via :math:`4 \pi r_i^2 ` where `i`
        is the index of the bin.
    results.bin_volume : numpy.ndarray, (n_bins)
        volume of a spherical shell of each bins (in Å^3) of the current frame.
        Calculated via :math:`\left 4\pi/3(r_{i+1}^3 - r_i^3 \right)` where `i`
        is the index of the bin.
    """

    def __init__(self,
                 atomgroups,
                 rmin,
                 rmax,
                 bin_width,
                 **kwargs):
        super(SphereBase, self).__init__(atomgroups, **kwargs)

        self.rmin = rmin
        self._rmax = rmax
        self._bin_width = bin_width

    def _compute_lab_frame_sphere(self):
        """Compute lab limit `rmax`."""
        if self._rmax is None:
            self.rmax = self._universe.dimensions[:3].min() / 2
        else:
            self.rmax = self._rmax

        # Transform into spherical coordinates
        self.pos_sph = self.transform_positions(self._universe.atoms.positions)

    def _prepare(self):
        """Prepare the spherical analysis."""
        self._compute_lab_frame_sphere()

        if self.rmin < 0:
            raise ValueError("Only values for `rmin` larger or equal 0 are "
                             "allowed.")

        if self._rmax is not None and self._rmax <= self.rmin:
            raise ValueError("`rmax` can not be smaller than or equal "
                             "to `rmin`!")

        try:
            if self._bin_width > 0:
                R = self.rmax - self.rmin
                self.n_bins = int(np.ceil(R / self._bin_width))
            else:
                raise ValueError("Binwidth must be a positive number.")
        except TypeError:
            raise ValueError("Binwidth must be a number.")

    def transform_positions(self, positions):
        """Transform positions into spherical coordinates.

        The origin of th coordinate system is at
        :attr:`AnalysisBase.box_center`.

        Parameters
        ----------
        positions : numpy.ndarray
            Cartesian coordinates (x,y,z)

        Returns
        -------
        trans_positions : numpy.ndarray
            Positions in spherical coordinates (r, phi, theta)
        """
        trans_positions = np.zeros(positions.shape)

        # shift origin to box center
        pos_xyz_center = positions - self.box_center

        # r component
        trans_positions[:, 0] = np.linalg.norm(pos_xyz_center, axis=1)

        # phi component
        np.arctan2(pos_xyz_center[:, 1],
                   pos_xyz_center[:, 0],
                   out=trans_positions[:, 1])

        # theta component
        np.arccos(pos_xyz_center[:, 2] / trans_positions[:, 0],
                  out=trans_positions[:, 2])

        return trans_positions

    def _single_frame(self):
        """Single frame for the sphercial analysis."""
        self._compute_lab_frame_sphere()
        self._obs.R = self.rmax - self.rmin

        self._obs.bin_edges = np.linspace(
            self.rmin, self.rmax, self.n_bins + 1, endpoint=True)

        self._obs.bin_width = self._obs.R / self.n_bins
        self._obs.bin_pos = self._obs.bin_edges[1:] - self._obs.bin_width / 2
        self._obs.bin_area = 4 * np.pi * self._obs.bin_pos**2
        self._obs.bin_volume = 4 * np.pi * np.diff(self._obs.bin_edges**3) / 3

    def _conclude(self):
        """Results calculations for the sphercial analysis."""
        super(SphereBase, self)._conclude()
        self.results.bin_pos = self.means.bin_pos


@render_docs
class ProfileSphereBase(SphereBase):
    """Base class for computing radial profiles in a spherical geometry.

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
    ${PROFILE_SPHERE_CLASS_PARAMETERS}
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
        super(ProfileSphereBase, self).__init__(atomgroups=atomgroups,
                                                multi_group=True,
                                                **kwargs)
        if f_kwargs is None:
            f_kwargs = {}

        self.function = lambda ag: function(ag, grouping, **f_kwargs)
        self.normalization = normalization.lower()
        self.grouping = grouping.lower()
        self.binmethod = binmethod.lower()
        self.output = output

    def _prepare(self):
        super(ProfileSphereBase, self)._prepare()

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

        # Arrays for accumulation
        self._obs.profile = np.zeros((self.n_bins, self.n_atomgroups))

        if self.normalization == 'number':
            self.tot_bincount = np.zeros((self.n_bins, self.n_atomgroups))

    def _single_frame(self):
        super(ProfileSphereBase, self)._single_frame()

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

            positions = self.transform_positions(positions)[:, 0]
            weights = self.function(selection)

            profile, _ = np.histogram(positions,
                                      bins=self.n_bins,
                                      range=(self.rmin, self.rmax),
                                      weights=weights)

            if self.normalization == 'number':
                # Use the 2D histogram function to perform the selection in
                # the z dimension.
                bincount, _ = np.histogram(positions,
                                           bins=self.n_bins,
                                           range=(self.rmin, self.rmax))

                self.tot_bincount[:, index] += bincount

                # If a bin does not contain any particles we divide by 0.
                with np.errstate(invalid='ignore'):
                    profile /= bincount
                profile = np.nan_to_num(profile)
            elif self.normalization == "volume":
                profile /= self._obs.bin_volume

            self._obs.profile[:, index] = profile

    def _conclude(self):
        super(ProfileSphereBase, self)._conclude()

        self.results.profile_mean = self.means.profile
        self.results.profile_err = self.sems.profile

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
                     (self.results.bin_pos[:, np.newaxis],
                      self.results.profile_mean,
                      self.results.profile_err)),
                     columns=columns)
