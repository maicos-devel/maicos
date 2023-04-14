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
from .base import ProfileBase
from .planar import AnalysisBase


logger = logging.getLogger(__name__)


@render_docs
class SphereBase(AnalysisBase):
    r"""Analysis class providing options and attributes for a spherical system.

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
class ProfileSphereBase(SphereBase, ProfileBase):
    """Base class for computing radial profiles in a spherical geometry.

    ${CORRELATION_INFO_SPHERE}

    Parameters
    ----------
    ${PROFILE_CLASS_PARAMETERS_PRIVATE}
    ${PROFILE_SPHERE_CLASS_PARAMETERS}

    Attributes
    ----------
    ${PROFILE_CYLINDER_CLASS_ATTRIBUTES}
    """

    def __init__(self,
                 weighting_function,
                 normalization,
                 atomgroups,
                 grouping,
                 bin_method,
                 output,
                 f_kwargs=None,
                 **kwargs):
        SphereBase.__init__(self,
                            atomgroups=atomgroups,
                            multi_group=True,
                            wrap_compound=grouping,
                            **kwargs)
        # `AnalysisBase` performs conversions on `atomgroups`.
        # Take converted `atomgroups` and not the user provided ones.
        ProfileBase.__init__(self,
                             atomgroups=self.atomgroups,
                             weighting_function=weighting_function,
                             normalization=normalization,
                             grouping=grouping,
                             bin_method=bin_method,
                             output=output,
                             f_kwargs=f_kwargs)

    def _prepare(self):
        SphereBase._prepare(self)
        ProfileBase._prepare(self)

        logger.info(f"Computing {self.grouping} radial profile.")

    def _compute_histogram(self, positions, weights):
        positions = self.transform_positions(positions)[:, 0]
        hist, _ = np.histogram(positions,
                               bins=self.n_bins,
                               range=(self.rmin, self.rmax),
                               weights=weights)

        return hist

    def _single_frame(self):
        SphereBase._single_frame(self)
        ProfileBase._single_frame(self)

        # Take the center bin of the zeroth group for correlation analysis.
        return self._obs.profile[0, 0]

    def _conclude(self):
        SphereBase._conclude(self)
        ProfileBase._conclude(self)
