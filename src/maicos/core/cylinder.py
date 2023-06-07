#!/usr/bin/env python
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
#
# Copyright (c) 2023 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the GNU Public Licence, v3 or any higher version
# SPDX-License-Identifier: GPL-3.0-or-later
"""Base class for cylindrical analysis."""
import logging
from typing import Callable, Dict, List, Optional, Union

import MDAnalysis as mda
import numpy as np

from ..lib.math import transform_cylinder
from ..lib.util import render_docs
from .base import ProfileBase
from .planar import PlanarBase


logger = logging.getLogger(__name__)


@render_docs
class CylinderBase(PlanarBase):
    r"""Analysis class providing options and attributes for a cylinder system.

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
    _obs.R : float
        Average length (in Å) along the radial dimension in the current frame.
    _obs.bin_pos : numpy.ndarray, (n_bins)
        Central bin position of each bin (in Å) in the current frame.
    _obs.bin_width : float
         Bin width (in Å) in the current frame
    _obs.bin_edges : numpy.ndarray, (n_bins + 1)
        Edges of the bins (in Å) in the current frame.
    _obs.bin_area : numpy.ndarray, (n_bins)
        Area of the annulus pf the each bin in the current frame. Calculated via
        :math:`\pi \left( r_{i+1}^2 - r_i^2 \right)` where `i` is the index of the bin.
    _obs.bin_volume : numpy.ndarray, (n_bins)
        Volume of an hollow cylinder of each bin (in Å^3) in the current frame.
        Calculated via :math:`\pi L \left( r_{i+1}^2 - r_i^2 \right)` where `i` is the
        index of the bin.
    """

    def __init__(
        self,
        atomgroups: Union[mda.AtomGroup, List[mda.AtomGroup]],
        rmin: float,
        rmax: Union[None, float],
        **kwargs,
    ):
        super().__init__(atomgroups, **kwargs)

        self.rmin = rmin
        self._rmax = rmax

    def _compute_lab_frame_cylinder(self):
        """Compute lab limit `rmax`."""
        if self._rmax is None:
            self.rmax = self._universe.dimensions[self.odims].min() / 2
        else:
            self.rmax = self._rmax

        # Transform into cylinder coordinates
        self.pos_cyl = transform_cylinder(
            self._universe.atoms.positions, origin=self.box_center, dim=self.dim
        )

    def _prepare(self):
        """Prepare the cylinder analysis."""
        super()._prepare()

        self._compute_lab_frame_cylinder()

        if self.rmin < 0:
            raise ValueError("Only values for `rmin` larger or equal 0 are " "allowed.")

        if self._rmax is not None and self._rmax <= self.rmin:
            raise ValueError("`rmax` can not be smaller than or equal " "to `rmin`!")

        try:
            if self._bin_width > 0:
                R = self.rmax - self.rmin
                self.n_bins = int(np.ceil(R / self._bin_width))
            else:
                raise ValueError("Binwidth must be a positive number.")
        except TypeError:
            raise ValueError("Binwidth must be a number.")

    def _single_frame(self):
        """Single frame for the cylinder analysis."""
        super()._single_frame()
        self._compute_lab_frame_cylinder()
        self._obs.R = self.rmax - self.rmin

        self._obs.bin_edges = np.linspace(
            self.rmin, self.rmax, self.n_bins + 1, endpoint=True
        )

        self._obs.bin_width = self._obs.R / self.n_bins
        self._obs.bin_pos = self._obs.bin_edges[1:] - self._obs.bin_width / 2
        self._obs.bin_area = np.pi * np.diff(self._obs.bin_edges**2)
        self._obs.bin_volume = self._obs.bin_area * self._obs.L

    def _conclude(self):
        """Results calculations for the cylinder analysis."""
        super()._conclude()
        self.results.bin_pos = self.means.bin_pos


@render_docs
class ProfileCylinderBase(CylinderBase, ProfileBase):
    """Base class for computing radial profiles in a cylindrical geometry.

    ${CORRELATION_INFO_CYLINDER}

    Parameters
    ----------
    ${PROFILE_CLASS_PARAMETERS_PRIVATE}
    ${PROFILE_CYLINDER_CLASS_PARAMETERS}

    Attributes
    ----------
    ${PROFILE_CYLINDER_CLASS_ATTRIBUTES}
    """

    def __init__(
        self,
        weighting_function: Callable,
        normalization: str,
        atomgroups: Union[mda.AtomGroup, List[mda.AtomGroup]],
        grouping: str,
        bin_method: str,
        output: str,
        f_kwargs: Optional[Dict] = None,
        **kwargs,
    ):
        CylinderBase.__init__(
            self,
            atomgroups=atomgroups,
            multi_group=True,
            wrap_compound=grouping,
            **kwargs,
        )
        # `AnalysisBase` performs conversions on `atomgroups`. Take converted
        # `atomgroups` and not the user provided ones.
        ProfileBase.__init__(
            self,
            atomgroups=self.atomgroups,
            weighting_function=weighting_function,
            normalization=normalization,
            grouping=grouping,
            bin_method=bin_method,
            output=output,
            f_kwargs=f_kwargs,
        )

    def _prepare(self):
        CylinderBase._prepare(self)
        ProfileBase._prepare(self)

        logger.info(
            f"Computing {self.grouping} radial profile along "
            f"{'XYZ'[self.dim]}-axes."
        )

    def _compute_histogram(
        self, positions: np.ndarray, weights: Optional[np.ndarray] = None
    ) -> np.ndarray:
        positions = transform_cylinder(positions, self.box_center, self.dim)
        # Use the 2D histogram function to perform the selection in the z dimension.
        hist, _, _ = np.histogram2d(
            positions[:, 0],
            positions[:, 2],
            bins=(self.n_bins, 1),
            range=((self.rmin, self.rmax), (self.zmin, self.zmax)),
            weights=weights,
        )

        # Reshape into 1D array
        hist = hist[:, 0]

        return hist

    def _single_frame(self) -> float:
        CylinderBase._single_frame(self)
        ProfileBase._single_frame(self)

        # Take the center bin of the zeroth group for correlation analysis.
        return self._obs.profile[0, 0]

    def _conclude(self):
        CylinderBase._conclude(self)
        ProfileBase._conclude(self)
