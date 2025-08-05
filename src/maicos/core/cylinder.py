#!/usr/bin/env python
#
# Copyright (c) 2025 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the GNU Public Licence, v3 or any higher version
# SPDX-License-Identifier: GPL-3.0-or-later
"""Base class for cylindrical analysis."""

import logging
from collections.abc import Callable

import MDAnalysis as mda
import numpy as np

from ..lib.math import transform_cylinder
from ..lib.util import render_docs
from .base import ProfileBase
from .planar import PlanarBase


@render_docs
class CylinderBase(PlanarBase):
    r"""Analysis class providing options and attributes for a cylinder system.

    Provide the results attribute `r`.

    Parameters
    ----------
    ${ATOMGROUP_PARAMETER}
    ${CYLINDER_CLASS_PARAMETERS}
    ${WRAP_COMPOUND_PARAMETER}

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
        atomgroup: mda.AtomGroup,
        unwrap: bool,
        pack: bool,
        refgroup: mda.AtomGroup | None,
        jitter: float,
        concfreq: int,
        dim: int,
        zmin: None | float,
        zmax: None | float,
        bin_width: float,
        rmin: float,
        rmax: None | float,
        wrap_compound: str,
    ):
        super().__init__(
            atomgroup=atomgroup,
            unwrap=unwrap,
            pack=pack,
            refgroup=refgroup,
            jitter=jitter,
            concfreq=concfreq,
            wrap_compound=wrap_compound,
            dim=dim,
            zmin=zmin,
            zmax=zmax,
            bin_width=bin_width,
        )

        self.rmin = rmin
        self._rmax = rmax

    def _compute_lab_frame_cylinder(self):
        """Compute lab limit `rmax`."""
        box_half = self.box_lengths[self.odims].min() / 2
        if self._rmax is None:
            self.rmax = box_half
        elif self._rmax <= box_half:
            self.rmax = self._rmax
        else:
            logging.warning(
                f"`rmax` is bigger than half the smallest box vector ({box_half:.2f}) "
                "in the radial direction. This will lead to artifacts at the edges."
            )
            self.rmax = self._rmax
        # enforce calculations in double precision
        self.rmin = np.float64(self.rmin)
        self.rmax = np.float64(self.rmax)
        # Transform into cylinder coordinates
        self.pos_cyl = transform_cylinder(
            self._universe.atoms.positions, origin=self.box_center, dim=self.dim
        )

    def _prepare(self):
        """Prepare the cylinder analysis."""
        super()._prepare()

        self._compute_lab_frame_cylinder()

        if self.rmin < 0:
            raise ValueError("Only values for `rmin` larger or equal 0 are allowed.")

        if self._rmax is not None and self._rmax <= self.rmin:
            raise ValueError("`rmax` can not be smaller than or equal to `rmin`!")

        try:
            if self._bin_width > 0:
                R = self.rmax - self.rmin
                self.n_bins = int(np.ceil(R / self._bin_width))
            else:
                raise ValueError("Binwidth must be a positive number.")
        except TypeError as err:
            raise ValueError("Binwidth must be a number.") from err

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

    ${CORRELATION_INFO_RADIAL}

    Parameters
    ----------
    ${PROFILE_CYLINDER_CLASS_PARAMETERS}
    ${PROFILE_CLASS_PARAMETERS_PRIVATE}

    Attributes
    ----------
    ${PROFILE_CYLINDER_CLASS_ATTRIBUTES}

    """

    def __init__(
        self,
        atomgroup: mda.AtomGroup,
        unwrap: bool,
        pack: bool,
        refgroup: mda.AtomGroup | None,
        jitter: float,
        concfreq: int,
        dim: int,
        zmin: None | float,
        zmax: None | float,
        bin_width: float,
        rmin: float,
        rmax: None | float,
        grouping: str,
        bin_method: str,
        output: str,
        weighting_function: Callable,
        weighting_function_kwargs: None | dict,
        normalization: str,
    ):
        CylinderBase.__init__(
            self,
            atomgroup=atomgroup,
            unwrap=unwrap,
            pack=pack,
            refgroup=refgroup,
            jitter=jitter,
            concfreq=concfreq,
            dim=dim,
            zmin=zmin,
            zmax=zmax,
            bin_width=bin_width,
            rmin=rmin,
            rmax=rmax,
            wrap_compound=grouping,  # same as grouping to avoid broken compounds
        )
        # `AnalysisBase` performs conversions on `atomgroup`. Take converted
        # `atomgroup` and not the user provided ones.
        ProfileBase.__init__(
            self,
            atomgroup=self.atomgroup,
            grouping=grouping,
            bin_method=bin_method,
            output=output,
            weighting_function=weighting_function,
            weighting_function_kwargs=weighting_function_kwargs,
            normalization=normalization,
        )

    def _prepare(self):
        CylinderBase._prepare(self)
        ProfileBase._prepare(self)

        logging.info(
            f"""Profile along the radial axis in a cylindrical coordinate system,"""
            f""" with the {"xyz"[self.dim]}-axis as cylindrical axis."""
        )

    def _compute_histogram(
        self, positions: np.ndarray, weights: np.ndarray | None = None
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
        return hist[:, 0]

    def _single_frame(self) -> float:
        CylinderBase._single_frame(self)
        ProfileBase._single_frame(self)

        # Take the center bin of the zeroth group for correlation analysis.
        return self._obs.profile[0]

    def _conclude(self):
        CylinderBase._conclude(self)
        ProfileBase._conclude(self)
