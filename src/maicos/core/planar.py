#!/usr/bin/env python
#
# Copyright (c) 2025 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the GNU Public Licence, v3 or any higher version
# SPDX-License-Identifier: GPL-3.0-or-later
"""Base class for planar analysis."""

import logging
from collections.abc import Callable

import MDAnalysis as mda
import numpy as np

from ..lib.math import symmetrize
from ..lib.util import render_docs
from .base import AnalysisBase, ProfileBase


@render_docs
class PlanarBase(AnalysisBase):
    r"""Analysis class providing options and attributes for a planar system.

    Parameters
    ----------
    ${ATOMGROUP_PARAMETER}
    ${PLANAR_CLASS_PARAMETERS}
    ${WRAP_COMPOUND_PARAMETER}

    Attributes
    ----------
    ${PLANAR_CLASS_ATTRIBUTES}
    zmin : float
         Minimal coordinate for evaluation (Å) with in the lab frame, where 0
         corresponds to the origin of the cell.
    zmax : float
         Maximal coordinate for evaluation (Å) with in the lab frame, where 0
         corresponds to the origin of the cell.
    _obs.L : float
        Length (in Å) along the chosen dimension in the current frame.
    _obs.bin_pos : numpy.ndarray, (n_bins)
        Central bin positions (in Å) of each bin (in Å) in the current frame.
    _obs.bin_width : float
         Bin width (in Å) in the current frame
    _obs.bin_edges : numpy.ndarray, (n_bins + 1)
        Edges of the bins (in Å) in the current frame.
    _obs.bin_area : numpy.ndarray, (n_bins)
        Area of the rectangle of each bin in the current frame. Calculated via
        :math:`L_x \cdot L_y / N_\mathrm{bins}` where :math:`L_x` and :math:`L_y` are
        the box lengths perpendicular to the dimension of evaluations given by `dim`.
        :math:`N_\mathrm{bins}` is the number of bins.
    _obs.bin_volume : numpy.ndarray, (n_bins)
        Volume of an cuboid of each bin (in Å^3) in the current frame.

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
        )

        if dim not in [0, 1, 2]:
            raise ValueError("Dimension can only be x=0, y=1 or z=2.")
        self.dim = dim

        # These values are requested by the user, but the actual ones are calculated
        # during runtime in the lab frame
        self._zmax = zmax
        self._zmin = zmin
        self._bin_width = bin_width

    @property
    def odims(self) -> np.ndarray:
        """Other dimensions perpendicular to dim i.e. (0,2) if dim = 1."""
        return np.roll(np.arange(3), -self.dim)[1:]

    def _compute_lab_frame_planar(self):
        """Compute lab limits `zmin` and `zmax`."""
        if self._zmin is None:
            self.zmin = 0
        else:
            self.zmin = self.box_center[self.dim] + self._zmin

        if self._zmax is None:
            self.zmax = self.box_lengths[self.dim]
        else:
            self.zmax = self.box_center[self.dim] + self._zmax
        # enforce calculations in double precision
        self.zmin = np.float64(self.zmin)
        self.zmax = np.float64(self.zmax)

    def _prepare(self):
        """Prepare the planar analysis."""
        self._compute_lab_frame_planar()

        # TODO(@hejamu): There are much more wrong combinations of zmin and zmax...
        if (
            self._zmax is not None
            and self._zmin is not None
            and self._zmax <= self._zmin
        ):
            raise ValueError("`zmax` can not be smaller or equal than `zmin`!")

        try:
            if self._bin_width > 0:
                L = self.zmax - self.zmin
                self.n_bins = int(np.ceil(L / self._bin_width))
            else:
                raise ValueError("Binwidth must be a positive number.")
        except TypeError as err:
            raise ValueError("Binwidth must be a number.") from err

    def _single_frame(self):
        """Single frame for the planar analysis."""
        self._compute_lab_frame_planar()
        self._obs.L = self.zmax - self.zmin
        self._obs.box_center = self.box_center
        self._obs.bin_edges = np.linspace(self.zmin, self.zmax, self.n_bins + 1)

        self._obs.bin_width = self._obs.L / self.n_bins
        self._obs.bin_pos = self._obs.bin_edges[1:] - self._obs.bin_width / 2
        # We define `bin_area` and `bin_volume` as array of length `n_bins` even though
        # each element has the same value. With this the array shape is consistent with
        # the cylindrical and spherical classes, where `bin_area` and `bin_volume` is
        # different in each bin.
        self._obs.bin_area = np.ones(self.n_bins) * np.prod(
            self.box_lengths[self.odims]
        )
        self._obs.bin_volume = self._obs.bin_area * self._obs.bin_width

    def _conclude(self):
        """Results calculations for the planar analysis."""
        # Convert coordinates back from lab frame to refgroup frame.
        self.results.bin_pos = self.means.bin_pos - self.means.box_center[self.dim]


@render_docs
class ProfilePlanarBase(PlanarBase, ProfileBase):
    """Base class for computing profiles in a cartesian geometry.

    ${CORRELATION_INFO_RADIAL}

    Parameters
    ----------
    ${PROFILE_PLANAR_CLASS_PARAMETERS}
    ${PROFILE_CLASS_PARAMETERS_PRIVATE}

    Attributes
    ----------
    ${PROFILE_PLANAR_CLASS_ATTRIBUTES}

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
        sym: bool,
        grouping: str,
        bin_method: str,
        output: str,
        weighting_function: Callable,
        weighting_function_kwargs: None | dict,
        normalization: str,
    ):
        PlanarBase.__init__(
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
            wrap_compound=grouping,  # same as grouping to avoid broken compounds
        )
        # `AnalysisBase` performs conversions on `atomgroup`.
        # Take converted `atomgroup` and not the user provided ones.
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

        self.sym = sym

    def _prepare(self):
        PlanarBase._prepare(self)
        ProfileBase._prepare(self)

        if self.sym and self.refgroup is None:
            raise ValueError("For symmetrization the `refgroup` argument is required.")

        logging.info(f"""Profile along {"xyz"[self.dim]}-axis normal to the plane.""")

    def _compute_histogram(
        self, positions: np.ndarray, weights: np.ndarray | None = None
    ) -> np.ndarray:
        positions = positions[:, self.dim]
        hist, bin_edges = np.histogram(
            positions, bins=self.n_bins, range=(self.zmin, self.zmax), weights=weights
        )

        bin_indices = np.digitize(positions, bin_edges[1:-1])

        return hist, bin_indices

    def _single_frame(self) -> float:
        PlanarBase._single_frame(self)
        ProfileBase._single_frame(self)

        # Take the center bin of the 0th group for correlation analysis.
        return self._obs.profile[self.n_bins // 2]

    def _conclude(self):
        PlanarBase._conclude(self)

        if self.sym:
            symmetrize(self.sums.profile, inplace=True)
            symmetrize(self.means.profile, inplace=True)
            symmetrize(self.sems.profile, inplace=True)

            if self.normalization == "number":
                symmetrize(self.sums.bincount, inplace=True)

        # Call conclude after symmetrize since `_concude` sets empty bins to `nan` and
        # this prevents symmetrizing.
        ProfileBase._conclude(self)
