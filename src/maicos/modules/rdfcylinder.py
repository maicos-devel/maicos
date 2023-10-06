#!/usr/bin/env python3
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
#
# Copyright (c) 2022 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the GNU Public Licence, v3 or any higher version
# SPDX-License-Identifier: GPL-3.0-or-later
r"""Module for computing 1D radial distribution functions."""

import logging
from typing import Optional

import MDAnalysis as mda
import numpy as np
from MDAnalysis.lib.distances import capped_distance
from scipy.interpolate import interp1d

from ..core import CylinderBase
from ..lib.math import sa_cylider_intersect_circle, transform_cylinder
from ..lib.util import get_center, get_compound, render_docs


logger = logging.getLogger(__name__)


@render_docs
class RDFCylinder(CylinderBase):
    r"""Cylindrical shell-wise 1D radial distribution functions.

    The radial distribution function :math:`g_\mathrm{1d}(r)` describes the
    spatial correlation between atoms in :math:`g1` and atoms in :math:`g2`.
    The 2D RDF can be used in systems that are inhomogeneous along one axis, and
    homogeneous in a plane. It gives the average number density of :math:`g2` as
    a function of lateral distance from a centered :math:`g1` atom. In fully
    homogeneous systems and in the limit of small 'drwidth' :math:`\Delta R`,
    it is the same as the well known three dimensional RDF.

    .. math::

         g_\mathrm{1D}(r) =
         \frac{1}{N_{g1}2 \Delta R} \cdot \sum_{i}^{N_{g1}} \sum_{j}^{N_{g2}}
         \delta(r - r_{ij}) \cdot \left( \theta \left(|R_{ij}| + {\Delta R}
         \right) - \theta \left( |R_{ij}| - {\Delta R} \right) \right) .

    As the density to normalise the RDF with is unknown, the output
    is in the dimension of number/volume in 1/Å^3.

    Functionally, RDFCylinder bins all pairwise :math:`g1`-:math:`g2` distances,
    where the radial distance is smaller than `drwidth` in a histogram.

    Parameters
    ----------
    g1 : MDAnalysis.core.groups.AtomGroup
        First AtomGroup.
    g2 : MDAnalysis.core.groups.AtomGroup
        Second AtomGroup.
    rdf_bin_width : int
        Binwidth of bins in the histogram of the RDF (Å).
    drwidth : float
        radial width of a RDF cylindrical shell (Å).
    dmin: float
        the minimum pairwise distance between 'g1' and 'g2' (Å).
    dmax : float
        the minimum pairwise distance between 'g1' and 'g2' (Å).
    bin_method : {``"com"``, ``"cog"``, ``"coc"``}
        Method for position binning; possible options are
        center of mass (``"com"``), center of geometry (``"cog"``) or
        center of charge (``"coc"``).
    output : str
        Output filename
    ${CYLINDER_CLASS_PARAMETERS}

    Attributes
    ----------
    ${CYLINDER_CLASS_ATTRIBUTES}
    results.bins: numpy.ndarray
        radial distances to which the RDF is calculated with shape (`rdf_nbins`) (Å)
    results.rdf: numpy.ndarray
        RDF with shape (`rdf_nbins`, `n_bins`) (:math:`\text{Å}^{-3}`)
    """

    def __init__(
        self,
        g1: mda.AtomGroup,
        g2: Optional[mda.AtomGroup] = None,
        rdf_bin_width: float = 0.3,
        drwidth: float = 0.1,
        dmin: float = 0,
        dmax: Optional[float] = None,
        bin_method: str = "com",
        output: str = "rdf.dat",
        refgroup: Optional[mda.AtomGroup] = None,
        unwrap: bool = False,
        concfreq: int = 0,
        jitter: float = 0.0,
        dim: int = 2,
        zmin: Optional[float] = None,
        zmax: Optional[float] = None,
        rmin: float = 0,
        rmax: Optional[float] = None,
        bin_width: float = 1,
        origin: Optional[np.ndarray] = None,
    ):
        self.comp_1 = get_compound(g1)
        super(RDFCylinder, self).__init__(
            atomgroups=g1,
            refgroup=refgroup,
            unwrap=unwrap,
            concfreq=concfreq,
            jitter=jitter,
            dim=dim,
            zmin=zmin,
            zmax=zmax,
            rmin=rmin,
            rmax=rmax,
            bin_width=bin_width,
            wrap_compound=self.comp_1,
        )

        self.g1 = g1
        if g2 is None:
            self.g2 = g1
        else:
            self.g2 = g2
        self.dmin = dmin
        self.dmax = dmax
        self.rdf_bin_width = rdf_bin_width
        self.drwidth = drwidth
        self.bin_width = bin_width
        self.output = output
        self.bin_method = bin_method.lower()
        self.origin = origin

        self.comp_2 = get_compound(self.g2)

        if self.dmax is None:
            self.dmax = self.box_center[self.dim]
            logger.info(
                "Setting maximum range of RDF to half the box size" f"({self.dmax} Å)."
            )
        elif self.dmax > self.box_center[self.dim]:
            raise ValueError(
                "Range of RDF exceeds half of the box size. "
                f"Set to smaller than {self.box_center[self.dim]} Å."
            )

        if self.origin is None:
            self.origin = self.box_center

        if self.rdf_bin_width < 0:
            raise ValueError("RDF bin_width must be a positive number.")

        self.rdf_nbins = int(np.ceil((self.dmax - self.dmin) / self.rdf_bin_width))

        self.rdf_bin_width = (self.dmax - self.dmin) / self.rdf_nbins

        # TODO: No longer needed?
        if self.bin_method not in ["cog", "com", "coc"]:
            raise ValueError(
                f"{self.bin_method!r} is an unknown binning "
                "method. Use 'cog', 'com' or 'coc'."
            )

        self.edges = np.histogram(
            [-1], bins=self.rdf_nbins, range=(self.dmin, self.dmax)
        )[1]
        self.results.bins = 0.5 * (self.edges[:-1] + self.edges[1:])

    def _prepare(self):
        super(RDFCylinder, self)._prepare()
        logger.info("Compute radial distribution function.")
        logger.info(f"Using {self.rdf_nbins} rdf bins.")

        # Set the max range to filter the search radius.
        self._maxrange = self.dmax

        # Initialize Volume*N_g1 integration bins for normalisation.
        self.integration_nbins = int(np.ceil((self.rmax - self.rmin) / self.drwidth))
        self.integration_bins = np.linspace(
            self.rmin,
            self.rmax,
            self.integration_nbins + 1,
            endpoint=True,
        )
        self.integration_bin_width = (self.rmax - self.rmin) / self.integration_nbins

    def _single_frame(self):
        super(RDFCylinder, self)._single_frame()
        self._obs.count = np.zeros((self.n_bins, self.rdf_nbins))

        g1_bin_positions = get_center(
            atomgroup=self.g1, bin_method=self.bin_method, compound=self.comp_1
        )
        g2_bin_positions = get_center(
            atomgroup=self.g2, bin_method=self.bin_method, compound=self.comp_2
        )

        # concatenate the bin positions with radial dimension of the
        # cylinderical coordinate system and zero as rest two dimensions so
        # that we can use MDA.capped_distance to filter positions with drwidth
        g1_bin_positions_w_radial = np.zeros((g1_bin_positions.shape[0], 6))
        g1_bin_positions_w_radial[:, :3] = g1_bin_positions
        g1_bin_positions_w_radial[:, 3] = transform_cylinder(
            g1_bin_positions, origin=self.origin, dim=self.dim
        )[:, 0]

        g2_bin_positions_w_radial = np.zeros((g2_bin_positions.shape[0], 6))
        g2_bin_positions_w_radial[:, :3] = g2_bin_positions
        g2_bin_positions_w_radial[:, 3] = transform_cylinder(
            g2_bin_positions, origin=self.origin, dim=self.dim
        )[:, 0]

        # Histogram the radial dimension of the cylinderical coordinate system
        self._obs.integration_bin_count = np.histogram(
            g1_bin_positions_w_radial[:, 3],
            bins=self.integration_nbins,
            range=(self.rmin, self.rmax),
        )[0]

        # Create a new box such that minimum image convention returns the
        # non-periodic coordinate along odims
        box_new = np.copy(self._universe.dimensions)
        box_new[self.odims] += self._maxrange

        # Calculate planar rdf per bin by averaging over all atoms in one bin.
        for r_bin in range(0, self.n_bins):
            # Get all atoms in a bin.
            g1_in_rbin_positions = g1_bin_positions_w_radial[
                np.logical_and(
                    g1_bin_positions_w_radial[:, 3] >= self._obs.bin_edges[r_bin],
                    g1_bin_positions_w_radial[:, 3] < self._obs.bin_edges[r_bin + 1],
                )
            ]

            g2_in_rbin_positions = g2_bin_positions_w_radial[
                np.logical_and(
                    g2_bin_positions_w_radial[:, 3]
                    >= self._obs.bin_edges[r_bin] - self.drwidth,
                    g2_bin_positions_w_radial[:, 3]
                    < self._obs.bin_edges[r_bin + 1] + self.drwidth,
                )
            ]

            n_g2 = len(g2_in_rbin_positions)

            # Automatically filter only those pairs with delta r < dr.
            r_pairs = capped_distance(
                g1_in_rbin_positions[:, 3:],
                g2_in_rbin_positions[:, 3:],
                self.drwidth,
                box=None,
                return_distances=False,
            )

            # Calculate pairwise distances between g1 and g2.
            pairs, xy_distances = capped_distance(
                g1_in_rbin_positions[:, :3],
                g2_in_rbin_positions[:, :3],
                self._maxrange,
                box=box_new,
            )

            # Map pairs (i, j) to a number i+N*j (so we can use np.isin).
            r_pairs_encode = r_pairs[:, 0] + n_g2 * r_pairs[:, 1]
            pairs_encode = pairs[:, 0] + n_g2 * pairs[:, 1]

            mask_in_dr = np.isin(pairs_encode, r_pairs_encode)
            mask_different_atoms = np.where(xy_distances > 0, True, False)

            relevant_xy_distances = xy_distances[mask_in_dr * mask_different_atoms]
            # Histogram the pairwise distances.
            self._obs.count[r_bin] = np.histogram(
                relevant_xy_distances, bins=self.rdf_nbins, range=(self.dmin, self.dmax)
            )[0]

    def _conclude(self):
        super()._conclude()

        # Calculate the Volume*N_g1 normalization factor for each bin.
        self.norm = np.zeros((self.n_bins, self.rdf_nbins))
        AN_bin_last = np.zeros(self.n_bins)
        for i in range(self.rdf_nbins):
            AN_integral = np.cumsum(
                sa_cylider_intersect_circle(
                    self.edges[i + 1], self.integration_bins[1:]
                )
                * self.means.integration_bin_count
            )
            AN_integral = np.append([0], AN_integral)
            AN_in_interp = interp1d(
                self.integration_bins, AN_integral, kind=1, assume_sorted=True
            )
            AN_bin = AN_in_interp(self.means.bin_edges[1:]) - AN_in_interp(
                self.means.bin_edges[:-1]
            )
            self.norm[:, i] = (AN_bin - AN_bin_last) * self.drwidth
            AN_bin_last = AN_bin

        # Normalise rdf using the normalisation factor.
        with np.errstate(invalid="ignore", divide="ignore"):
            self.results.rdf = self.means.count / self.norm / 2
        self.results.rdf = np.nan_to_num(self.results.rdf.T, nan=0, posinf=0, neginf=0)

    def save(self):
        """Save results."""
        columns = ["r [Å]"]
        for z in self.results.bin_pos:
            columns.append(f"rdf at {z:.2f} Å [Å^-3]")

        self.savetxt(
            self.output,
            np.hstack([self.results.bins[:, np.newaxis], self.results.rdf]),
            columns=columns,
        )
