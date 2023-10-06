#!/usr/bin/env python
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
#
# Copyright (c) 2023 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the GNU Public Licence, v3 or any higher version
# SPDX-License-Identifier: GPL-3.0-or-later
r"""Module for computing 2D radial distribution functions."""

import logging
from typing import Optional

import MDAnalysis as mda
import numpy as np
from MDAnalysis.lib.distances import capped_distance

from ..core import PlanarBase
from ..lib.util import get_center, get_compound, render_docs


logger = logging.getLogger(__name__)


@render_docs
class RDFPlanar(PlanarBase):
    r"""Slab-wise planar 2D radial distribution functions.

    The radial distribution function :math:`g_\mathrm{2D}(r)` describes the spatial
    correlation between atoms in :math:`g1` and atoms in :math:`g2`. The 2D RDF can be
    used in systems that are inhomogeneous along one axis, and homogeneous in a plane.
    It gives the average number density of :math:`g2` as a function of lateral distance
    from a centered :math:`g1` atom. In fully homogeneous systems and in the limit of
    small 'dzheight' :math:`\Delta z`, it is the same as the well known three
    dimensional RDF.

    .. math::

     g_\mathrm{2D}(r) =
     \frac{1}{N_{g1}2 \Delta z} \cdot \sum_{i}^{N_{g1}} \sum_{j}^{N_{g2}}
     \delta(r - r_{ij}) \cdot \left( \theta \left(|z_{ij}| + {\Delta z}
     \right) - \theta \left( |z_{ij}| - {\Delta z} \right) \right) .

    As the density to normalise the RDF with is unknown, the output is in the dimension
    of number/volume in 1/Å^3.

    Functionally, RDFPlanar bins all pairwise :math:`g1`-:math:`g2` distances, where the
    z distance is smaller than `dzheight` in a histogram.

    Parameters
    ----------
    g1 : MDAnalysis.core.groups.AtomGroup
        First AtomGroup.
    g2 : MDAnalysis.core.groups.AtomGroup
        Second AtomGroup.
    rdf_bin_width : float
        Binwidth of bins in the histogram of the RDF (Å).
    dzheight : float
        dz height of a RDF slab (Å).
    dmin : float
        the minimum pairwise distance between 'g1' and 'g2' (Å).
    dmax : float
        the maximum pairwise distance between 'g1' and 'g2' (Å).
    bin_method : {``"com"``, ``"cog"``, ``"coc"``}
        Method for position binning; possible options are
        center of mass (``"com"``), center of geometry (``"cog"``) or
        center of charge (``"coc"``).
    output : str
        Output filename
    ${PLANAR_CLASS_PARAMETERS}

    Attributes
    ----------
    ${PLANAR_CLASS_ATTRIBUTES}
    results.bins: numpy.ndarray
        distances to which the RDF is calculated with shape (rdf_nbins) (Å)
    results.rdf: np.ndrray
        RDF with shape (rdf_nbins, n_bins) (1/Å^3)
    """

    def __init__(
        self,
        g1: mda.AtomGroup,
        g2: Optional[mda.AtomGroup] = None,
        rdf_bin_width: float = 0.3,
        dzheight: float = 0.1,
        dmin: float = 0.0,
        dmax: Optional[float] = None,
        bin_method: str = "com",
        output: str = "rdf.dat",
        # Planar base arguments
        refgroup: Optional[mda.AtomGroup] = None,
        unwrap: bool = False,
        concfreq: int = 0,
        jitter: float = 0.0,
        dim: int = 2,
        zmin: Optional[float] = None,
        zmax: Optional[float] = None,
        bin_width: float = 1,
    ):
        self._locals = locals()
        self.comp_1 = get_compound(g1)
        super().__init__(
            atomgroups=g1,
            refgroup=refgroup,
            unwrap=unwrap,
            concfreq=concfreq,
            jitter=jitter,
            dim=dim,
            zmin=zmin,
            zmax=zmax,
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
        self.dzheight = dzheight
        self.output = output
        self.bin_method = bin_method.lower()

        self.comp_2 = get_compound(self.g2)

    def _prepare(self):
        super()._prepare()
        logger.info("Compute radial distribution function.")

        half_of_box_size = min(self.box_center)
        if self.dmax is None:
            self.dmax = min(self.box_center)
            logger.info(
                "Setting maximum range of RDF to half the box size ({self.range[1]} Å)."
            )
        elif self.dmax > min(self.box_center):
            raise ValueError(
                "Range of RDF exceeds half of the box size. Set to smaller than "
                f"{half_of_box_size} Å."
            )

        try:
            if self.rdf_bin_width > 0:
                self.rdf_nbins = int(
                    np.ceil((self.dmax - self.dmin) / self.rdf_bin_width)
                )
            else:
                raise ValueError("RDF bin_width must be a positive number.")
        except TypeError:
            raise ValueError("RDF bin_width must be a number.")

        if self.bin_method not in ["cog", "com", "coc"]:
            raise ValueError(
                f"{self.bin_method} is an unknown binning method. Use `cog`, `com` or "
                "`coc`."
            )

        logger.info(f"Using {self.rdf_nbins} rdf bins.")

        # Empty histogram self.count to store the RDF.
        self.edges = np.histogram(
            [-1], bins=self.rdf_nbins, range=(self.dmin, self.dmax)
        )[1]
        self.results.bins = 0.5 * (self.edges[:-1] + self.edges[1:])

        # Set the max range to filter the search radius.
        self._maxrange = self.dmax

    def _single_frame(self):
        super()._single_frame()
        self._obs.n_g1 = np.zeros((self.n_bins, 1))
        self._obs.count = np.zeros((self.n_bins, self.rdf_nbins))

        bin_width = (self.zmax - self.zmin) / self.n_bins

        g1_bin_positions = get_center(
            atomgroup=self.g1, bin_method=self.bin_method, compound=self.comp_1
        )
        g2_bin_positions = get_center(
            atomgroup=self.g2, bin_method=self.bin_method, compound=self.comp_2
        )

        # Calculate planar rdf per bin by averaging over all atoms in one bin.
        for z_bin in range(0, self.n_bins):
            # Set zmin and zmax of the bin.
            z_min = self.zmin + bin_width * z_bin
            z_max = self.zmin + bin_width * (z_bin + 1)

            # Get all atoms in a bin.
            g1_in_zbin_positions = g1_bin_positions[
                np.logical_and(
                    g1_bin_positions[:, self.dim] >= z_min,
                    g1_bin_positions[:, self.dim] < z_max,
                )
            ]

            g2_in_zbin_positions = g2_bin_positions[
                np.logical_and(
                    g2_bin_positions[:, self.dim] >= z_min - self.dzheight,
                    g2_bin_positions[:, self.dim] < z_max + self.dzheight,
                )
            ]

            n_g1 = len(g1_in_zbin_positions)
            n_g2 = len(g2_in_zbin_positions)
            self._obs.n_g1[z_bin] = n_g1

            # Extract z coordinate.
            z_g1 = np.copy(g1_in_zbin_positions)
            z_g2 = np.copy(g2_in_zbin_positions)
            # Set other coordinates to 0.
            z_g1[:, self.odims] = 0
            z_g2[:, self.odims] = 0

            # Automatically filter only those pairs with delta z < dz.
            z_pairs, _ = capped_distance(
                z_g1, z_g2, self.dzheight, box=self._universe.dimensions
            )

            # Calculate pairwise distances between g1 and g2.
            pairs, xy_distances = capped_distance(
                g1_in_zbin_positions,
                g2_in_zbin_positions,
                self._maxrange,
                box=self._universe.dimensions,
            )

            # Map pairs (i, j) to a number i+N*j (so we can use np.isin).
            z_pairs_encode = z_pairs[:, 0] + n_g2 * z_pairs[:, 1]
            pairs_encode = pairs[:, 0] + n_g2 * pairs[:, 1]

            mask_in_dz = np.isin(pairs_encode, z_pairs_encode)
            mask_different_atoms = np.where(xy_distances > 0, True, False)

            relevant_xy_distances = xy_distances[mask_in_dz * mask_different_atoms]
            # Histogram the pairwise distances.
            self._obs.count[z_bin] = np.histogram(
                relevant_xy_distances, bins=self.rdf_nbins, range=(self.dmin, self.dmax)
            )[0]

    def _conclude(self):
        super()._conclude()

        # Normalise rdf using the volumes of a ring with height dz.
        ring_volumes = (
            np.pi * (self.edges[1:] ** 2 - self.edges[:-1] ** 2) * self.dzheight
        )
        ring_volumes = np.expand_dims(ring_volumes, axis=0)
        self.results.bins = self.results.bins
        self.results.rdf = self.means.count / self.means.n_g1 / ring_volumes / 2
        self.results.rdf = np.nan_to_num(self.results.rdf.T, nan=0)

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
