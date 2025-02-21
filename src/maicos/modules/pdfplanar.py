#!/usr/bin/env python
#
# Copyright (c) 2025 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the GNU Public Licence, v3 or any higher version
# SPDX-License-Identifier: GPL-3.0-or-later
r"""Module for computing 2D planar pair distribution functions."""

import logging

import MDAnalysis as mda
import numpy as np
from MDAnalysis.lib.distances import capped_distance

from ..core import PlanarBase
from ..lib.util import get_center, get_compound, render_docs

logger = logging.getLogger(__name__)


@render_docs
class PDFPlanar(PlanarBase):
    r"""Slab-wise planar 2D pair distribution functions.

    The pair distribution function :math:`g_\mathrm{2D}(r)` describes the
    spatial correlation between atoms in :math:`g_1` and atoms in
    :math:`g_2`, which lie in the same plane.
    It gives the average number density of :math:`g_2` atoms as a function of lateral
    distance :math:`r` from a centered :math:`g_1` atom.
    PDFPlanar can be used in systems that are inhomogeneous along one axis,
    and homogeneous in a plane.
    In fully homogeneous systems and in the limit of small 'dzheight'
    :math:`\Delta z`, it is the same as the well known three dimensional PDF.

    The planar PDF is defined by

    .. math::

        g_\mathrm{2D}(r) = \left \langle
        \frac{1}{N_{g1}} \cdot \sum_{i}^{N_{g1}} \sum_{j}^{N_{g2}}
        \frac{1}{2 \pi r} \delta(r - r_{ij}) \delta(z_{ij})
        \right \rangle .

    where the brackets :math:`\langle \cdot \rangle` denote the ensemble
    average. :math:`\delta(r- r_{ij})` counts the :math:`g_2` atoms at distance
    :math:`r` from atom :math:`i`.
    :math:`\delta(z_{ij})` ensures that only atoms, which lie
    in the same plane :math:`z_i = z_j`, are considered for the PDF.

    Discretized for computational purposes the equation reads as

    .. math::

        g_\mathrm{2D}(r) =
        \frac{1}{N_{g1}} \cdot \sum_{i}^{N_{g1}} \frac{\mathrm{count}\; g_2 \;
        \mathrm{in}\; \Delta V_i(r) }{\Delta V_i(r)} .

    where :math:`\Delta V_i(r)` is a ring around atom i, with inner
    radius :math:`r - \frac{\Delta r}{2}`, outer radius
    :math:`r + \frac{\Delta r}{2}` and height :math:`2 \Delta z`.

    As the density to normalise the PDF with is unknown, the output is in
    the dimension of number/volume in 1/Å^3.

    Functionally, PDFPlanar bins all pairwise :math:`g_1`-:math:`g_2` distances,
    where the z distance is smaller than 'dzheight' in a histogram.

    For a more detailed explanation refer to
    :ref:`Explanation: PDF<pdfs-explanation>` and
    :ref:`PDFPlanar Derivation<pdfplanar-derivation>`

    Parameters
    ----------
    ${PDF_PARAMETERS}
    pdf_bin_width : float
        Binwidth of bins in the histogram of the PDF (Å).
    dzheight : float
        dz height of a PDF slab :math:`\Delta z` (Å). :math:`\Delta z` is
        introduced to discretize the delta function :math:`\delta(z_{ij})`.
        It is the maximum :math:`z` distance between atoms which are
        considered to lie in the same plane.
        In the limit of :math:`\Delta z \to 0`, PDFPlanar reaches the
        continous limit. However, if :math:`\Delta z` is too small, there
        are no atoms in ``g2`` to sample.
        We recommend a choice of :math:`\Delta z` that is 1/10th of
        a bond length.
    dmin : float
        Minimum pairwise distance between ``g1`` and ``g2`` (Å).
    dmax : float
        Maximum pairwise distance between ``g1`` and ``g2`` (Å).
    ${BIN_METHOD_PARAMETER}
    ${OUTPUT_PARAMETER}
    ${PLANAR_CLASS_PARAMETERS}

    Attributes
    ----------
    ${PLANAR_CLASS_ATTRIBUTES}
    results.bins: numpy.ndarray
        distances to which the PDF is calculated with shape (pdf_nbins) (Å)
    results.pdf: np.ndrray
        PDF with shape (pdf_nbins, n_bins) (1/Å^3)

    """

    def __init__(
        self,
        g1: mda.AtomGroup,
        g2: mda.AtomGroup | None = None,
        pdf_bin_width: float = 0.3,
        dzheight: float = 0.1,
        dmin: float = 0.0,
        dmax: float | None = None,
        bin_method: str = "com",
        output: str = "pdf.dat",
        unwrap: bool = False,
        pack: bool = True,
        refgroup: mda.AtomGroup | None = None,
        concfreq: int = 0,
        jitter: float = 0.0,
        dim: int = 2,
        zmin: float | None = None,
        zmax: float | None = None,
        bin_width: float = 1,
    ) -> None:
        self._locals = locals()
        self.comp_1 = get_compound(g1)
        super().__init__(
            atomgroup=g1,
            refgroup=refgroup,
            unwrap=unwrap,
            pack=pack,
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
        self.pdf_bin_width = pdf_bin_width
        self.dzheight = dzheight
        self.output = output
        self.bin_method = bin_method.lower()

        self.comp_2 = get_compound(self.g2)

    def _prepare(self) -> None:
        super()._prepare()
        logger.info("Compute pair distribution function.")

        half_of_box_size = min(self.box_center)
        if self.dmax is None:
            self.dmax = min(self.box_center)
            logger.info(
                "Setting maximum range of PDF to half the box size ({self.range[1]} Å)."
            )
        elif self.dmax > min(self.box_center):
            raise ValueError(
                "Range of PDF exceeds half of the box size. Set to smaller than "
                f"{half_of_box_size} Å."
            )

        try:
            if self.pdf_bin_width > 0:
                self.pdf_nbins = int(
                    np.ceil((self.dmax - self.dmin) / self.pdf_bin_width)
                )
            else:
                raise ValueError("PDF bin_width must be a positive number.")
        except TypeError as err:
            raise ValueError("PDF bin_width must be a number.") from err

        if self.bin_method not in ["cog", "com", "coc"]:
            raise ValueError(
                f"{self.bin_method} is an unknown binning method. Use `cog`, `com` or "
                "`coc`."
            )

        logger.info(f"Using {self.pdf_nbins} pdf bins.")

        # Empty histogram self.count to store the PDF.
        self.edges = np.histogram(
            [-1], bins=self.pdf_nbins, range=(self.dmin, self.dmax)
        )[1]
        self.results.bins = 0.5 * (self.edges[:-1] + self.edges[1:])

        # Set the max range to filter the search radius.
        self._maxrange = self.dmax

    def _single_frame(self) -> None:
        super()._single_frame()
        self._obs.n_g1 = np.zeros((self.n_bins, 1))
        self._obs.count = np.zeros((self.n_bins, self.pdf_nbins))

        bin_width = (self.zmax - self.zmin) / self.n_bins

        g1_bin_positions = get_center(
            atomgroup=self.g1, bin_method=self.bin_method, compound=self.comp_1
        )
        g2_bin_positions = get_center(
            atomgroup=self.g2, bin_method=self.bin_method, compound=self.comp_2
        )

        # Calculate planar pdf per bin by averaging over all atoms in one bin.
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
                relevant_xy_distances, bins=self.pdf_nbins, range=(self.dmin, self.dmax)
            )[0]

    def _conclude(self) -> None:
        super()._conclude()

        # Normalise pdf using the volumes of a ring with height 2*dz.
        ring_volumes = (
            np.pi * (self.edges[1:] ** 2 - self.edges[:-1] ** 2) * 2 * self.dzheight
        )
        ring_volumes = np.expand_dims(ring_volumes, axis=0)
        self.results.bins = self.results.bins
        self.results.pdf = self.means.count / self.means.n_g1 / ring_volumes
        self.results.pdf = np.nan_to_num(self.results.pdf.T, nan=0)

    @render_docs
    def save(self) -> None:
        """${SAVE_METHOD_DESCRIPTION}"""  # noqa: D415
        columns = ["r [Å]"]
        for z in self.results.bin_pos:
            columns.append(f"pdf at {z:.2f} Å [Å^-3]")

        self.savetxt(
            self.output,
            np.hstack([self.results.bins[:, np.newaxis], self.results.pdf]),
            columns=columns,
        )
