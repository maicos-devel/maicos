#!/usr/bin/env python3
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
#
# Copyright (c) 2022 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the GNU Public Licence, v3 or any higher version
# SPDX-License-Identifier: GPL-3.0-or-later
r"""Module for computing 1D cylindrical pair distribution functions."""

import logging
from typing import Optional

import MDAnalysis as mda
import numpy as np
from MDAnalysis.lib.distances import capped_distance

from ..core import CylinderBase
from ..lib.math import transform_cylinder
from ..lib.util import get_center, get_compound, render_docs


logger = logging.getLogger(__name__)


@render_docs
class PDFCylinder(CylinderBase):
    r"""Shell-wise one-dimensional (cylindrical) pair distribution functions.

    The one-dimensional pair distribution functions :math:`g_{\text{1d}}(\phi)`
    and :math:`g_{\text{1d}}(z)` describes the pair distribution to particles
    which lie on the same cylinder along the angular and axial directions
    respectively. These functions can be used in cylindrical systems that are
    inhomogeneous along radial coordinate, and homogeneous in the angular and
    axial directions. It gives the average number density of :math:`g2` as a
    function of angular and axial distances respectively from a :math:`g1` atom.
    Then the angular pair distribution function is

    .. math::

            g_{\text{1d}}(\phi) = \left \langle \sum_{i}^{N_{g_1}}
            \sum_{j}^{N_{g2}} \delta(\phi - \phi_{ij}) \delta(R_{ij}) \delta(z_{ij})
            \right \rangle


    And the axial pair distribution function is

    .. math::

            g_{\text{1d}}(z) = \left \langle \sum_{i}^{N_{g_1}}
            \sum_{j}^{N_{g2}} \delta(z - z_{ij}) \delta(R_{ij}) \delta(\phi_{ij})
            \right \rangle

    Even though due to consistency reasons the results are called pair distribution
    functions the output is not unitless. The default output is is in dimension of
    number/volume in :math:`Å^{-3}`. If ``density`` is set to :py:obj:`True`, the
    output is normalised by the density of :math:`g2`.

    Parameters
    ----------
    ${PDF_PARAMETERS}
    pdf_z_bin_width : float
        Binwidth of bins in the histogram of the axial PDF (Å).
    pdf_phi_bin_width : float
        Binwidth of bins in the histogram of the angular PDF (Å).
    drwidth : float
        radial width of a PDF cylindrical shell (Å), and axial or angular (arc) slices.
    dmin: float
        the minimum pairwise distance between 'g1' and 'g2' (Å).
    dmax : float
        the maximum pairwise distance between 'g1' and 'g2' (Å).
    density : bool
        normalise the PDF by the density of 'g2' (:math:`Å^{-3}`).
    origin : numpy.ndarray
        Set origin of the cylindrical coordinate system (x,y,z). If :obj:`None` the
        origin will be set according to the ``refgroup`` parameter.
    ${BIN_METHOD_PARAMETER}
    ${CYLINDER_CLASS_PARAMETERS}
    ${OUTPUT_PARAMETER}

    Attributes
    ----------
    ${CYLINDER_CLASS_ATTRIBUTES}
    results.phi_bins: numpy.ndarray
        Angular distances to which the PDF is calculated with shape (`pdf_nbins`) (Å)
    results.z_bins: numpy.ndarray
        axial distances to which the PDF is calculated with shape (`pdf_nbins`) (Å)
    results.phi_pdf: numpy.ndarray
        Angular PDF with shape (`pdf_nbins`, `n_bins`) (:math:`\text{Å}^{-3}`)
    results.z_pdf: numpy.ndarray
        Axial PDF with shape (`pdf_nbins`, `n_bins`) (:math:`\text{Å}^{-3}`)
    """

    def __init__(
        self,
        g1: mda.AtomGroup,
        g2: Optional[mda.AtomGroup] = None,
        bin_width_pdf_z: float = 0.3,
        bin_width_pdf_phi: float = 0.1,
        drwidth: float = 0.1,
        dmin: Optional[float] = None,
        dmax: Optional[float] = None,
        density: bool = False,
        origin: Optional[np.ndarray] = None,
        bin_method: str = "com",
        unwrap: bool = False,
        refgroup: Optional[mda.AtomGroup] = None,
        jitter: float = 0.0,
        concfreq: int = 0,
        dim: int = 2,
        zmin: Optional[float] = None,
        zmax: Optional[float] = None,
        rmin: float = 0,
        rmax: Optional[float] = None,
        bin_width: float = 1,
        output: str = "pdf.dat",
    ):
        self.comp_1 = get_compound(g1)
        super(PDFCylinder, self).__init__(
            atomgroups=g1,
            refgroup=refgroup,
            unwrap=unwrap,
            concfreq=concfreq,
            jitter=jitter,
            dim=dim,
            rmin=rmin,
            rmax=rmax,
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

        self.bin_width_pdf_phi = bin_width_pdf_phi
        self.bin_width_pdf_z = bin_width_pdf_z
        self.drwidth = drwidth
        self.bin_width = bin_width
        self.output = output
        self.bin_method = bin_method.lower()

        if origin is not None and origin.shape != (3,):
            raise ValueError(
                f"Origin has length {origin.shape} but only (3,) is allowed."
            )
        else:
            self.origin = origin

        self.comp_2 = get_compound(self.g2)
        self.nbins_pdf_phi = 100
        self.nbins_pdf_z = 100

        self.dmin = dmin
        self.dmax = dmax
        self.density = density

    def _prepare(self):
        super(PDFCylinder, self)._prepare()
        logger.info("Compute pair distribution function.")

        if self.origin is None:
            self.origin = self.box_center

        if self.dmin is None:
            self.dmin = 0
        if self.dmax is None:
            self.dmax = self.box_center[self.dim]
        else:
            if self.dmax > self.box_center[self.dim]:
                raise ValueError(
                    "Axial range of PDF exceeds half of the box size. "
                    "This will lead to unexpected results."
                )

        if self.bin_width_pdf_z > 0:
            self.nbins_pdf_z = int(
                np.ceil((self.dmax - self.dmin) / self.bin_width_pdf_z)
            )
            self.bin_width_pdf_z = (self.dmax - self.dmin) / self.nbins_pdf_z
        else:
            raise ValueError("PDF bin_width must be a positive number.")
        if self.bin_width_pdf_phi > 0:
            self.nbins_pdf_phi = int(np.ceil(np.pi / self.bin_width_pdf_phi))
            self.bin_width_pdf_phi = np.pi / self.nbins_pdf_phi
        else:
            raise ValueError("PDF bin_width must be a positive number.")

        if self.bin_method not in ["cog", "com", "coc"]:
            raise ValueError(
                f"{self.bin_method} is an unknown binning method. Use `cog`, `com` or "
                "`coc`."
            )

        logger.info(
            f"Using {self.nbins_pdf_phi} pdf bins in phi direction and "
            f"{self.nbins_pdf_z} in z direction."
        )

    def _single_frame(self):
        super(PDFCylinder, self)._single_frame()
        self._obs.n_g1 = np.zeros((self.n_bins, 1))
        self._obs.n_g2 = np.zeros((self.n_bins, 1))
        self._obs.count_phi = np.zeros((self.n_bins, self.nbins_pdf_phi))
        self._obs.count_z = np.zeros((self.n_bins, self.nbins_pdf_z))

        # Get the center of each atom in g1 and g2.
        g1_bin_positions = get_center(
            atomgroup=self.g1, bin_method=self.bin_method, compound=self.comp_1
        )
        g2_bin_positions = get_center(
            atomgroup=self.g2, bin_method=self.bin_method, compound=self.comp_2
        )

        # convert to cylinderical coordinates
        g1_bin_positions_cyl = transform_cylinder(
            g1_bin_positions, origin=self.origin, dim=self.dim
        )
        g2_bin_positions_cyl = transform_cylinder(
            g2_bin_positions, origin=self.origin, dim=self.dim
        )

        # Calculate pdf per bin by averaging over all atoms in one bin.
        for r_bin in range(0, self.n_bins):
            # Get all atoms in a bin.
            g1_in_rbin_positions = g1_bin_positions_cyl[
                np.logical_and(
                    g1_bin_positions_cyl[:, 0] >= self._obs.bin_edges[r_bin],
                    g1_bin_positions_cyl[:, 0] < self._obs.bin_edges[r_bin + 1],
                )
            ]

            g2_in_rbin_positions = g2_bin_positions_cyl[
                np.logical_and(
                    g2_bin_positions_cyl[:, 0]
                    >= self._obs.bin_edges[r_bin] - self.drwidth,
                    g2_bin_positions_cyl[:, 0]
                    < self._obs.bin_edges[r_bin + 1] + self.drwidth,
                )
            ]

            self._obs.n_g1[r_bin] = len(g1_in_rbin_positions)
            self._obs.n_g2[r_bin] = len(g2_in_rbin_positions)

            # Below we abuse the 3D `capped_distance` search for do a distance search in
            # 1D distance by setting the other positions as well as the box size in
            # these directions to 0.

            # Filter only those pairs with delta r < dr.
            r_pairs = capped_distance(
                g1_in_rbin_positions * [1, 0, 0],
                g2_in_rbin_positions * [1, 0, 0],
                self.drwidth,
                box=None,
                return_distances=False,
            )

            # Filter only those pairs with delta phi < dphi.
            phi_pairs = capped_distance(
                g1_in_rbin_positions * [0, 1, 0],
                g2_in_rbin_positions * [0, 1, 0],
                # define: r dphi = drwidth
                # therefore: dphi = drwidth / r
                self.drwidth / self._obs.bin_pos[r_bin],
                box=[0, 2 * np.pi, 0, 90, 90, 90],
                return_distances=False,
            )

            # Filter only those pairs with delta z < dz.
            z_pairs = capped_distance(
                g1_in_rbin_positions * [0, 0, 1],
                g2_in_rbin_positions * [0, 0, 1],
                self.drwidth,  # define: dz = drwidth
                box=[0, 0, self._universe.dimensions[self.dim], 90, 90, 90],
                return_distances=False,
            )

            # Calculate pairwise phi distances between g1 and g2.
            phi_dist_pairs, phi_distances = capped_distance(
                g1_in_rbin_positions * [0, 1, 0],
                g2_in_rbin_positions * [0, 1, 0],
                np.pi,  # maximum phi distance is pi
                box=[
                    0,
                    2 * np.pi,
                    0,
                    90,
                    90,
                    90,
                ],  # minimum image convention in phi direction (0, 2pi)
            )

            # Calculate pairwise z distances between g1 and g2.
            z_dist_pairs, z_distances = capped_distance(
                g1_in_rbin_positions * [0, 0, 1],
                g2_in_rbin_positions * [0, 0, 1],
                self.dmax,
                box=[
                    0,
                    0,
                    self._universe.dimensions[self.dim],
                    90,
                    90,
                    90,
                ],  # minimum image convention in z direction (0, boxsize)
            )

            # Map pairs (i, j) to a number i+N*j (so we can use np.isin).
            r_pairs_encode = r_pairs[:, 0] + self._obs.n_g2[r_bin] * r_pairs[:, 1]
            phi_pairs_encode = phi_pairs[:, 0] + self._obs.n_g2[r_bin] * phi_pairs[:, 1]
            z_pairs_encode = z_pairs[:, 0] + self._obs.n_g2[r_bin] * z_pairs[:, 1]
            phi_dist_pairs_encode = (
                phi_dist_pairs[:, 0] + self._obs.n_g2[r_bin] * phi_dist_pairs[:, 1]
            )
            z_dist_pairs_encode = (
                z_dist_pairs[:, 0] + self._obs.n_g2[r_bin] * z_dist_pairs[:, 1]
            )

            # Filter pairs that are in the same dr bin and dz bin.
            mask_in_dr_and_dz = np.isin(
                phi_dist_pairs_encode, r_pairs_encode
            ) * np.isin(phi_dist_pairs_encode, z_pairs_encode)

            # Filter pairs that are in the same dr bin and dphi bin.
            mask_in_dr_and_dphi = np.isin(
                z_dist_pairs_encode, r_pairs_encode
            ) * np.isin(z_dist_pairs_encode, phi_pairs_encode)

            mask_same_atom = phi_distances > 0
            relevant_phi_distances = phi_distances[mask_in_dr_and_dz * mask_same_atom]

            mask_same_atom = z_distances > 0
            relevant_z_distances = z_distances[mask_in_dr_and_dphi * mask_same_atom]

            # Histogram the pairwise distances.
            self._obs.count_phi[r_bin] = np.histogram(
                relevant_phi_distances, bins=self.nbins_pdf_phi, range=(0, np.pi)
            )[0]

            self._obs.count_z[r_bin] = np.histogram(
                relevant_z_distances,
                bins=self.nbins_pdf_z,
                range=(self.dmin, self.dmax),
            )[0]

    def _conclude(self):
        super()._conclude()

        # Calculate the density of g2.
        if self.density:
            g2_density = self.means.n_g2 / self.means.bin_volume
        else:
            g2_density = 1

        # Normalising volume for the angular PDF. This is 2(R*dR*2dz*2dphi),
        # where R is the radius of the bin center, dR is the width of the pdf bin.
        phi_norm = (
            np.array(
                [
                    2
                    * (self.means.bin_edges[1:] + self.means.bin_edges[:-1])
                    / 2
                    * self.bin_width_pdf_phi
                    * 2
                    * self.drwidth
                    * 2
                    * self.drwidth
                ]
            ).T
            * g2_density
        )

        # Normalising volume for the axial PDF. This is 2(dZ*2dr*2dz),
        # where dZ is the width of the pdf bin.
        z_norm = (
            2 * self.bin_width_pdf_z * 2 * self.drwidth * 2 * self.drwidth * g2_density
        )

        # Normalise pdf using the normalisation factor.
        with np.errstate(invalid="ignore", divide="ignore"):
            pdf_phi = self.means.count_phi / self.means.n_g1 / phi_norm
        self.results.pdf_phi = np.nan_to_num(pdf_phi, nan=0, posinf=0, neginf=0)

        with np.errstate(invalid="ignore", divide="ignore"):
            pdf_z = self.means.count_z / self.means.n_g1 / z_norm
        self.results.pdf_z = np.nan_to_num(pdf_z, nan=0, posinf=0, neginf=0)

        # Calculate the bin centers.
        edges_phi = np.histogram([-1], bins=self.nbins_pdf_phi, range=(0, np.pi))[1]
        edges_z = np.histogram(
            [-1], bins=self.nbins_pdf_z, range=(self.dmin, self.dmax)
        )[1]
        self.results.bins_phi = 0.5 * (edges_phi[1:] + edges_phi[:-1])
        self.results.bins_z = 0.5 * (edges_z[1:] + edges_z[:-1])

    @render_docs
    def save(self):
        """${SAVE_DESCRIPTION}"""
        columns = ["r [Å]"]
        for r in self.results.bin_pos:
            columns.append(f"pdf at {r:.2f} Å [Å^-3]")

        self.savetxt(
            "phi_" + self.output,
            np.hstack([self.results.bins_phi[:, np.newaxis], self.results.pdf_phi.T]),
            columns=columns,
        )

        self.savetxt(
            "z_" + self.output,
            np.hstack([self.results.bins_z[:, np.newaxis], self.results.pdf_z.T]),
            columns=columns,
        )
