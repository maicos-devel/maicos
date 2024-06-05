#!/usr/bin/env python3
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
#
# Copyright (c) 2024 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the GNU Public Licence, v3 or any higher version
# SPDX-License-Identifier: GPL-3.0-or-later
"""Module for computing Radial Distribution functions for dipoles."""
from typing import Optional

import MDAnalysis as mda
import numpy as np
from MDAnalysis.lib import distances

from ..core import AnalysisBase
from ..lib.util import get_center, render_docs
from ..lib.weights import diporder_pair_weights


@render_docs
class RDFDiporder(AnalysisBase):
    r"""Spherical Radial Distribution function between dipoles.

    The implementation is heavily inspired by :class:`MDAnalysis.analysis.rdf.InterRDF`
    and is according to :footcite:t:`zhang_dipolar_2014` given by

    .. math::
        g_\mathrm{\hat{\boldsymbol{\mu}}, \hat{\boldsymbol{\mu}}}(r) = \frac{1}{N}
        \left\langle \sum_i \frac{1}{n_i(r)} \sum_{j=1}^{n_i(r)}
        (\hat{\boldsymbol{\mu}}_i \cdot \hat{\boldsymbol{\mu}}_j) \right \rangle

    where :math:`\hat{\boldsymbol{\mu}}` is the normalized dipole moment of a
    ``grouping`` and :math:`n_i(r)` is the number of dipoles within a spherical shell of
    distance :math:`r` and :math:`r + \delta r` from dipole :math:`i`.

    For the correlation time estimation the module will use the value of the RDF with
    the largest possible :math:`r` value.

    For an detailed example on the usage refer to the :ref:`how-to on dipolar
    correlation functions <howto-spatial-dipole-dipole-correlations>`.

    Parameters
    ----------
    ${PDF_PARAMETERS}
    ${BIN_WIDTH_PARAMETER}
    ${RADIAL_CLASS_PARAMETERS}
    ${BIN_METHOD_PARAMETER}
    norm : str, {'rdf', 'density', 'none'}
        For 'rdf' calculate :math:`g_{ab}(r)`. For 'density' the single group density
        :math:`n_{ab}(r)` is computed. 'none' computes the number of particles
        occurences in each spherical shell.
    ${GROUPING_PARAMETER}
    ${BASE_CLASS_PARAMETERS}
    ${OUTPUT_PARAMETER}

    Attributes
    ----------
    results.bins: numpy.ndarray
        radial distances to which the RDF is calculated with shape (``rdf_nbins``) (Å)
    results.rdf: numpy.ndarray
        RDF either in :math:`\text{eÅ}^{-2}` if norm is ``"rdf"`` or ``"density"`` or
        :math:`\text{eÅ}` if norm is ``"none"``.

    References
    ----------
    .. footbibliography::
    """

    def __init__(
        self,
        g1: mda.AtomGroup,
        g2: Optional[mda.AtomGroup] = None,
        bin_width: float = 0.1,
        rmin: float = 0.0,
        rmax: float = 15.0,
        bin_method: str = "com",
        norm: str = "rdf",
        grouping: str = "residues",
        unwrap: bool = True,
        pack: bool = True,
        refgroup: Optional[mda.AtomGroup] = None,
        jitter: float = 0.0,
        concfreq: int = 0,
        output: str = "diporderrdf.dat",
    ) -> None:
        self._locals = locals()
        super().__init__(
            g1,
            unwrap=unwrap,
            pack=pack,
            refgroup=refgroup,
            jitter=jitter,
            wrap_compound=grouping,
            concfreq=concfreq,
        )

        self.g1 = g1
        if g2 is None:
            self.g2 = g1
        else:
            self.g2 = g2

        self.bin_width = bin_width
        self.rmin = rmin
        self.rmax = rmax
        self.bin_method = str(bin_method).lower()
        self.norm = norm
        self.output = output

    def _prepare(self):
        self.n_bins = int(np.ceil((self.rmax - self.rmin) / self.bin_width))

        supported_norms = ["rdf", "density", "none"]
        if self.norm not in supported_norms:
            raise ValueError(
                f"{self.norm!r} is an invalid `norm`. "
                f"Choose from: {', '.join(supported_norms)}"
            )

    def _single_frame(self):
        if self.unwrap:
            self.g1.unwrap(compound=self.wrap_compound)
            self.g2.unwrap(compound=self.wrap_compound)

        pos_1 = get_center(
            self.g1, bin_method=self.bin_method, compound=self.wrap_compound
        )
        pos_2 = get_center(
            self.g2, bin_method=self.bin_method, compound=self.wrap_compound
        )

        pairs, dist = distances.capped_distance(
            pos_1,
            pos_2,
            min_cutoff=self.rmin,
            max_cutoff=self.rmax,
            box=self._ts.dimensions,
        )

        weights = diporder_pair_weights(self.g1, self.g2, compound=self.wrap_compound)
        weights_sel = np.array([weights[ix[0], ix[1]] for ix in pairs])

        self._obs.profile, _ = np.histogram(
            a=dist,
            bins=self.n_bins,
            range=(self.rmin, self.rmax),
            weights=weights_sel,
        )

        if self.norm == "rdf":
            self._obs.volume = self._ts.volume

        return self._obs.profile[-1]

    def _conclude(self):
        _, edges = np.histogram(a=[-1], bins=self.n_bins, range=(self.rmin, self.rmax))
        self.results.bins = 0.5 * (edges[:-1] + edges[1:])

        norm = 1
        if self.norm in ["rdf", "density"]:
            # Volume in each radial shell
            vols = np.power(edges, 3)
            norm *= 4 / 3 * np.pi * np.diff(vols)

        if self.norm == "rdf":
            # Number of each selection
            if self.wrap_compound != "molecules":
                nA = getattr(self.g1, f"n_{self.wrap_compound}")
                nB = getattr(self.g2, f"n_{self.wrap_compound}")
            else:
                nA = len(np.unique(self.g1.molnums))
                nB = len(np.unique(self.g1.molnums))

            N = nA * nB

            # Average number density
            norm *= N / self.means.volume

        self.results.rdf = self.means.profile / norm

    @render_docs
    def save(self) -> None:
        """${SAVE_METHOD_DESCRIPTION}"""
        columns = ["r (Å)", "rdf"]
        if self.norm in ["rdf", "density"]:
            columns[1] += " (Å^3)"

        self.savetxt(
            self.output,
            np.vstack([self.results.bins, self.results.rdf]).T,
            columns=columns,
        )
