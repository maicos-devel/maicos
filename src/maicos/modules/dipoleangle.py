#!/usr/bin/env python
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
#
# Copyright (c) 2023 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the GNU Public Licence, v3 or any higher version
# SPDX-License-Identifier: GPL-3.0-or-later
"""Module for computing dipole angle timeseries."""

from typing import Optional

import MDAnalysis as mda
import numpy as np

from ..core import AnalysisBase
from ..lib.util import get_compound, render_docs


@render_docs
class DipoleAngle(AnalysisBase):
    r"""Angle timeseries of dipole moments with respect to an axis.

    The analysis can be applied to study the orientational dynamics of water molecules
    during an excitation pulse. For more details read
    :footcite:t:`elgabartyEnergyTransferHydrogen2020`.

    Parameters
    ----------
    ${ATOMGROUP_PARAMETER}
    ${BASE_CLASS_PARAMETERS}
    dim : int
        Reference vector for angle (``x=0``, ``y=1``, ``z=2``).
    output : str
       Prefix for output filenames.
    concfreq : int
        Default number of frames after which results are calculated and files refreshed.
        If ``0`` results are only calculated at the end of the analysis and not saved by
        default.

    Attributes
    ----------
    results.t : numpy.ndarray
        time (ps).
    resulst.cos_theta_i : numpy.ndarray
        Average :math:`\cos` between dipole and axis.
    resulst.cos_theta_ii : numpy.ndarray
        Average :math:`\cos²` of the dipoles and axis.
    resulst.cos_theta_ij : numpy.ndarray
        Product :math:`\cos` of dipole i and cos of dipole j (``i != j``).

    References
    ----------
    .. footbibliography::
    """

    def __init__(
        self,
        atomgroup: mda.AtomGroup,
        refgroup: Optional[mda.AtomGroup] = None,
        unwrap: bool = False,
        concfreq: float = 0,
        dim: int = 2,
        output: str = "dipangle.dat",
        jitter: float = 0.0,
    ):
        self.wrap_compound = get_compound(atomgroup)
        super().__init__(
            atomgroup,
            refgroup=refgroup,
            unwrap=unwrap,
            concfreq=concfreq,
            wrap_compound=self.wrap_compound,
            jitter=jitter,
        )
        self.dim = dim
        self.output = output

    def _prepare(self):
        self.n_residues = self.atomgroup.residues.n_residues

        # unit normal vector
        self.unit = np.zeros(3)
        self.unit[self.dim] += 1

        self.cos_theta_i = np.empty(self.n_frames)
        self.cos_theta_ii = np.empty(self.n_frames)
        self.cos_theta_ij = np.empty(self.n_frames)

    def _single_frame(self):
        # make broken molecules whole again!
        self.atomgroup.unwrap(compound="molecules")

        chargepos = self.atomgroup.positions * self.atomgroup.charges[:, np.newaxis]
        dipoles = self.atomgroup.accumulate(chargepos, compound=self.wrap_compound)

        cos_theta = np.dot(dipoles, self.unit) / np.linalg.norm(dipoles, axis=1)
        matrix = np.outer(cos_theta, cos_theta)

        trace = matrix.trace()
        self.cos_theta_i[self._frame_index] = cos_theta.mean()
        self.cos_theta_ii[self._frame_index] = trace / self.n_residues
        self.cos_theta_ij[self._frame_index] = matrix.sum() - trace
        self.cos_theta_ij[self._frame_index] /= self.n_residues**2 - self.n_residues

        if (
            self.concfreq
            and self._frame_index % self.concfreq == 0
            and self._frame_index > 0
        ):
            self._conclude()
            self.save()

    def _conclude(self):
        self.results.t = self.times
        self.results.cos_theta_i = self.cos_theta_i[: self._index]
        self.results.cos_theta_ii = self.cos_theta_ii[: self._index]
        self.results.cos_theta_ij = self.cos_theta_ij[: self._index]

    def save(self):
        """Save result."""
        self.savetxt(
            self.output,
            np.vstack(
                [
                    self.results.t,
                    self.results.cos_theta_i,
                    self.results.cos_theta_ii,
                    self.results.cos_theta_ij,
                ]
            ).T,
            columns=["t", "<cos(θ_i)>", "<cos(θ_i)cos(θ_i)>", "<cos(θ_i)cos(θ_j)>"],
        )
