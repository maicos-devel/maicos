#!/usr/bin/env python
#
# Copyright (c) 2025 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the GNU Public Licence, v3 or any higher version
# SPDX-License-Identifier: GPL-3.0-or-later
"""Module for computing dipole angle timeseries."""

import MDAnalysis as mda
import numpy as np

from ..core import AnalysisBase
from ..lib.util import get_compound, render_docs, unit_vectors_planar
from ..lib.weights import diporder_weights


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
    ${GROUPING_PARAMETER}
    ${PDIM_PLANAR_PARAMETER}
    ${OUTPUT_PARAMETER}

    Attributes
    ----------
    results.t : numpy.ndarray
        time (ps).
    results.cos_theta_i : numpy.ndarray
        Average :math:`\cos` between dipole and axis.
    results.cos_theta_ii : numpy.ndarray
        Average :math:`\cos²` of the dipoles and axis.
    results.cos_theta_ij : numpy.ndarray
        Product :math:`\cos` of dipole i and cos of dipole j (``i != j``).

    References
    ----------
    .. footbibliography::

    """

    def __init__(
        self,
        atomgroup: mda.AtomGroup,
        unwrap: bool = False,
        pack: bool = True,
        refgroup: mda.AtomGroup | None = None,
        concfreq: int = 0,
        grouping: str = "residues",
        pdim: int = 2,
        output: str = "dipangle.dat",
        jitter: float = 0.0,
    ) -> None:
        self._locals = locals()
        self.wrap_compound = get_compound(atomgroup)
        super().__init__(
            atomgroup,
            refgroup=refgroup,
            unwrap=unwrap,
            pack=pack,
            concfreq=concfreq,
            wrap_compound=self.wrap_compound,
            jitter=jitter,
        )
        self.grouping = grouping
        self.pdim = pdim
        self.output = output

    def _prepare(self) -> None:
        self.n_residues = self.atomgroup.residues.n_residues

        def get_unit_vectors(atomgroup: mda.AtomGroup, grouping: str):
            return unit_vectors_planar(
                atomgroup=atomgroup, grouping=grouping, pdim=self.pdim
            )

        self.get_unit_vectors = get_unit_vectors

        self.cos_theta_i = np.empty(self.n_frames)
        self.cos_theta_ii = np.empty(self.n_frames)
        self.cos_theta_ij = np.empty(self.n_frames)

    def _single_frame(self) -> None:
        cos_theta = diporder_weights(
            self.atomgroup,
            grouping=self.grouping,
            order_parameter="cos_theta",
            get_unit_vectors=self.get_unit_vectors,
        )
        matrix = np.outer(cos_theta, cos_theta)

        trace = matrix.trace()
        self.cos_theta_i[self._frame_index] = cos_theta.mean()
        self.cos_theta_ii[self._frame_index] = trace / self.n_residues
        self.cos_theta_ij[self._frame_index] = matrix.sum() - trace
        self.cos_theta_ij[self._frame_index] /= self.n_residues**2 - self.n_residues

    def _conclude(self) -> None:
        self.results.t = self.times
        self.results.cos_theta_i = self.cos_theta_i[: self._index]
        self.results.cos_theta_ii = self.cos_theta_ii[: self._index]
        self.results.cos_theta_ij = self.cos_theta_ij[: self._index]

    @render_docs
    def save(self) -> None:
        """${SAVE_METHOD_DESCRIPTION}"""  # noqa: D415
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
