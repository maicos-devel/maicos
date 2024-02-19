#!/usr/bin/env python3
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
#
# Copyright (c) 2024 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the GNU Public Licence, v3 or any higher version
# SPDX-License-Identifier: GPL-3.0-or-later
"""Module for computing structure factor for dipoles."""
from typing import Optional

import MDAnalysis as mda
import numpy as np

from ..core import AnalysisBase
from ..lib.math import compute_structure_factor
from ..lib.util import get_center, render_docs, unit_vectors_planar
from ..lib.weights import diporder_weights


@render_docs
class DiporderStructureFactor(AnalysisBase):
    r"""Structure factor for dipoles.

    Extension the standard structure factor by weighting it with different the
    normalized dipole moment :math:`\hat{\boldsymbol{\mu}}` of a ``group`` according to

    .. math::
        S(q)_{\hat{\boldsymbol{\mu}} \hat{\boldsymbol{\mu}}} = \left \langle
        \frac{1}{N} \sum_{i,j=1}^N \hat \mu_i \hat \mu_j \, \exp(-i\boldsymbol q\cdot
        [\boldsymbol r_i - \boldsymbol r_j]) \right \rangle

    For the correlation time estimation the module will use the value of the structure
    factor with the smallest possible :math:`q` value.

    For an detailed example on the usage refer to the :ref:`how-to on dipolar
    correlation functions <howto-spatial-dipole-dipole-correlations>`. For general
    details on the theory behind the structure factor refer to :ref:`saxs-explanations`.

    Parameters
    ----------
    ${ATOMGROUP_PARAMETER}
    ${BASE_CLASS_PARAMETERS}
    ${Q_SPACE_PARAMETERS}
    ${OUTPUT_PARAMETER}

    Attributes
    ----------
    results.q : numpy.ndarray
        length of binned q-vectors
    results.struct_factor : numpy.ndarray
        Structure factor
    """

    def __init__(
        self,
        atomgroup: mda.AtomGroup,
        bin_method: str = "com",
        grouping: str = "molecules",
        refgroup: Optional[mda.AtomGroup] = None,
        unwrap: bool = True,
        jitter: float = 0.0,
        concfreq: int = 0,
        startq: float = 0,
        endq: float = 6,
        dq: float = 0.01,
        output: str = "sq.dat",
    ):
        self._locals = locals()
        super().__init__(
            atomgroup,
            unwrap=unwrap,
            refgroup=refgroup,
            jitter=jitter,
            wrap_compound=grouping,
            concfreq=concfreq,
        )

        self.bin_method = str(bin_method).lower()
        self.startq = startq
        self.endq = endq
        self.dq = dq
        self.output = output

    def _prepare(self):
        self.n_bins = int(np.ceil((self.endq - self.startq) / self.dq))

    def _single_frame(self):
        box = np.diag(mda.lib.mdamath.triclinic_vectors(self._ts.dimensions))

        positions = get_center(
            atomgroup=self.atomgroup,
            bin_method=self.bin_method,
            compound=self.wrap_compound,
        )

        self._obs.struct_factor = np.zeros(self.n_bins)

        # Calculate structure factor per vector component and sum them up
        for pdim in range(3):

            def get_unit_vectors(
                atomgroup: mda.AtomGroup, grouping: str, pdim: int = pdim
            ):
                return unit_vectors_planar(
                    atomgroup=atomgroup, grouping=grouping, pdim=pdim
                )

            weights = diporder_weights(
                atomgroup=self.atomgroup,
                grouping=self.wrap_compound,
                order_parameter="cos_theta",
                get_unit_vectors=get_unit_vectors,
            )

            q_ts, S_ts = compute_structure_factor(
                np.double(positions),
                np.double(box),
                self.startq,
                self.endq,
                0,
                np.pi,
                weights,
            )

            q_ts = q_ts.flatten()
            S_ts = S_ts.flatten()
            nonzeros = np.where(S_ts != 0)[0]

            q_ts = q_ts[nonzeros]
            S_ts = S_ts[nonzeros]

            struct_ts, _ = np.histogram(
                a=q_ts,
                bins=self.n_bins,
                range=(self.startq, self.endq),
                weights=S_ts,
            )
            bincount, _ = np.histogram(
                a=q_ts,
                bins=self.n_bins,
                range=(self.startq, self.endq),
                weights=None,
            )
            with np.errstate(invalid="ignore"):
                struct_ts /= bincount

            self._obs.struct_factor += np.nan_to_num(struct_ts)

        # Normalize with respect to the number of compounds
        self._obs.struct_factor /= len(positions)

        return self._obs.struct_factor[-1]

    def _conclude(self):
        q = np.arange(self.startq, self.endq, self.dq) + 0.5 * self.dq
        nonzeros = np.where(self.means.struct_factor != 0)[0]
        struct_factor = self.means.struct_factor[nonzeros]

        self.results.q = q[nonzeros]
        self.results.struct_factor = struct_factor

    @render_docs
    def save(self):
        """${SAVE_DESCRIPTION}"""
        self.savetxt(
            self.output,
            np.vstack([self.results.q, self.results.struct_factor]).T,
            columns=["q (1/Å)", "S(q) (arb. units)"],
        )
