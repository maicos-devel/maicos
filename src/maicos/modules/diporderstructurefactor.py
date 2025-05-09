#!/usr/bin/env python3
#
# Copyright (c) 2025 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the GNU Public Licence, v3 or any higher version
# SPDX-License-Identifier: GPL-3.0-or-later
"""Module for computing structure factor for dipoles."""

import logging

import MDAnalysis as mda
import numpy as np

from ..core import AnalysisBase
from ..lib.math import compute_structure_factor
from ..lib.util import get_center, render_docs, unit_vectors_planar
from ..lib.weights import diporder_weights


@render_docs
class DiporderStructureFactor(AnalysisBase):
    r"""Structure factor for dipoles.

    Extension the standard structure factor :math:`S(q)` by weighting it with different
    the normalized dipole moment :math:`\hat{\boldsymbol{\mu}}` of a ``group`` according
    to

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
    results.structure_factors : numpy.ndarray
        Structure factor

    """

    def __init__(
        self,
        atomgroup: mda.AtomGroup,
        bin_method: str = "com",
        grouping: str = "molecules",
        refgroup: mda.AtomGroup | None = None,
        unwrap: bool = True,
        pack: bool = True,
        jitter: float = 0.0,
        concfreq: int = 0,
        qmin: float = 0,
        qmax: float = 6,
        dq: float = 0.01,
        output: str = "sq.dat",
    ) -> None:
        self._locals = locals()
        super().__init__(
            atomgroup,
            unwrap=unwrap,
            pack=pack,
            refgroup=refgroup,
            jitter=jitter,
            wrap_compound=grouping,
            concfreq=concfreq,
        )

        self.bin_method = str(bin_method).lower()
        self.qmin = qmin
        self.qmax = qmax
        self.dq = dq
        self.output = output

    def _prepare(self) -> None:
        logging.info("Analysis of the structure factor of dipoles.")
        self.n_bins = int(np.ceil((self.qmax - self.qmin) / self.dq))

    def _single_frame(self) -> float:
        box = np.diag(mda.lib.mdamath.triclinic_vectors(self._ts.dimensions))

        positions = get_center(
            atomgroup=self.atomgroup,
            bin_method=self.bin_method,
            compound=self.wrap_compound,
        )

        self._obs.structure_factors = np.zeros(self.n_bins)

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

            scattering_vectors, structure_factors = compute_structure_factor(
                np.double(positions),
                np.double(box),
                self.qmin,
                self.qmax,
                0,
                np.pi,
                weights,
            )

            scattering_vectors = scattering_vectors.flatten()
            structure_factors = structure_factors.flatten()
            nonzeros = np.where(structure_factors != 0)[0]

            scattering_vectors = scattering_vectors[nonzeros]
            structure_factors = structure_factors[nonzeros]

            histogram_kwargs = dict(
                a=scattering_vectors,
                bins=self.n_bins,
                range=(self.qmin, self.qmax),
            )
            structure_factors_binned, _ = np.histogram(
                weights=structure_factors, **histogram_kwargs
            )
            bincount, _ = np.histogram(weights=None, **histogram_kwargs)
            with np.errstate(invalid="ignore"):
                structure_factors_binned /= bincount

            self._obs.structure_factors += np.nan_to_num(structure_factors_binned)

        # Normalize with respect to the number of compounds
        self._obs.structure_factors /= len(positions)

        return self._obs.structure_factors[-1]

    def _conclude(self) -> None:
        scattering_vectors = np.arange(self.qmin, self.qmax, self.dq) + 0.5 * self.dq
        nonzeros = np.where(self.means.structure_factors != 0)[0]
        structure_factors = self.means.structure_factors[nonzeros]

        self.results.scattering_vectors = scattering_vectors[nonzeros]
        self.results.structure_factors = structure_factors

    @render_docs
    def save(self) -> None:
        """${SAVE_METHOD_DESCRIPTION}"""  # noqa: D415
        self.savetxt(
            self.output,
            np.vstack(
                [self.results.scattering_vectors, self.results.structure_factors]
            ).T,
            columns=["q (1/Å)", "S(q) (arb. units)"],
        )
