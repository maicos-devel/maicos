#!/usr/bin/env python
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
#
# Copyright (c) 2024 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the GNU Public Licence, v3 or any higher version
# SPDX-License-Identifier: GPL-3.0-or-later
r"""Module for computing Saxs structure factors and scattering intensities."""

import logging
from typing import Optional

import MDAnalysis as mda
import numpy as np

from ..core import AnalysisBase
from ..lib import tables
from ..lib.math import compute_form_factor, compute_structure_factor
from ..lib.util import render_docs


logger = logging.getLogger(__name__)


@render_docs
class Saxs(AnalysisBase):
    r"""Small angle X-Ray scattering intensities (SAXS).

    This module computes the structure factor :math:`S(q)`, the scattering intensity
    :math:`I(q)` and their corresponding scattering vectors :math:`q`. For a system
    containing only one element the structure factor and the scattering intensity are
    connected via the form factor :math:`f(q)`

    .. math::
        I(q) = [f(q)]^2 S(q)

    For more details on the theory behind this module see :ref:`saxs-explanations`.

    By default the scattering vectors :math:`\boldsymbol{q}` are binned according to
    their length :math:`q` using a bin width given by ``dq``. Setting the option
    ``bin_spectrum=False``, also the raw scattering vectors and their corresponding
    Miller indices can be saved. Saving the scattering vectors and Miller indices is
    only possible when the box vectors are constant in the whole trajectory (NVT) since
    for changing cells the same Miller indices correspond to different scattering
    vectors.

    Analyzed scattering vectors :math:`q` can be restricted by a minimal and maximal
    angle with the z-axis. For ``0`` and ``180``, all possible vectors are taken into
    account. To obtain the scattering intensities, the structure factor is normalized by
    an element-specific form factor based on Cromer-Mann parameters
    :footcite:t:`princeInternationalTablesCrystallography2004`.

    For the correlation time estimation the module will use the value of the scattering
    intensity with the largest possible :math:`q` value.

    For an example on the usage refer to :ref:`How-to: SAXS<howto-saxs>`.

    Parameters
    ----------
    ${ATOMGROUP_PARAMETER}
    ${BASE_CLASS_PARAMETERS}
    bin_spectrum : bool
        Bin the spectrum. If :py:obj:`False` Miller indices of q-vector are returned.
        Only works for NVT simulations.
    ${Q_SPACE_PARAMETERS}
    thetamin : float
        Minimal angle (°) between the q vectors and the z-axis.
    thetamax : float
        Maximal angle (°) between the q vectors and the z-axis.
    ${OUTPUT_PARAMETER}

    Attributes
    ----------
    results.scattering_vectors : numpy.ndarray
        Length of the binned scattering vectors.
    results.miller_indices : numpy.ndarray
        Miller indices of q-vector (only available if ``bin_spectrum==False``).
    results.struture_factors : numpy.ndarray
        structure factors :math:`S(q)`
    results.scattering_intensities : numpy.ndarray
        scattering intensities :math:`I(q)`

    References
    ----------
    .. footbibliography::
    """

    def __init__(
        self,
        atomgroup: mda.AtomGroup,
        unwrap: bool = False,
        pack: bool = True,
        refgroup: Optional[mda.AtomGroup] = None,
        jitter: float = 0.0,
        concfreq: int = 0,
        bin_spectrum: bool = True,
        qmin: float = 0,
        qmax: float = 6,
        dq: float = 0.1,
        thetamin: float = 0,
        thetamax: float = 180,
        output: str = "sq.dat",
    ) -> None:
        self._locals = locals()
        super().__init__(
            atomgroup,
            unwrap=unwrap,
            pack=pack,
            refgroup=refgroup,
            jitter=jitter,
            concfreq=concfreq,
            wrap_compound="atoms",
        )
        self.bin_spectrum = bin_spectrum
        self.qmin = qmin
        self.qmax = qmax
        self.dq = dq
        self.thetamin = thetamin
        self.thetamax = thetamax
        self.output = output

    def _prepare(self) -> None:
        self.thetamin = min(self.thetamin, self.thetamax)
        self.thetamax = max(self.thetamin, self.thetamax)

        if self.thetamin < 0 or self.thetamin > 180:
            raise ValueError(f"thetamin ({self.thetamin}°) has to between 0 and 180°.")

        if self.thetamax < 0 or self.thetamax > 180:
            raise ValueError(f"thetamax ({self.thetamax}°) has to between 0 and 180°.")

        if self.thetamin > self.thetamax:
            raise ValueError(
                f"thetamin ({self.thetamin}°) larger than thetamax ({self.thetamax}°)."
            )

        # Convert angles from degrees to radians
        self.thetamin *= np.pi / 180
        self.thetamax *= np.pi / 180

        self.groups = []
        self.weights = []
        self.atom_types = []

        logger.info("\nMap the following atomtypes:")
        for atom_type in np.unique(self.atomgroup.types).astype(str):
            try:
                element = tables.atomtypes[atom_type]
            except KeyError:
                raise KeyError(
                    f"No suitable element for '{atom_type}' found. You can change the "
                    f"`types` of the input `atomgroup` to match the known elements in "
                    "`maicos.lib.tables.atomtypes`."
                )
            if element == "DUM":
                continue

            group = self.atomgroup.select_atoms("type {}*".format(atom_type))

            self.groups.append(group)
            # Actual weights (form factors) are applied in post processing after
            self.weights.append(np.ones(group.n_atoms))
            self.atom_types.append(atom_type)

            logger.info("{:>14} --> {:>5}".format(atom_type, element))

        if self.bin_spectrum:
            self.n_bins = int(np.ceil((self.qmax - self.qmin) / self.dq))
        else:
            self.box = np.diag(
                mda.lib.mdamath.triclinic_vectors(self._universe.dimensions)
            )
            self.scattering_vector_factors = 2 * np.pi / self.box
            self.max_n = np.ceil(self.qmax / self.scattering_vector_factors).astype(int)

    def _single_frame(self) -> float:
        box = np.diag(mda.lib.mdamath.triclinic_vectors(self._ts.dimensions))

        if self.bin_spectrum:
            self._obs.structure_factors = np.zeros(self.n_bins)
            self._obs.scattering_intensities = np.zeros(self.n_bins)
        else:
            if not np.all(box == self.box):
                raise ValueError(
                    f"Dimensions in frame {self.frame_index} are different from "
                    "initial dimenions. Can not use `bin_spectrum=False`."
                )

            self._obs.structure_factors = np.zeros(self.max_n)
            self._obs.scattering_intensities = np.zeros(self.max_n)

        for i_group, group in enumerate(self.groups):
            # Map coordinates onto cubic cell
            positions = group.atoms.positions - box * np.round(
                group.atoms.positions / box
            )

            scattering_vectors, structure_factors = compute_structure_factor(
                np.double(positions),
                np.double(box),
                self.qmin,
                self.qmax,
                self.thetamin,
                self.thetamax,
                self.weights[i_group],
            )

            scattering_intensities = (
                compute_form_factor(scattering_vectors, self.atom_types[i_group]) ** 2
                * structure_factors
            )

            if self.bin_spectrum:
                scattering_vectors = scattering_vectors.flatten()
                structure_factors = structure_factors.flatten()
                scattering_intensities = scattering_intensities.flatten()

                nonzeros = np.where(structure_factors != 0)[0]
                scattering_vectors = scattering_vectors[nonzeros]
                structure_factors = structure_factors[nonzeros]
                scattering_intensities = scattering_intensities[nonzeros]

                histogram_kwargs = dict(
                    a=scattering_vectors,
                    bins=self.n_bins,
                    range=(self.qmin, self.qmax),
                )
                structure_factors_binned, _ = np.histogram(
                    weights=structure_factors, **histogram_kwargs
                )
                scattering_intensities_binned, _ = np.histogram(
                    weights=scattering_intensities, **histogram_kwargs
                )
                bincount, _ = np.histogram(weights=None, **histogram_kwargs)
                with np.errstate(invalid="ignore"):
                    structure_factors_binned /= bincount
                    scattering_intensities_binned /= bincount

                self._obs.structure_factors += np.nan_to_num(structure_factors_binned)
                self._obs.scattering_intensities += np.nan_to_num(
                    scattering_intensities_binned
                )
            else:
                self._obs.structure_factors += structure_factors
                self._obs.scattering_intensities += scattering_intensities

        return structure_factors.flatten()[-1]

    def _conclude(self) -> None:
        if self.bin_spectrum:
            scattering_vectors = (
                np.arange(self.qmin, self.qmax, self.dq) + 0.5 * self.dq
            )

            structure_factors = self.means.structure_factors
            scattering_intensities = self.means.scattering_intensities

        else:
            miller_indices = np.array(list(np.ndindex(tuple(self.max_n))))
            scattering_vectors = np.linalg.norm(
                miller_indices * self.scattering_vector_factors[np.newaxis, :], axis=1
            )

            structure_factors = self.means.structure_factors
            scattering_intensities = self.means.scattering_intensities

            # Flatten results to have same shape as scattering vectors and
            # miller_indices
            structure_factors = structure_factors.flatten()
            scattering_intensities = scattering_intensities.flatten()

            # sort results according to scattering_vectors
            argsort = np.argsort(scattering_vectors)
            scattering_vectors = scattering_vectors[argsort]
            miller_indices = miller_indices[argsort]
            structure_factors = structure_factors[argsort]
            scattering_intensities = scattering_intensities[argsort]

        # remove zeros
        nonzeros = np.where(structure_factors != 0)[0]
        scattering_vectors = scattering_vectors[nonzeros]
        structure_factors = structure_factors[nonzeros]
        scattering_intensities = scattering_intensities[nonzeros]

        # normalize
        structure_factors /= self.atomgroup.n_atoms
        scattering_intensities /= self.atomgroup.n_atoms

        self.results.scattering_vectors = scattering_vectors
        self.results.structure_factors = structure_factors
        self.results.scattering_intensities = scattering_intensities

        if not self.bin_spectrum:
            self.results.miller_indices = miller_indices[nonzeros]

    @render_docs
    def save(self) -> None:
        """${SAVE_METHOD_DESCRIPTION}"""
        if self.bin_spectrum:
            self.savetxt(
                self.output,
                np.vstack(
                    [
                        self.results.scattering_vectors,
                        self.results.structure_factors,
                        self.results.scattering_intensities,
                    ]
                ).T,
                columns=["q (1/Å)", "S(q) (arb. units)", "I(q) (arb. units)"],
            )
        else:
            out = np.hstack(
                [
                    self.results.scattering_vectors[:, np.newaxis],
                    self.results.miller_indices,
                    self.results.structure_factors[:, np.newaxis],
                    self.results.scattering_intensities[:, np.newaxis],
                ]
            )

            boxinfo = (
                "box_x = {0:.3f} Å, box_y = {1:.3f} Å, "
                "box_z = {2:.3f} Å\n".format(*self.box)
            )
            self.savetxt(
                self.output,
                out,
                columns=[
                    boxinfo,
                    "q (1/Å)",
                    "q_i",
                    "q_j",
                    "q_k",
                    "S(q) (arb. units)",
                    "I(q) (arb. units)",
                ],
            )
