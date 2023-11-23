#!/usr/bin/env python
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
#
# Copyright (c) 2023 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the GNU Public Licence, v3 or any higher version
# SPDX-License-Identifier: GPL-3.0-or-later
r"""Module for computing Small Angle X-Ray scattering intensities."""

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

    By default the scattering vectors :math:`\boldsymbol{q}` are binned acording to
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

    For an examples on the usage refer to :ref:`How-to: SAXS<howto-saxs>` and for
    details on the theory see :ref:`saxs-explanations`.

    Parameters
    ----------
    ${ATOMGROUP_PARAMETER}
    ${BASE_CLASS_PARAMETERS}
    bin_spectrum : bool
        Bin the spectrum. If :py:obj:`False` Miller indices of q-vector are returned.
        Only works for NVT simulations.
    startq : float
        Starting q (1/Å)
    endq : float
        Ending q (1/Å)
    dq : float
        bin_width (1/Å)
    mintheta : float
        Minimal angle (°) between the q vectors and the z-axis.
    maxtheta : float
        Maximal angle (°) between the q vectors and the z-axis.
    ${OUTPUT_PARAMETER}

    Attributes
    ----------
    results.q : numpy.ndarray
        length of binned q-vectors
    results.q_indices : numpy.ndarray
        Miller indices of q-vector (only if ``bin_spectrum==False``)
    results.scat_factor : numpy.ndarray
        Scattering intensities

    References
    ----------
    .. footbibliography::
    """

    def __init__(
        self,
        atomgroup: mda.AtomGroup,
        unwrap: bool = False,
        refgroup: Optional[mda.AtomGroup] = None,
        jitter: float = 0.0,
        concfreq: int = 0,
        bin_spectrum: bool = True,
        startq: float = 0,
        endq: float = 6,
        dq: float = 0.1,
        mintheta: float = 0,
        maxtheta: float = 180,
        output: str = "sq.dat",
    ):
        self._locals = locals()
        super().__init__(
            atomgroup,
            unwrap=unwrap,
            refgroup=refgroup,
            jitter=jitter,
            concfreq=concfreq,
        )
        self.bin_spectrum = bin_spectrum
        self.startq = startq
        self.endq = endq
        self.dq = dq
        self.mintheta = mintheta
        self.maxtheta = maxtheta
        self.output = output

    def _prepare(self):
        self.mintheta = min(self.mintheta, self.maxtheta)
        self.maxtheta = max(self.mintheta, self.maxtheta)

        if self.mintheta < 0 or self.mintheta > 180:
            raise ValueError(f"mintheta ({self.mintheta}°) has to between 0 and 180°.")

        if self.maxtheta < 0 or self.maxtheta > 180:
            raise ValueError(f"maxtheta ({self.maxtheta}°) has to between 0 and 180°.")

        if self.mintheta > self.maxtheta:
            raise ValueError(
                f"mintheta ({self.mintheta}°) larger than maxtheta ({self.maxtheta}°)."
            )

        # Convert angles from degrees to radians
        self.mintheta *= np.pi / 180
        self.maxtheta *= np.pi / 180

        self.groups = []
        self.weights = []
        self.atom_types = []

        logger.info("\nMap the following atomtypes:")
        for atom_type in np.unique(self.atomgroup.types).astype(str):
            try:
                element = tables.atomtypes[atom_type]
            except KeyError:
                raise RuntimeError(
                    f"No suitable element for {atom_type!r} found. You can add "
                    "{atom_type!r} together with a suitable element to "
                    "'share/atomtypes.dat'."
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
            self.n_bins = int(np.ceil((self.endq - self.startq) / self.dq))
        else:
            self.box = np.diag(
                mda.lib.mdamath.triclinic_vectors(self._universe.dimensions)
            )
            self.q_factor = 2 * np.pi / self.box
            self.maxn = np.ceil(self.endq / self.q_factor).astype(int)

    def _single_frame(self):
        box = np.diag(mda.lib.mdamath.triclinic_vectors(self._ts.dimensions))

        if self.bin_spectrum:
            self._obs.struct_factor = np.zeros([self.n_bins, len(self.groups)])
        else:
            if not np.all(box == self.box):
                raise ValueError(
                    f"Dimensions in frame {self.frame_index} are different from "
                    "initial dimenions. Can not use `bin_spectrum=False`."
                )

            self._obs.S_array = np.zeros(list(self.maxn) + [len(self.groups)])

        for i_group, group in enumerate(self.groups):
            # Map coordinates onto cubic cell
            positions = group.atoms.positions - box * np.round(
                group.atoms.positions / box
            )

            q_ts, S_ts = compute_structure_factor(
                np.double(positions),
                np.double(box),
                self.startq,
                self.endq,
                self.mintheta,
                self.maxtheta,
                self.weights[i_group],
            )

            S_ts *= compute_form_factor(q_ts, self.atom_types[i_group]) ** 2

            if self.bin_spectrum:
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

                self._obs.struct_factor[:, i_group] = np.nan_to_num(struct_ts)
            else:
                self._obs.S_array[:, :, :, i_group] = S_ts

    def _conclude(self):
        if self.bin_spectrum:
            q = np.arange(self.startq, self.endq, self.dq) + 0.5 * self.dq
            nonzeros = np.where(self.means.struct_factor[:, 0] != 0)[0]
            scat_factor = self.means.struct_factor[nonzeros]

            self.results.q = q[nonzeros]
            self.results.scat_factor = scat_factor.sum(axis=1)
        else:
            self.results.scat_factor = self.means.S_array.sum(axis=3)
            self.results.q_indices = np.array(list(np.ndindex(tuple(self.maxn))))
            self.results.q = np.linalg.norm(
                self.results.q_indices * self.q_factor[np.newaxis, :], axis=1
            )

        self.results.scat_factor /= self.atomgroup.n_atoms

    @render_docs
    def save(self):
        """${SAVE_DESCRIPTION}"""
        if self.bin_spectrum:
            self.savetxt(
                self.output,
                np.vstack([self.results.q, self.results.scat_factor]).T,
                columns=["q (1/Å)", "S(q) (arb. units)"],
            )
        else:
            out = np.hstack(
                [
                    self.results.q[:, np.newaxis],
                    self.results.q_indices,
                    self.results.scat_factor.flatten()[:, np.newaxis],
                ]
            )
            nonzeros = np.where(out[:, 4] != 0)[0]
            out = out[nonzeros]
            argsort = np.argsort(out[:, 0])
            out = out[argsort]

            boxinfo = (
                "box_x = {0:.3f} Å, box_y = {1:.3f} Å, "
                "box_z = {2:.3f} Å\n".format(*self.box)
            )
            self.savetxt(
                self.output,
                out,
                columns=[boxinfo, "q (1/Å)", "q_i", "q_j", "q_k", "S(q) (arb. units)"],
            )
