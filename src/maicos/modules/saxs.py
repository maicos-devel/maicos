#!/usr/bin/env python3
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
#
# Copyright (c) 2022 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the GNU Public Licence, v3 or any higher version
# SPDX-License-Identifier: GPL-3.0-or-later
r"""Small Angle X-Ray scattering intensities."""

import logging

import MDAnalysis as mda
import numpy as np

from ..core import AnalysisBase
from ..lib import tables
from ..lib.math import compute_form_factor, compute_structure_factor
from ..lib.util import render_docs


logger = logging.getLogger(__name__)


@render_docs
class Saxs(AnalysisBase):
    """Compute small angle X-Ray scattering intensities (SAXS).

    The q vectors are binned by their length using a bin_width given by -dq.
    Using the -nobin option the raw intensity for each q_{i,j,k} vector
    is saved using. Note that this only works reliable using constant
    box vectors! The possible scattering vectors q can be restricted by a
    miminal and maximal angle with the z-axis. For 0 and 180 all possible
    vectors are taken into account. For the scattering factor the structure
    factor is multiplied by a atom type specific form factor based on
    Cromer-Mann parameters. By using the -sel option atoms can be selected
    for which the profile is calculated. The selection uses the
    MDAnalysis selection commands.

    Parameters
    ----------
    ${ATOMGROUP_PARAMETER}
    ${BASE_CLASS_PARAMETERS}
    noboindata : bool
        Do not bin the data. Only works reliable for NVT!
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
    output : str
        Output filename

    Attributes
    ----------
    results.q : numpy.ndarray
        length of binned q-vectors
    results.q_indices : numpy.ndarray
        Miller indices of q-vector (only if noboindata==True)
    results.scat_factor : numpy.ndarray
        Scattering intensities
    """

    def __init__(self,
                 atomgroup,
                 refgroup=None,
                 unwrap=False,
                 jitter=0.0,
                 concfreq=0,
                 nobin=False,
                 startq=0,
                 endq=6,
                 dq=0.005,
                 mintheta=0,
                 maxtheta=180,
                 output="sq.dat",):
        super(Saxs, self).__init__(atomgroup,
                                   unwrap=unwrap,
                                   refgroup=refgroup,
                                   jitter=jitter,
                                   concfreq=concfreq)
        self.nobindata = nobin
        self.startq = startq
        self.endq = endq
        self.dq = dq
        self.mintheta = mintheta
        self.maxtheta = maxtheta
        self.output = output

    def _prepare(self):

        self.mintheta = min(self.mintheta, self.maxtheta)
        self.maxtheta = max(self.mintheta, self.maxtheta)

        if self.mintheta < 0:
            logger.info(f"mintheta = {self.mintheta}° < 0°: "
                        "Set mininmal angle to 0°.")
            self.mintheta = 0
        if self.maxtheta > 180:
            logger.info(f"maxtheta = {self.maxtheta}° > 180°: "
                        "Set maximal angle to 180°.")
            self.maxtheta = np.pi

        self.mintheta *= np.pi / 180
        self.maxtheta *= np.pi / 180

        self.groups = []
        self.atom_types = []
        logger.info("\nMap the following atomtypes:")
        for atom_type in np.unique(self.atomgroup.types).astype(str):
            try:
                element = tables.atomtypes[atom_type]
            except KeyError:
                raise RuntimeError(f"No suitable element for {atom_type!r} "
                                   f"found. You can add {atom_type!r} "
                                   "together with a suitable element "
                                   "to 'share/atomtypes.dat'.")
            if element == "DUM":
                continue
            self.groups.append(
                self.atomgroup.select_atoms("type {}*".format(atom_type)))
            self.atom_types.append(atom_type)

            logger.info("{:>14} --> {:>5}".format(atom_type, element))

        if self.nobindata:
            self.box = np.diag(
                mda.lib.mdamath.triclinic_vectors(
                    self._universe.dimensions))
            self.q_factor = 2 * np.pi / self.box
            self.maxn = np.ceil(self.endq / self.q_factor).astype(int)
        else:
            self.n_bins = int(np.ceil((self.endq - self.startq) / self.dq))

    def _single_frame(self):
        # Convert everything to cartesian coordinates.
        if self.nobindata:
            self._obs.S_array = np.zeros(list(self.maxn) + [len(self.groups)])
        else:
            self._obs.struct_factor = np.zeros([self.n_bins, len(self.groups)])
        box = np.diag(mda.lib.mdamath.triclinic_vectors(self._ts.dimensions))
        for i, t in enumerate(self.groups):
            # map coordinates onto cubic cell
            positions = t.atoms.positions - box * \
                np.round(t.atoms.positions / box)
            q_ts, S_ts = compute_structure_factor(
                np.double(positions), np.double(box), self.startq,
                self.endq, self.mintheta, self.maxtheta)

            S_ts *= compute_form_factor(q_ts, self.atom_types[i])**2

            if self.nobindata:
                self._obs.S_array[:, :, :, i] = S_ts
            else:
                q_ts = q_ts.flatten()
                S_ts = S_ts.flatten()
                nonzeros = np.where(S_ts != 0)[0]

                q_ts = q_ts[nonzeros]
                S_ts = S_ts[nonzeros]

                struct_ts = np.histogram(q_ts,
                                         bins=self.n_bins,
                                         range=(self.startq, self.endq),
                                         weights=S_ts)[0]
                with np.errstate(divide='ignore', invalid='ignore'):
                    struct_ts /= np.histogram(q_ts,
                                              bins=self.n_bins,
                                              range=(self.startq, self.endq))[0]
                self._obs.struct_factor[:, i] = np.nan_to_num(struct_ts)

    def _conclude(self):
        if self.nobindata:
            self.results.scat_factor = self.means.S_array.sum(axis=3)
            self.results.q_indices = np.array(
                list(np.ndindex(tuple(self.maxn))))
            self.results.q = np.linalg.norm(self.results.q_indices
                                            * self.q_factor[np.newaxis, :],
                                            axis=1)
        else:
            q = np.arange(self.startq, self.endq, self.dq) + 0.5 * self.dq
            nonzeros = np.where(self.means.struct_factor[:, 0] != 0)[0]
            scat_factor = self.means.struct_factor[nonzeros]

            self.results.q = q[nonzeros]
            self.results.scat_factor = scat_factor.sum(axis=1)

        self.results.scat_factor /= (self.atomgroup.n_atoms)

    def save(self):
        """Save the current profiles to a file."""
        if self.nobindata:
            out = np.hstack([
                self.results.q[:, np.newaxis], self.results.q_indices,
                self.results.scat_factor.flatten()[:, np.newaxis]])
            nonzeros = np.where(out[:, 4] != 0)[0]
            out = out[nonzeros]
            argsort = np.argsort(out[:, 0])
            out = out[argsort]

            boxinfo = "box_x = {0:.3f} Å, box_y = {1:.3f} Å, " \
                      "box_z = {2:.3f} Å\n".format(*self.box)
            self.savetxt(self.output, out,
                         columns=[boxinfo, "q (1/Å)", "q_i", "q_j",
                                  "q_k", "S(q) (arb. units)"])
        else:
            self.savetxt(self.output,
                         np.vstack([self.results.q,
                                    self.results.scat_factor]).T,
                         columns=["q (1/Å)", "S(q) (arb. units)"])
