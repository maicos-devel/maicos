#!/usr/bin/env python
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
#
# Copyright (c) 2024 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the GNU Public Licence, v3 or any higher version
# SPDX-License-Identifier: GPL-3.0-or-later
"""Module for computing kinetic energy timeseries."""

from typing import Optional

import MDAnalysis as mda
import numpy as np

from ..core import AnalysisBase
from ..lib.util import get_compound, render_docs


@render_docs
class KineticEnergy(AnalysisBase):
    """Kinetic energy timeseries.

    The kinetic energy function computes the translational and rotational kinetic energy
    with respect to molecular center (center of mass, center of charge) of a molecular
    dynamics simulation trajectory.

    The analysis can be applied to study the dynamics of water molecules during an
    excitation pulse. For more details read
    :footcite:t:`elgabartyEnergyTransferHydrogen2020`.

    Parameters
    ----------
    ${ATOMGROUP_PARAMETER}
    ${BASE_CLASS_PARAMETERS}
    refpoint : str
        reference point for molecular center: center of mass (``"com"``) or center of
        charge (``"coc"``).
    ${OUTPUT_PARAMETER}

    Attributes
    ----------
    results.t : numpy.ndarray
        time (ps).
    results.trans : numpy.ndarray
        translational kinetic energy (kJ/mol).
    results.rot : numpy.ndarray
        rotational kinetic energy (kJ/mol).

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
        output: str = "ke.dat",
        refpoint: str = "COM",
    ):
        self._locals = locals()
        super().__init__(
            atomgroup,
            unwrap=unwrap,
            refgroup=refgroup,
            jitter=jitter,
            concfreq=concfreq,
        )
        self.output = output
        self.refpoint = refpoint.lower()

    def _prepare(self):
        """Set things up before the analysis loop begins."""
        if self.refpoint not in ["com", "coc"]:
            raise ValueError(
                f"Invalid choice for dens: {self.refpoint} (choose from 'com' or 'coc')"
            )

        self.masses = self.atomgroup.atoms.accumulate(
            self.atomgroup.atoms.masses, compound=get_compound(self.atomgroup)
        )
        self.abscharges = self.atomgroup.atoms.accumulate(
            np.abs(self.atomgroup.atoms.charges), compound=get_compound(self.atomgroup)
        )
        # Total kinetic energy
        self.E_kin = np.zeros(self.n_frames)

        # Molecular center energy
        self.E_center = np.zeros(self.n_frames)

    def _single_frame(self):
        self.E_kin[self._frame_index] = np.dot(
            self.atomgroup.masses,
            np.linalg.norm(self.atomgroup.velocities, axis=1) ** 2,
        )

        if self.refpoint == "com":
            massvel = self.atomgroup.velocities * self.atomgroup.masses[:, np.newaxis]
            v = self.atomgroup.accumulate(
                massvel, compound=get_compound(self.atomgroup)
            )
            v /= self.masses[:, np.newaxis]

        elif self.refpoint == "coc":
            abschargevel = (
                self.atomgroup.velocities
                * np.abs(self.atomgroup.charges)[:, np.newaxis]
            )
            v = self.atomgroup.accumulate(
                abschargevel, compound=get_compound(self.atomgroup)
            )
            v /= self.abscharges[:, np.newaxis]

        self.E_center[self._frame_index] = np.dot(
            self.masses, np.linalg.norm(v, axis=1) ** 2
        )

    def _conclude(self):
        self.results.t = self.times
        self.results.trans = self.E_center / 2 / 100
        self.results.rot = (self.E_kin - self.E_center) / 2 / 100

    @render_docs
    def save(self) -> None:
        """${SAVE_METHOD_DESCRIPTION}"""
        self.savetxt(
            self.output,
            np.vstack([self.results.t, self.results.trans, self.results.rot]).T,
            columns=["t", "E_kin^trans [kJ/mol]", "E_kin^rot [kJ/mol]"],
        )
