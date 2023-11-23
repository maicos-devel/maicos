#!/usr/bin/env python
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
#
# Copyright (c) 2023 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the GNU Public Licence, v3 or any higher version
# SPDX-License-Identifier: GPL-3.0-or-later
"""Module for calculating spherical dielectric profiles."""

import logging
from typing import Optional

import MDAnalysis as mda
import numpy as np
import scipy.constants

from ..core import SphereBase
from ..lib.util import charge_neutral, citation_reminder, get_compound, render_docs


logger = logging.getLogger(__name__)


@render_docs
@charge_neutral(filter="error")
class DielectricSphere(SphereBase):
    r"""Spherical dielectric profiles.

    Components are calculated along the radial (:math:`r`) direction either with respect
    to the center of the simulation box or the center of mass of the refgroup, if
    provided.

    For usage, please refer to :ref:`How-to: Dielectric
    constant<howto-dielectric>` and for details on the theory see
    :ref:`dielectric-explanations`.

    For correlation analysis, the radial (:math:`r`) component is used.
    ${CORRELATION_INFO}

    Also, please read and cite :footcite:p:`schaafDielectricResponseWater2015`.

    Parameters
    ----------
    ${ATOMGROUP_PARAMETER}
    ${SPHERE_CLASS_PARAMETERS}
    ${TEMPERATURE_PARAMETER}
    ${OUTPUT_PREFIX_PARAMETER}

    Attributes
    ----------
    ${RADIAL_CLASS_ATTRIBUTES}
    results.eps_rad : numpy.ndarray
        Reduced inverse radial dielectric profile (:math:`\varepsilon^{-1}_r - 1)`
    results.deps_rad : numpy.ndarray
        Uncertainty of inverse radial dielectric profile

    References
    ----------
    .. footbibliography::
    """

    def __init__(
        self,
        atomgroup: mda.AtomGroup,
        bin_width: float = 0.1,
        temperature: float = 300,
        output_prefix: str = "eps_sph",
        refgroup: Optional[mda.AtomGroup] = None,
        concfreq: int = 0,
        jitter: float = 0.0,
        rmin: float = 0,
        rmax: Optional[float] = None,
        unwrap: bool = True,
    ):
        self._locals = locals()
        self.comp = get_compound(atomgroup.atoms)
        ix = atomgroup._get_compound_indices(self.comp)
        _, self.inverse_ix = np.unique(ix, return_inverse=True)
        if rmin != 0 or rmax is not None:
            logger.warning(
                "Setting `rmin` and `rmax` might cut off molecules. This will lead to "
                "severe artifacts in the dielectric profiles."
            )

        super().__init__(
            atomgroup,
            concfreq=concfreq,
            jitter=jitter,
            refgroup=refgroup,
            rmin=rmin,
            rmax=rmax,
            bin_width=bin_width,
            unwrap=unwrap,
            wrap_compound=self.comp,
        )
        self.output_prefix = output_prefix
        self.bin_width = bin_width
        self.temperature = temperature

    def _prepare(self):
        # Print the Christian Schaaf citation
        logger.info(citation_reminder("10.1103/PhysRevE.92.032718"))

        super()._prepare()

    def _single_frame(self) -> float:
        super()._single_frame()
        rbins = np.digitize(self.pos_sph[:, 0], self._obs.bin_edges[1:-1])
        curQ_rad = np.bincount(
            rbins[self.atomgroup.ix],
            weights=self.atomgroup.charges,
            minlength=self.n_bins,
        )

        self._obs.m_r = (
            -np.cumsum(
                (curQ_rad / self._obs.bin_volume)
                * self._obs.bin_pos**2
                * self._obs.bin_width
            )
            / self._obs.bin_pos**2
        )

        curQ_rad_tot = np.bincount(
            rbins, weights=self._universe.atoms.charges, minlength=self.n_bins
        )

        self._obs.m_r_tot = (
            -np.cumsum(
                (curQ_rad_tot / self._obs.bin_volume)
                * self._obs.bin_pos**2
                * self._obs.bin_width
            )
            / self._obs.bin_pos**2
        )

        # This is not really the systems dipole moment, but it keeps the Nomenclature
        # consistent with the DielectricPlanar module.
        self._obs.M_r = np.sum(self._obs.m_r_tot * self._obs.bin_width)
        self._obs.mM_r = self._obs.m_r * self._obs.M_r

        return self._obs.M_r

    def _conclude(self):
        super()._conclude()

        pref = 1 / scipy.constants.epsilon_0
        pref /= scipy.constants.Boltzmann * self.temperature
        # Convert from ~e^2/m to ~base units
        pref /= scipy.constants.angstrom / (scipy.constants.elementary_charge) ** 2

        cov_rad = self.means.mM_r - self.means.m_r * self.means.M_r

        dcov_rad = 0.5 * np.sqrt(
            self.sems.mM_r**2
            + self.sems.m_r**2 * self.means.M_r**2
            + self.means.m_r**2 * self.sems.M_r**2
        )

        self.results.eps_rad = 1 - (
            4 * np.pi * self.results.bin_pos**2 * pref * cov_rad
        )
        self.results.deps_rad = 4 * np.pi * self.results.bin_pos**2 * pref * dcov_rad

    @render_docs
    def save(self):
        """${SAVE_DESCRIPTION}"""
        outdata_rad = np.array(
            [self.results.bin_pos, self.results.eps_rad, self.results.deps_rad]
        ).T

        columns = ["positions [Å]", "eps_rad - 1", "eps_rad error"]

        self.savetxt(
            "{}{}".format(self.output_prefix, "_rad.dat"), outdata_rad, columns=columns
        )
