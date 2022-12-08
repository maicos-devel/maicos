#!/usr/bin/env python3
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
#
# Copyright (c) 2022 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the GNU Public Licence, v3 or any higher version
# SPDX-License-Identifier: GPL-3.0-or-later
"""Module for computing spherical dielectric profiles."""

import logging

import numpy as np
import scipy.constants

from ..core import SphereBase
from ..lib.util import charge_neutral, get_compound, render_docs


logger = logging.getLogger(__name__)


@render_docs
@charge_neutral(filter="error")
class DielectricSphere(SphereBase):
    r"""Calculate spherical dielectric profiles.

    Components are calculated along and radial direction at the
    system's center of mass.

    Parameters
    ----------
    ${ATOMGROUP_PARAMETER}
    ${SPHERE_CLASS_PARAMETERS}
    temperature : float
        temperature (K)
    output_prefix : str
        Prefix for output_prefix files

    Attributes
    ----------
    ${RADIAL_CLASS_ATTRIBUTES}
    results.eps_rad : numpy.ndarray
        Reduced inverse radial dielectric profile
        (:math:`\varepsilon^{-1}_r - 1)`
    results.deps_rad : numpy.ndarray
        Uncertainty of inverse radial dielectric profile
    """

    def __init__(self,
                 atomgroup,
                 bin_width=0.1,
                 temperature=300,
                 output_prefix="eps_sph",
                 refgroup=None,
                 concfreq=0,
                 rmin=0,
                 rmax=None,
                 unwrap=True):
        super(DielectricSphere, self).__init__(atomgroup,
                                               concfreq=concfreq,
                                               refgroup=refgroup,
                                               rmin=rmin,
                                               rmax=rmax,
                                               bin_width=bin_width,
                                               unwrap=unwrap)
        self.output_prefix = output_prefix
        self.bin_width = bin_width
        self.temperature = temperature

    def _prepare(self):
        super(DielectricSphere, self)._prepare()
        self.comp, ix = get_compound(self.atomgroup.atoms, return_index=True)
        _, self.inverse_ix = np.unique(ix, return_inverse=True)

    def _single_frame(self):
        super(DielectricSphere, self)._single_frame()
        rbins = np.digitize(
            self.transform_positions(self.atomgroup.positions)[:, 0],
            self._obs.bin_edges[1:])

        curQ_rad, _ = np.histogram(rbins,
                                   bins=np.arange(self.n_bins + 1),
                                   weights=self.atomgroup.charges)

        self._obs.m_rad = -np.cumsum(
            (curQ_rad / self._obs.bin_volume) * self._obs.bin_pos**2
            * self._obs.bin_width) / self._obs.bin_pos**2
        self._obs.M_rad = np.sum(self._obs.m_rad * self._obs.bin_width)
        self._obs.mM_rad = self._obs.m_rad * self._obs.M_rad

    def _conclude(self):
        super(DielectricSphere, self)._conclude()

        pref = 1 / scipy.constants.epsilon_0
        pref /= scipy.constants.Boltzmann * self.temperature
        # Convert from ~e^2/m to ~base units
        pref /= scipy.constants.angstrom / \
            (scipy.constants.elementary_charge)**2

        cov_rad = self.means.mM_rad - self.means.m_rad * self.means.M_rad

        dcov_rad = 0.5 * np.sqrt(
            self.sems.mM_rad**2 + self.sems.m_rad**2 * self.means.M_rad**2
            + self.means.m_rad**2 * self.sems.M_rad**2)

        self.results.eps_rad = 1 - (4 * np.pi * self.results.bin_pos**2
                                    * pref * cov_rad)
        self.results.deps_rad = (4 * np.pi * self.results.bin_pos**2
                                 * pref * dcov_rad)

    def save(self):
        """Save result."""
        outdata_rad = np.array([
            self.results.bin_pos, self.results.eps_rad, self.results.deps_rad
            ]).T

        columns = ["positions [Å]", "eps_rad - 1", "eps_rad error"]

        self.savetxt("{}{}".format(self.output_prefix, "_rad.dat"),
                     outdata_rad,
                     columns=columns)
