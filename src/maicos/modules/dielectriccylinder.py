#!/usr/bin/env python3
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
#
# Copyright (c) 2022 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the GNU Public Licence, v3 or any higher version
# SPDX-License-Identifier: GPL-3.0-or-later
"""Module for computing cylindrical dielectric profiles."""

import logging

import numpy as np
import scipy.constants

from ..core import CylinderBase
from ..lib.util import (
    charge_neutral,
    citation_reminder,
    get_compound,
    render_docs,
    )


logger = logging.getLogger(__name__)


@render_docs
@charge_neutral(filter="error")
class DielectricCylinder(CylinderBase):
    r"""Cylindrical dielectric profiles.

    Components are calculated along the axial (:math:`z`) and radial (:math:`r`)
    direction either with respect to the center of the simulation box or the
    center of mass of the refgroup, if provided. The axial direction is selected
    using the ``dim`` parameter.

    For usage please refer to :ref:`How-to: Dielectric
    constant<howto-dielectric>` and for details on the theory see
    :ref:`dielectric-explanations`.

    Also, please read and cite :footcite:p:`locheGiantaxialDielectric2019`.

    Parameters
    ----------
    ${ATOMGROUP_PARAMETER}
    ${CYLINDER_CLASS_PARAMETERS}
    temperature : float
        temperature (K)
    single : bool
        For a single chain of molecules the average of M is zero. This flag sets
        <M> = 0.

    Attributes
    ----------
    ${CYLINDER_CLASS_ATTRIBUTES}
    results.eps_z : numpy.ndarray
        Reduced axial dielectric profile :math:`(\varepsilon_z - 1)` of the
        selected atomgroup
    results.deps_z : numpy.ndarray
        Estimated uncertainty of axial dielectric profile
    results.eps_r : numpy.ndarray
        Reduced inverse radial dielectric profile
        :math:`(\varepsilon^{-1}_r - 1)`
    results.deps_r : numpy.ndarray
        Estimated uncertainty of inverse radial dielectric profile

    References
    ----------
    .. footbibliography::
    """

    def __init__(self,
                 atomgroup,
                 bin_width=0.1,
                 temperature=300,
                 single=False,
                 output_prefix="eps_cyl",
                 refgroup=None,
                 concfreq=0,
                 jitter=0.0,
                 dim=2,
                 rmin=0,
                 rmax=None,
                 zmin=None,
                 zmax=None,
                 vcutwidth=0.1,
                 unwrap=True):
        self.comp, ix = get_compound(atomgroup.atoms, return_index=True)
        _, self.inverse_ix = np.unique(ix, return_inverse=True)
        super(DielectricCylinder, self).__init__(atomgroup,
                                                 concfreq=concfreq,
                                                 jitter=jitter,
                                                 refgroup=refgroup,
                                                 rmin=rmin,
                                                 rmax=rmax,
                                                 zmin=zmin,
                                                 zmax=zmax,
                                                 dim=dim,
                                                 bin_width=bin_width,
                                                 unwrap=unwrap,
                                                 wrap_compound=self.comp)
        self.output_prefix = output_prefix
        self.temperature = temperature
        self.single = single
        self.vcutwidth = vcutwidth

    def _prepare(self):
        super(DielectricCylinder, self)._prepare()

    def _single_frame(self):
        super(DielectricCylinder, self)._single_frame()

        # Use polarization density (for radial component)
        # ========================================================
        rbins = np.digitize(self.pos_cyl[:, 0],
                            self._obs.bin_edges[1:-1])

        curQ_r = np.bincount(rbins[self.atomgroup.ix],
                             weights=self.atomgroup.charges,
                             minlength=self.n_bins)

        self._obs.m_r = \
            -np.cumsum((curQ_r / self._obs.bin_volume)
                       * self._obs.bin_pos * self._obs.bin_width
                       ) / self._obs.bin_pos

        curQ_r_tot = np.bincount(rbins,
                                 weights=self._universe.atoms.charges,
                                 minlength=self.n_bins)

        self._obs.m_r_tot = \
            -np.cumsum((curQ_r_tot / self._obs.bin_volume)
                       * self._obs.bin_pos * self._obs.bin_width
                       ) / self._obs.bin_pos
        # This is not really the systems dipole moment, but it keeps the
        # Nomenclature consistent with the DielectricPlanar module.
        self._obs.M_r = np.sum(self._obs.m_r_tot * self._obs.bin_width)
        self._obs.mM_r = self._obs.m_r * self._obs.M_r
        # Use virtual cutting method ( for axial component )
        # ========================================================
        # number of virtual cuts ("many")
        nbinsz = np.ceil(self._obs.L / self.vcutwidth).astype(int)

        # Move all r-positions to 'center of charge' such that we avoid
        # monopoles in r-direction. We only want to cut in z direction.
        chargepos = (self.pos_cyl[self.atomgroup.ix, 0]
                     * np.abs(self.atomgroup.charges))
        center = (self.atomgroup.accumulate(chargepos, compound=self.comp)
                  / self.atomgroup.accumulate(np.abs(self.atomgroup.charges),
                                              compound=self.comp))
        testpos = center[self.inverse_ix]

        rbins = np.digitize(testpos, self._obs.bin_edges[1:-1])
        z = (np.arange(nbinsz)) * (self._obs.L / nbinsz)
        zbins = np.digitize(self.pos_cyl[self.atomgroup.ix, 2], z[1:])

        curQz = np.bincount(rbins + self.n_bins * zbins,
                            weights=self.atomgroup.charges,
                            minlength=self.n_bins * nbinsz
                            ).reshape(nbinsz, self.n_bins)

        curqz = np.cumsum(curQz, axis=0) / (self._obs.bin_area)[np.newaxis, :]
        self._obs.m_z = -curqz.mean(axis=0)
        self._obs.M_z = np.dot(self._universe.atoms.charges, self.pos_cyl[:, 2]
                               ) / (2 * np.pi * self._obs.L)
        self._obs.mM_z = self._obs.m_z * self._obs.M_z

    def _conclude(self):
        super(DielectricCylinder, self)._conclude()

        pref = 1 / scipy.constants.epsilon_0
        pref /= scipy.constants.Boltzmann * self.temperature
        # Convert from ~e^2/m to ~base units
        pref /= scipy.constants.angstrom / \
            (scipy.constants.elementary_charge)**2

        if not self.single:
            cov_z = self.means.mM_z - self.means.m_z * self.means.M_z
            cov_r = self.means.mM_r - self.means.m_r * self.means.M_r

            dcov_z = 0.5 * np.sqrt(
                self.sems.mM_z**2 + self.sems.m_z**2 * self.means.M_z**2
                + self.means.m_z**2 * self.sems.M_z**2)
            dcov_r = 0.5 * np.sqrt(
                self.sems.mM_r**2 + self.sems.m_r**2 * self.means.M_r**2
                + self.means.m_r**2 * self.sems.M_r**2)
        else:
            # <M> = 0 for a single line of water molecules.
            cov_z = self.means.mM_z
            cov_r = self.means.mM_r
            dcov_z = self.sems.mM_z
            dcov_r = self.sems.mM_r

        self.results.eps_z = pref * cov_z
        self.results.deps_z = pref * dcov_z

        self.results.eps_r = - (2 * np.pi * self._obs.L
                                * pref * self.results.bin_pos * cov_r)
        self.results.deps_r = (2 * np.pi * self._obs.L
                                 * pref * self.results.bin_pos * dcov_r)

        # Print Philip Loche citation
        logger.info(citation_reminder("10.1021/acs.jpcb.9b09269"))

    def save(self):
        """Save result."""
        outdata_z = np.array([
            self.results.bin_pos, self.results.eps_z, self.results.deps_z
            ]).T
        outdata_r = np.array([
            self.results.bin_pos, self.results.eps_r, self.results.deps_r
            ]).T

        columns = ["positions [Å]"]

        columns += ["ε_z - 1", "Δε_z"]

        self.savetxt("{}{}".format(self.output_prefix, "_z.dat"),
                     outdata_z, columns=columns)

        columns = ["positions [Å]"]

        columns += ["ε^-1_r - 1", "Δε^-1_r"]

        self.savetxt("{}{}".format(self.output_prefix, "_r.dat"),
                     outdata_r, columns=columns)
