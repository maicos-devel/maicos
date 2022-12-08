#!/usr/bin/env python3
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
#
# Copyright (c) 2022 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the GNU Public Licence, v3 or any higher version
# SPDX-License-Identifier: GPL-3.0-or-later
"""Module for computing cylindrical dielectric profile."""

import logging

import numpy as np
import scipy.constants

from ..core import CylinderBase
from ..lib.util import charge_neutral, get_compound, render_docs


logger = logging.getLogger(__name__)


@render_docs
@charge_neutral(filter="error")
class DielectricCylinder(CylinderBase):
    r"""Calculate cylindrical dielectric profiles.

    Components are calculated along the axial (z) and radial (along xy)
    direction at the system's center of mass.

    Parameters
    ----------
    ${ATOMGROUP_PARAMETER}
    ${CYLINDER_CLASS_PARAMETERS}
    atomgroup : AtomGroup
        :class:`~MDAnalysis.core.groups.AtomGroup` for which
        the dielectric profiles are calculated
    geometry : str
        A structure file without water from which com is calculated.
    radius : float
        Radius of the cylinder (Å)
    bin_width : float
        Bindiwdth the bin_width (Å)
    variable_dr : bool
        Use a variable bin_width, where the volume is kept fixed.
    length : float
        Length of the cylinder (Å)
    temperature : float
        temperature (K)
    single : bool
        For a single chain of molecules the average of M is zero. This flag sets
        <M> = 0.
    temperature : float
        temperature (K)
    output_prefix : str
        Prefix for output_prefix files

    Attributes
    ----------
    ${CYLINDER_CLASS_ATTRIBUTES}
    results.eps_ax : numpy.ndarray
        Reduced axial dielectric profile :math:`(\varepsilon_z - 1)` of the
        selected atomgroup
    results.deps_ax : numpy.ndarray
        Uncertainty of axial dielectric profile
    results.eps_rad : numpy.ndarray
        Reduced inverse radial dielectric profile
        :math:`(\varepsilon^{-1}_\rho - 1)`
    results.deps_rad : numpy.ndarray
        Uncertainty of inverse radial dielectric profile
    """

    def __init__(self,
                 atomgroup,
                 bin_width=0.1,
                 temperature=300,
                 single=False,
                 output_prefix="eps_cyl",
                 refgroup=None,
                 concfreq=0,
                 dim=2,
                 rmin=0,
                 rmax=None,
                 zmin=None,
                 zmax=None,
                 vcutwidth=0.1,
                 unwrap=True):
        super(DielectricCylinder, self).__init__(atomgroup,
                                                 concfreq=concfreq,
                                                 refgroup=refgroup,
                                                 rmin=rmin,
                                                 rmax=rmax,
                                                 zmin=zmin,
                                                 zmax=zmax,
                                                 dim=dim,
                                                 bin_width=bin_width,
                                                 unwrap=unwrap)
        self.output_prefix = output_prefix
        self.temperature = temperature
        self.single = single
        self.vcutwidth = vcutwidth

    def _prepare(self):
        super(DielectricCylinder, self)._prepare()
        self.comp, ix = get_compound(self.atomgroup.atoms, return_index=True)
        _, self.inverse_ix = np.unique(ix, return_inverse=True)

    def _single_frame(self):
        super(DielectricCylinder, self)._single_frame()

        # Use polarization density (for radial component)
        # ========================================================
        rbins = np.digitize(
            self.transform_positions(self.atomgroup.positions)[:, 0],
            self._obs.bin_edges[1:])

        curQ_rad, _ = np.histogram(rbins,
                                   bins=np.arange(self.n_bins + 1),
                                   weights=self.atomgroup.charges)

        self._obs.m_rad = -np.cumsum(
            (curQ_rad / self._obs.bin_volume) * self._obs.bin_pos
            * self._obs.bin_width) / self._obs.bin_pos
        self._obs.M_rad = np.sum(self._obs.m_rad * self._obs.bin_width)
        self._obs.mM_rad = self._obs.m_rad * self._obs.M_rad
        # Use virtual cutting method ( for axial component )
        # ========================================================
        # number of virtual cuts ("many")
        nbinsz = np.ceil(self._obs.L / self.vcutwidth).astype(int)

        # Move all r-positions to 'center of charge' such that we avoid
        # monopoles in r-direction. We only want to cut in z direction.
        chargepos = self.pos_cyl[:, 0] * np.abs(self.atomgroup.charges)
        center = (self.atomgroup.accumulate(chargepos, compound=self.comp)
                  / self.atomgroup.accumulate(np.abs(self.atomgroup.charges),
                                              compound=self.comp))
        testpos = center[self.inverse_ix]
        rbins = np.digitize(testpos, self._obs.bin_edges[1:])
        z = (np.arange(nbinsz) + 1) * (self._obs.L / nbinsz)
        zbins = np.digitize(self.pos_cyl[:, 2], z)

        curQz, _, _ = np.histogram2d(
            zbins, rbins,
            bins=[np.arange(nbinsz + 1), np.arange(self.n_bins + 1)],
            weights=self.atomgroup.charges)

        curqz = np.cumsum(curQz, axis=0) / (self._obs.bin_area)[np.newaxis, :]
        self._obs.m_ax = -curqz.mean(axis=0)
        self._obs.M_ax = np.dot(self.atomgroup.charges, self.pos_cyl[:, 2])
        self._obs.mM_ax = self._obs.m_ax * self._obs.M_ax

    def _conclude(self):
        super(DielectricCylinder, self)._conclude()

        pref = 1 / scipy.constants.epsilon_0
        pref /= scipy.constants.Boltzmann * self.temperature
        # Convert from ~e^2/m to ~base units
        pref /= scipy.constants.angstrom / \
            (scipy.constants.elementary_charge)**2

        if not self.single:
            cov_ax = self.means.mM_ax - self.means.m_ax * self.means.M_ax
            cov_rad = self.means.mM_rad - self.means.m_rad * self.means.M_rad

            dcov_ax = 0.5 * np.sqrt(
                self.sems.mM_ax**2 + self.sems.m_ax**2 * self.means.M_ax**2
                + self.means.m_ax**2 * self.sems.M_ax**2)
            dcov_rad = 0.5 * np.sqrt(
                self.sems.mM_rad**2 + self.sems.m_rad**2 * self.means.M_rad**2
                + self.means.m_rad**2 * self.sems.M_rad**2)
        else:
            # <M> = 0 for a single line of water molecules.
            cov_ax = self.means.mM_ax
            cov_rad = self.means.mM_rad
            dcov_ax = self.sems.mM_ax
            dcov_rad = self.sems.mM_rad

        self.results.eps_ax = pref * cov_ax
        self.results.deps_ax = pref * dcov_ax

        self.results.eps_rad = - (2 * np.pi * self._obs.L
                                    * pref * self.results.bin_pos * cov_rad)
        self.results.deps_rad = (2 * np.pi * self._obs.L
                                 * pref * self.results.bin_pos * dcov_rad)

    def save(self):
        """Save result."""
        outdata_ax = np.array([
            self.results.bin_pos, self.results.eps_ax, self.results.deps_ax
            ]).T
        outdata_rad = np.array([
            self.results.bin_pos, self.results.eps_rad, self.results.deps_rad
            ]).T

        columns = ["positions [Å]"]

        columns += ["ε_z - 1", "Δε_z"]

        self.savetxt("{}{}".format(self.output_prefix, "_ax.dat"),
                     outdata_ax, columns=columns)

        columns = ["positions [Å]"]

        columns += ["ε^-1_r - 1", "Δε^-1_r"]

        self.savetxt("{}{}".format(self.output_prefix, "_rad.dat"),
                     outdata_rad, columns=columns)
