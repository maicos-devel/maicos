#!/usr/bin/env python
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
#
# Copyright (c) 2024 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the GNU Public Licence, v3 or any higher version
# SPDX-License-Identifier: GPL-3.0-or-later
"""Module for computing planar dielectric profiles."""

import logging
from typing import List, Optional, Union

import MDAnalysis as mda
import numpy as np
import scipy.constants

from ..core import PlanarBase
from ..lib.math import symmetrize
from ..lib.util import charge_neutral, citation_reminder, get_compound, render_docs


logger = logging.getLogger(__name__)


@render_docs
@charge_neutral(filter="error")
class DielectricPlanar(PlanarBase):
    r"""Planar dielectric profiles.

    For usage please refer to :ref:`How-to: Dielectric constant<howto-dielectric>` and
    for details on the theory see :ref:`dielectric-explanations`.

    For correlation analysis, the norm of the parallel total dipole moment is used.
    ${CORRELATION_INFO}

    Also, please read and cite
    :footcite:t:`schlaichWaterDielectricEffects2016` and Refs.
    :footcite:p:`locheUniversalNonuniversalAspects2020`,
    :footcite:p:`bonthuisProfileStaticPermittivity2012`.

    Parameters
    ----------
    ${ATOMGROUPS_PARAMETER}
    ${PLANAR_CLASS_PARAMETERS}
    is_3d : bool
        Use 3d-periodic boundary conditions, i.e., include the dipole correction for
        the interaction between periodic images
        :footcite:p:`sternCalculationDielectricPermittivity2003`.
    ${SYM_PARAMETER}
    ${TEMPERATURE_PARAMETER}
    ${OUTPUT_PREFIX_PARAMETER}
    vcutwidth : float
        Spacing of virtual cuts (bins) along the parallel directions.

    Attributes
    ----------
    ${PLANAR_CLASS_ATTRIBUTES}
    results.eps_par : numpy.ndarray
        Reduced parallel dielectric profile
        :math:`(\varepsilon_\parallel - 1)` of the selected atomgroups
    results.deps_par : numpy.ndarray
        Uncertainty of parallel dielectric profile
    results.eps_par_self : numpy.ndarray
        Reduced self contribution of parallel dielectric profile
        :math:`(\varepsilon_{\parallel,\mathrm{self}} - 1)`
    results.eps_par_coll : numpy.ndarray
        Reduced collective contribution of parallel dielectric profile
        :math:`(\varepsilon_{\parallel,\mathrm{coll}} - 1)`
    results.eps_perp : numpy.ndarray
        Reduced inverse perpendicular dielectric profile
        :math:`(\varepsilon^{-1}_\perp - 1)`
    results.deps_perp : numpy.ndarray
        Uncertainty of inverse perpendicular dielectric profile
    results.eps_perp_self : numpy.ndarray
        Reduced self contribution of the inverse perpendicular dielectric
        profile :math:`(\varepsilon^{-1}_{\perp,\mathrm{self}} - 1)`
    results.eps_perp_coll : numpy.ndarray
        Reduced collective contribution of the inverse perpendicular dielectric profile
        :math:`(\varepsilon^{-1}_{\perp,\mathrm{coll}} - 1)`

    References
    ----------
    .. footbibliography::
    """

    def __init__(
        self,
        atomgroups: Union[mda.AtomGroup, List[mda.AtomGroup]],
        dim: int = 2,
        zmin: Optional[float] = None,
        zmax: Optional[float] = None,
        bin_width: float = 0.5,
        refgroup: Optional[mda.AtomGroup] = None,
        is_3d: bool = False,
        sym: bool = False,
        unwrap: bool = True,
        temperature: float = 300,
        output_prefix: str = "eps",
        concfreq: int = 0,
        jitter: float = 0.0,
        vcutwidth: float = 0.1,
    ) -> None:
        self._locals = locals()
        if type(atomgroups) not in (list, tuple):
            wrap_compound = get_compound(atomgroups)
        else:  # Get wrap_compound based on fist atom group only
            wrap_compound = get_compound(atomgroups[0])

        if zmin is not None or zmax is not None:
            logger.warn(
                "Setting `zmin` and `zmax` might cut off molecules. This will lead to "
                "severe artifacts in the dielectric profiles."
            )

        super().__init__(
            atomgroups=atomgroups,
            multi_group=True,
            unwrap=unwrap,
            refgroup=refgroup,
            jitter=jitter,
            dim=dim,
            zmin=zmin,
            zmax=zmax,
            bin_width=bin_width,
            wrap_compound=wrap_compound,
            concfreq=concfreq,
        )
        self.is_3d = is_3d
        self.sym = sym

        self.temperature = temperature
        self.output_prefix = output_prefix
        self.vcutwidth = vcutwidth

    def _prepare(self) -> None:
        # Print Alex Schlaich citation
        logger.info(citation_reminder("10.1103/PhysRevLett.117.048001"))

        super()._prepare()

        self.comp = []
        self.inverse_ix = []

        for sel in self.atomgroups:
            comp = get_compound(sel.atoms)
            ix = sel.atoms._get_compound_indices(comp)
            _, inverse_ix = np.unique(ix, return_inverse=True)
            self.comp.append(comp)
            self.inverse_ix.append(inverse_ix)

    def _single_frame(self) -> float:
        super()._single_frame()

        # precalculate total polarization of the box
        self._obs.M = np.dot(
            self._universe.atoms.charges, self._universe.atoms.positions
        )

        self._obs.M_perp = self._obs.M[self.dim]
        self._obs.M_perp_2 = self._obs.M[self.dim] ** 2
        self._obs.M_par = self._obs.M[self.odims]

        n_ag = self.n_atomgroups

        self._obs.m_par = np.zeros((self.n_bins, 2, n_ag))
        self._obs.mM_par = np.zeros((self.n_bins, n_ag))
        self._obs.mm_par = np.zeros((self.n_bins, n_ag))
        self._obs.cmM_par = np.zeros((self.n_bins, n_ag))
        self._obs.cM_par = np.zeros((self.n_bins, 2, n_ag))

        self._obs.m_perp = np.zeros((self.n_bins, n_ag))
        self._obs.mM_perp = np.zeros((self.n_bins, n_ag))
        self._obs.mm_perp = np.zeros((self.n_bins, n_ag))
        self._obs.cmM_perp = np.zeros((self.n_bins, n_ag))
        self._obs.cM_perp = np.zeros((self.n_bins, n_ag))

        # Use polarization density (for perpendicular component)
        # ======================================================
        for i, sel in enumerate(self.atomgroups):
            zbins = np.digitize(
                sel.atoms.positions[:, self.dim], self._obs.bin_edges[1:-1]
            )

            curQ = np.bincount(zbins, weights=sel.atoms.charges, minlength=self.n_bins)

            self._obs.m_perp[:, i] = -np.cumsum(curQ / self._obs.bin_area)
            self._obs.mM_perp[:, i] = self._obs.m_perp[:, i] * self._obs.M_perp
            self._obs.mm_perp[:, i] = self._obs.m_perp[:, i] ** 2 * self._obs.bin_volume
            self._obs.cmM_perp[:, i] = self._obs.m_perp[:, i] * (
                self._obs.M_perp - self._obs.m_perp[:, i] * self._obs.bin_volume
            )

            self._obs.cM_perp[:, i] = (
                self._obs.M_perp - self._obs.m_perp[:, i] * self._obs.bin_volume
            )

            # Use virtual cutting method (for parallel component)
            # ===================================================
            # Move all z-positions to 'center of charge' such that we avoid monopoles in
            # z-direction (compare Eq. 33 in Bonthuis 2012; we only want to cut in x/y
            # direction)
            testpos = sel.center(weights=np.abs(sel.charges), compound=self.comp[i])[
                self.inverse_ix[i], self.dim
            ]

            # Average parallel directions
            for j, direction in enumerate(self.odims):
                # At this point we should not use the wrap, which causes unphysical
                # dipoles at the borders
                Lx = self._ts.dimensions[direction]
                Ax = self._ts.dimensions[self.odims[1 - j]] * self._obs.bin_width

                vbinsx = np.ceil(Lx / self.vcutwidth).astype(int)
                x_bin_edges = (np.arange(vbinsx)) * (Lx / vbinsx)

                zpos = np.digitize(testpos, self._obs.bin_edges[1:-1])
                xbins = np.digitize(sel.atoms.positions[:, direction], x_bin_edges[1:])

                curQx = np.bincount(
                    zpos + self.n_bins * xbins,
                    weights=self.atomgroups[i].charges,
                    minlength=vbinsx * self.n_bins,
                ).reshape(vbinsx, self.n_bins)

                # integral over x, so uniself._ts of area
                self._obs.m_par[:, j, i] = -np.cumsum(curQx / Ax, axis=0).mean(axis=0)

            # Can not use array for operations below, without extensive reshaping of
            # each array... Therefore, take first element only since the volume of each
            # bin is the same in planar geometry.
            bin_volume = self._obs.bin_volume[0]

            self._obs.mM_par[:, i] = np.dot(self._obs.m_par[:, :, i], self._obs.M_par)
            self._obs.mm_par[:, i] = (
                self._obs.m_par[:, :, i] * self._obs.m_par[:, :, i]
            ).sum(axis=1) * bin_volume
            self._obs.cmM_par[:, i] = (
                self._obs.m_par[:, :, i]
                * (self._obs.M_par - self._obs.m_par[:, :, i] * bin_volume)
            ).sum(axis=1)
            self._obs.cM_par[:, :, i] = (
                self._obs.M_par - self._obs.m_par[:, :, i] * bin_volume
            )

        # Save norm of the total parallel dipole moment for correlation analysis.
        return np.linalg.norm(self._obs.M_par)

    def _conclude(self) -> None:
        super()._conclude()

        pref = 1 / scipy.constants.epsilon_0
        pref /= scipy.constants.Boltzmann * self.temperature
        # Convert from ~e^2/m to ~base units
        pref /= scipy.constants.angstrom / (scipy.constants.elementary_charge) ** 2

        self.results.pref = pref
        self.results.V = self.means.bin_volume.sum()

        # Perpendicular component
        # =======================
        cov_perp = self.means.mM_perp - self.means.m_perp * self.means.M_perp

        # Using propagation of uncertainties
        dcov_perp = np.sqrt(
            self.sems.mM_perp**2
            + (self.means.M_perp * self.sems.m_perp) ** 2
            + (self.means.m_perp * self.sems.M_perp) ** 2
        )

        var_perp = self.means.M_perp_2 - self.means.M_perp**2

        cov_perp_self = self.means.mm_perp - (
            self.means.m_perp**2 * self.means.bin_volume[0]
        )
        cov_perp_coll = self.means.cmM_perp - self.means.m_perp * self.means.cM_perp

        if not self.is_3d:
            self.results.eps_perp = -pref * cov_perp
            self.results.eps_perp_self = -pref * cov_perp_self
            self.results.eps_perp_coll = -pref * cov_perp_coll
            self.results.deps_perp = pref * dcov_perp

        else:
            self.results.eps_perp = -cov_perp / (pref**-1 + var_perp / self.results.V)
            self.results.deps_perp = pref * dcov_perp

            self.results.eps_perp_self = (-pref * cov_perp_self) / (
                1 + pref / self.results.V * var_perp
            )
            self.results.eps_perp_coll = (-pref * cov_perp_coll) / (
                1 + pref / self.results.V * var_perp
            )

        # Parallel component
        # ==================
        cov_par = np.zeros((self.n_bins, self.n_atomgroups))
        dcov_par = np.zeros((self.n_bins, self.n_atomgroups))
        cov_par_self = np.zeros((self.n_bins, self.n_atomgroups))
        cov_par_coll = np.zeros((self.n_bins, self.n_atomgroups))

        for i in range(self.n_atomgroups):
            cov_par[:, i] = 0.5 * (
                self.means.mM_par[:, i]
                - np.dot(self.means.m_par[:, :, i], self.means.M_par)
            )

            # Using propagation of uncertainties
            dcov_par[:, i] = 0.5 * np.sqrt(
                self.sems.mM_par[:, i] ** 2
                + np.dot(self.sems.m_par[:, :, i] ** 2, self.means.M_par**2)
                + np.dot(self.means.m_par[:, :, i] ** 2, self.sems.M_par**2)
            )

            cov_par_self[:, i] = 0.5 * (
                self.means.mm_par[:, i]
                - np.dot(
                    self.means.m_par[:, :, i], self.means.m_par[:, :, i].sum(axis=0)
                )
            )
            cov_par_coll[:, i] = 0.5 * (
                self.means.cmM_par[:, i]
                - (self.means.m_par[:, :, i] * self.means.cM_par[:, :, i]).sum(axis=1)
            )

        self.results.eps_par = pref * cov_par
        self.results.deps_par = pref * dcov_par
        self.results.eps_par_self = pref * cov_par_self
        self.results.eps_par_coll = pref * cov_par_coll

        if self.sym:
            symmetrize(self.results.eps_perp, axis=0, inplace=True)
            symmetrize(self.results.deps_perp, axis=0, inplace=True)
            symmetrize(self.results.eps_perp_self, axis=0, inplace=True)
            symmetrize(self.results.eps_perp_coll, axis=0, inplace=True)

            symmetrize(self.results.eps_par, axis=0, inplace=True)
            symmetrize(self.results.deps_par, axis=0, inplace=True)
            symmetrize(self.results.eps_par_self, axis=0, inplace=True)
            symmetrize(self.results.eps_par_coll, axis=0, inplace=True)

    @render_docs
    def save(self) -> None:
        """${SAVE_METHOD_DESCRIPTION}"""
        columns = ["position [Å]"]
        for i, _ in enumerate(self.atomgroups):
            columns.append(f"ε^-1_⟂ - 1 ({i+1})")
        for i, _ in enumerate(self.atomgroups):
            columns.append(f"Δε^-1_⟂ ({i+1})")
        for i, _ in enumerate(self.atomgroups):
            columns.append(f"self ε^-1_⟂ - 1 ({i+1})")
        for i, _ in enumerate(self.atomgroups):
            columns.append(f"coll. ε^-1_⟂ - 1 ({i+1})")

        outdata_perp = np.hstack(
            [
                self.results.bin_pos[:, np.newaxis],
                self.results.eps_perp,
                self.results.deps_perp,
                self.results.eps_perp_self,
                self.results.eps_perp_coll,
            ]
        )

        self.savetxt(
            "{}{}".format(self.output_prefix, "_perp"), outdata_perp, columns=columns
        )

        columns = ["position [Å]"]
        for i, _ in enumerate(self.atomgroups):
            columns.append(f"ε_∥ - 1 ({i+1})")
        for i, _ in enumerate(self.atomgroups):
            columns.append(f"Δε_∥ ({i+1})")
        for i, _ in enumerate(self.atomgroups):
            columns.append(f"self ε_∥ - 1 ({i+1})")
        for i, _ in enumerate(self.atomgroups):
            columns.append(f"coll ε_∥ - 1 ({i+1})")

        outdata_par = np.hstack(
            [
                self.results.bin_pos[:, np.newaxis],
                self.results.eps_par,
                self.results.deps_par,
                self.results.eps_par_self,
                self.results.eps_par_coll,
            ]
        )

        self.savetxt(
            "{}{}".format(self.output_prefix, "_par"), outdata_par, columns=columns
        )
