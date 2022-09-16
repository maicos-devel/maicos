#!/usr/bin/env python3
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
#
# Copyright (c) 2022 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the GNU Public Licence, v3 or any higher version
# SPDX-License-Identifier: GPL-3.0-or-later
r"""Tools for computing density temperature, and chemical potential profiles.

The density modules of MAICoS allow for computing density,
temperature, and chemical potential profiles from molecular
simulation trajectory files. Profiles can be extracted either
in Cartesian or cylindrical coordinate systems. Units for the density
are the same as GROMACS, i.e. mass, number or charge (see the `gmx density`_
manual for details about the unit system).

.. _`gmx density`: https://manual.gromacs.org/documentation/current/onlinehelp/gmx-density.html  # noqa: E501
"""

import logging
import warnings

import numpy as np
from MDAnalysis.exceptions import NoDataError

from ..core import ProfileCylinderBase, ProfilePlanarBase
from ..lib.math import dmu, mu
from ..lib.util import atomgroup_header, render_docs
from ..lib.weights import density_weights, temperature_weights


logger = logging.getLogger(__name__)


@render_docs
class ChemicalPotentialPlanar(ProfilePlanarBase):
    """Compute the chemical potential in a cartesian geometry.

    Parameters
    ----------
    ${PROFILE_PLANAR_CLASS_PARAMETERS}
    center : bool
        Calculate chemical potential only in the center of the simulation cell.
    temperature : float
        temperature (K) for chemical potential
    mass : float
        Mass (u) for the chemical potential. By default taken from topology.
    zpos : float
        position (Å) at which the chemical potential will be computed.
        By default average over box.
    muout : str
        Prefix for output filename for chemical potential

    Attributes
    ----------
    ${PROFILE_PLANAR_CLASS_ATTRIBUTES}
    results.mu : float
        chemical potential (only if `mu=True`)
    results.dmu : float
        error of chemical potential (only if `mu=True`)
    """

    def __init__(self,
                 atomgroups,
                 dim=2,
                 zmin=None,
                 zmax=None,
                 bin_width=1,
                 refgroup=None,
                 sym=False,
                 grouping="atoms",
                 unwrap=True,
                 binmethod="com",
                 output="density.dat",
                 concfreq=0,
                 center=False,
                 temperature=300,
                 mass=None,
                 zpos=None,
                 muout="muout.dat"):
        super(ChemicalPotentialPlanar, self).__init__(
            function=density_weights,
            f_kwargs={"dens": "number"},
            normalization="volume",
            atomgroups=atomgroups,
            dim=dim,
            zmin=zmin,
            zmax=zmax,
            bin_width=bin_width,
            refgroup=refgroup,
            sym=sym,
            grouping=grouping,
            unwrap=unwrap,
            binmethod=binmethod,
            output=output,
            concfreq=concfreq)

        self.center = center
        self.temperature = temperature
        self.mass = mass
        self.zpos = zpos
        self.muout = muout

    def _prepare(self):
        super(ChemicalPotentialPlanar, self)._prepare()

        if not self.mass:
            try:
                self.atomgroups[0].universe.atoms.masses
            except NoDataError:
                raise ValueError("Calculation of the chemical potential "
                                 "is only possible when masses are "
                                 "present in the topology or masses are "
                                 "supplied by the user.")
            get_mass_from_topol = True
            self.mass = np.array([])
        else:
            if len([self.mass]) == 1:
                self.mass = np.array([self.mass])
            else:
                self.mass = np.array(self.mass)
            get_mass_from_topol = False

        self.n_res = np.array([])
        self.n_atoms = np.array([])

        for ag in self.atomgroups:
            if not len(ag.atoms) == len(ag.residues.atoms):
                with warnings.catch_warnings():
                    warnings.simplefilter('always')
                    warnings.warn("Selections contains incomplete residues."
                                  "MAICoS uses the total mass of the "
                                  "residues to calculate the chemical "
                                  "potential. Your results will be "
                                  "incorrect! You can supply your own "
                                  "masses with the -mass flag.")

            ag_res = ag.residues
            mass = []
            n_atoms = 0
            n_res = 0
            while len(ag_res.atoms):
                n_res += 1
                resgroup = ag_res - ag_res
                n_atoms += len(ag_res.residues[0].atoms)

                for res in ag_res.residues:
                    if np.all(res.atoms.types
                              == ag_res.residues[0].atoms.types):
                        resgroup = resgroup + res
                ag_res = ag_res - resgroup
                if get_mass_from_topol:
                    mass.append(resgroup.total_mass() / resgroup.n_residues)
            if not n_res == n_atoms and n_res > 1:
                raise NotImplementedError(
                    "Selection contains multiple types of residues and at "
                    "least one them is a molecule. Molecules are not "
                    "supported when selecting multiple residues."
                    )
            self.n_res = np.append(self.n_res, n_res)
            self.n_atoms = np.append(self.n_atoms, n_atoms)
            if get_mass_from_topol:
                self.mass = np.append(self.mass, np.sum(mass))

    def _conclude(self):
        super(ChemicalPotentialPlanar, self)._conclude()

        if self.zpos is not None:
            this = (np.rint(
                (self.zpos + self.means.bin_width / 2) / self.means.bin_width)
                % self.n_bins).astype(int)
            if self.center:
                this += np.rint(self.n_bins / 2).astype(int)
            self.results.mu = mu(self.results.profile_mean[this]
                                 / self.n_atoms,
                                 self.temperature, self.mass)
            self.results.dmu = dmu(self.results.profile_mean[this]
                                   / self.n_atoms,
                                   self.results.profile_err[this]
                                   / self.n_atoms, self.temperature)
        else:
            self.results.mu = np.mean(
                mu(self.results.profile_mean / self.n_atoms,
                   self.temperature,
                   self.mass), axis=0)
            self.results.dmu = np.mean(
                dmu(self.results.profile_mean / self.n_atoms,
                    self.results.profile_err,
                    self.temperature), axis=0)

    def save(self):
        """Save results of analysis to file."""
        super(ChemicalPotentialPlanar, self).save()

        if self.zpos is not None:
            columns = "Chemical potential calculated at "
            columns += f"z = {self.zpos} Å."
        else:
            columns = "Chemical potential averaged over the whole system."
        columns += "\nstatistics over "
        columns += "{self._index * self._trajectory.dt:.1f} ps\n"
        try:
            for group in self.atomgroups:
                columns += atomgroup_header(group) + " μ [kJ/mol]" + "\t"
            for group in self.atomgroups:
                columns += atomgroup_header(group) + " μ error [kJ/mol]" \
                    + "\t"
        except AttributeError:
            with warnings.catch_warnings():
                warnings.simplefilter('always')
                warnings.warn("AtomGroup does not contain resnames."
                              " Not writing residues information to output.")
        self.savetxt(self.muout,
                     np.hstack((self.results.mu, self.results.dmu))[None],
                     columns=columns)


@render_docs
class TemperaturePlanar(ProfilePlanarBase):
    """Compute temperature profile in a cartesian geometry.

    Currently only atomistic temperature profiles are supported.

    Parameters
    ----------
    ${PROFILE_PLANAR_CLASS_PARAMETERS}

    Attributes
    ----------
    ${PROFILE_PLANAR_CLASS_ATTRIBUTES}
    """

    def __init__(self,
                 atomgroups,
                 dim=2,
                 zmin=None,
                 zmax=None,
                 bin_width=1,
                 refgroup=None,
                 sym=False,
                 grouping="atoms",
                 unwrap=True,
                 binmethod="com",
                 output="temperature.dat",
                 concfreq=0):

        super(TemperaturePlanar, self).__init__(
            function=temperature_weights,
            normalization="number",
            atomgroups=atomgroups,
            dim=dim,
            zmin=zmin,
            zmax=zmax,
            bin_width=bin_width,
            refgroup=refgroup,
            sym=sym,
            grouping=grouping,
            unwrap=unwrap,
            binmethod=binmethod,
            output=output,
            concfreq=concfreq)


@render_docs
class DensityPlanar(ProfilePlanarBase):
    r"""Compute the partial density profile in a cartesian geometry.

    Calculation are carried out for mass
    (:math:`\rm u \cdot A^{-3}`), number (:math`\rm A^{-3}`) or
    charge (:math:`\rm e \cdot A^{-3}`) density profiles along a certain
    cartesian axes [x,y,z] of the simulation cell. Supported cells can be of
    arbitrary shapes and as well fluctuate over time.

    For grouping with respect to molecules, residues etc. the corresponding
    centers (i.e center of mass) using of periodic boundary conditions
    are calculated.
    For these center calculations molecules will be unwrapped/made whole.
    Trajectories containing already whole molecules can be run with
    `unwrap=False` to gain a speedup.
    For grouping with respect to atoms the `unwrap` option is always
    ignored.

    Parameters
    ----------
    ${PROFILE_PLANAR_CLASS_PARAMETERS}
    dens : str {'mass', 'number', 'charge'}
        density type to be calculated

    Attributes
    ----------
    ${PROFILE_PLANAR_CLASS_ATTRIBUTES}
    """

    def __init__(self,
                 atomgroups,
                 dens="mass",
                 dim=2,
                 zmin=None,
                 zmax=None,
                 bin_width=1,
                 refgroup=None,
                 sym=False,
                 grouping="atoms",
                 unwrap=True,
                 binmethod="com",
                 output="density.dat",
                 concfreq=0):

        super(DensityPlanar, self).__init__(
            function=density_weights,
            f_kwargs={"dens": dens},
            normalization="volume",
            atomgroups=atomgroups,
            dim=dim,
            zmin=zmin,
            zmax=zmax,
            bin_width=bin_width,
            refgroup=refgroup,
            sym=sym,
            grouping=grouping,
            unwrap=unwrap,
            binmethod=binmethod,
            output=output,
            concfreq=concfreq)


@render_docs
class DensityCylinder(ProfileCylinderBase):
    r"""Compute partial densities across a cylinder.

    Calculation are carried out for mass
    (:math:`\rm u \cdot A^{-3}`), number (:math`\rm A^{-3}`) or
    charge (:math:`\rm e \cdot A^{-3}`) density profiles along the radial
    axes.

    For grouping with respect to molecules, residues etc. the corresponding
    centers (i.e center of mass) using of periodic boundary conditions
    are calculated.
    For these center calculations molecules will be unwrapped/made whole.
    Trajectories containing already whole molecules can be run with
    `unwrap=False` to gain a speedup.
    For grouping with respect to atoms the `unwrap` option is always
    ignored.

    Parameters
    ----------
    ${PROFILE_CYLINDER_CLASS_PARAMETERS}
    dens : str {'mass', 'number', 'charge'}
        density type to be calculated

    Attributes
    ----------
    ${PROFILE_CYLINDER_CLASS_ATTRIBUTES}
    """

    def __init__(self,
                 atomgroups,
                 dens="mass",
                 dim=2,
                 zmin=None,
                 zmax=None,
                 bin_width=1,
                 rmin=0,
                 rmax=None,
                 refgroup=None,
                 grouping="atoms",
                 unwrap=True,
                 binmethod="com",
                 output="density.dat",
                 concfreq=0):

        super(DensityCylinder, self).__init__(
            function=density_weights,
            f_kwargs={"dens": dens},
            normalization="volume",
            atomgroups=atomgroups,
            dim=dim,
            zmin=zmin,
            zmax=zmax,
            bin_width=bin_width,
            rmin=rmin,
            rmax=rmax,
            refgroup=refgroup,
            grouping=grouping,
            unwrap=unwrap,
            binmethod=binmethod,
            output=output,
            concfreq=concfreq)
