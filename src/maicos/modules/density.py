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
from scipy import constants

from ..decorators import set_profile_planar_class_doc, set_verbose_doc
from ..utils import atomgroup_header
from .base import AnalysisBase, ProfilePlanarBase


logger = logging.getLogger(__name__)


def mu(rho, temperature, m):
    """Calculate the chemical potential.

    The chemical potential is calculated from the
    density: mu = k_B T log(rho. / m)
    """
    # kT in KJ/mol
    kT = temperature * constants.Boltzmann \
        * constants.Avogadro / constants.kilo

    results = []

    for srho, mass in zip(np.array(rho).T, m):
        # De Broglie (converted to nm)
        db = np.sqrt(
            constants.h ** 2 / (2 * np.pi * mass * constants.atomic_mass
                                * constants.Boltzmann * temperature)
            ) / constants.angstrom

        if np.all(srho > 0):
            results.append(kT * np.log(srho * db ** 3))
        elif np.any(srho == 0):
            results.append(np.float64("-inf") * np.ones(srho.shape))
        else:
            results.append(np.float64("nan") * np.ones(srho.shape))
    return np.squeeze(np.array(results).T)


def dmu(rho, drho, temperature):
    """Calculate the error of the chemical potential.

    The error is calculated from the density using propagation of uncertainty.
    """
    kT = temperature * constants.Boltzmann \
        * constants.Avogadro / constants.kilo

    results = []

    for srho, sdrho in zip(np.array(rho).T, np.array(drho).T):
        if np.all(srho > 0):
            results.append(kT * (sdrho / srho))
        else:
            results.append(np.float64("nan") * np.ones(srho.shape))
    return np.squeeze(np.array(results).T)


def _density_weights(atomgroup, grouping, dim, dens):
    """Calculate the weights for the histogram.

    Supported values are `mass`, `number` or `charge`.
    """
    if dens == "number":
        # There exist no properrty like n_molecules
        if grouping == "molecules":
            numbers = len(np.unique(atomgroup.molnums))
        else:
            numbers = getattr(atomgroup, f"n_{grouping}")
        return np.ones(numbers)
    elif dens == "mass":
        if grouping == "atoms":
            masses = atomgroup.masses
        else:
            masses = atomgroup.total_mass(compound=grouping)
        return masses
    elif dens == "charge":
        if grouping == "atoms":
            return atomgroup.charges
        else:
            return atomgroup.total_charge(compound=grouping)
    else:
        raise ValueError(f"`{dens}` not supported. "
                         "Use `mass`, `number` or `charge`.")


def _temperature(ag, grouping, dim):
    """Calculate contribution of each atom to the temperature."""
    # ((1 amu * Å^2) / (ps^2)) / Boltzmann constant
    prefac = constants.atomic_mass * 1e4 / constants.Boltzmann
    return (ag.velocities ** 2).sum(axis=1) * ag.atoms.masses / 2 * prefac


def _weights_legacy(ag, dens):
    """Calculate the weights for the histogram in cylindrical systems."""
    if dens == "temp":
        return _temperature(ag, grouping="atoms", dim=None)
    elif dens in ["mass", "number", "charge"]:
        return _density_weights(ag, grouping="atoms", dim=None, dens=dens)
    else:
        raise ValueError(f"`{dens}` not supported. "
                         "Use `mass`, `number`, `charge` or `temp`.")


@set_verbose_doc
@set_profile_planar_class_doc
class ChemicalPotentialPlanar(ProfilePlanarBase):
    """Compute the chemical potential in a cartesian geometry.

    Parameters
    ----------
    ${PLANAR_PROFILE_CLASS_PARAMETERS}
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
    ${VERBOSE_PARAMETER}

    Attributes
    ----------
    ${PLANAR_PROFILE_CLASS_ATTRIBUTES}
    results.mu : float
        chemical potential (only if `mu=True`)
    results.dmu : float
        error of chemical potential (only if `mu=True`)
    """

    def __init__(self,
                 atomgroups,
                 dim=2,
                 zmin=0,
                 zmax=None,
                 binwidth=1,
                 comgroup=None,
                 sym=False,
                 grouping="atoms",
                 make_whole=True,
                 binmethod="com",
                 output="density.dat",
                 concfreq=0,
                 center=False,
                 temperature=300,
                 mass=None,
                 zpos=None,
                 muout="muout.dat",
                 **kwargs):
        super(ChemicalPotentialPlanar, self).__init__(
            function=_density_weights,
            f_kwargs={"dens": "number"},
            normalization="volume",
            atomgroups=atomgroups,
            dim=dim,
            zmin=zmin,
            zmax=zmax,
            binwidth=binwidth,
            comgroup=comgroup,
            sym=sym,
            grouping=grouping,
            make_whole=make_whole,
            binmethod=binmethod,
            output=output,
            concfreq=concfreq,
            **kwargs)

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
                (self.zpos + self.binwidth / 2) / self.binwidth)
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


@set_verbose_doc
@set_profile_planar_class_doc
class TemperaturePlanar(ProfilePlanarBase):
    """Compute temperature profile in a cartesian geometry.

    Currently only atomistic temperature profiles are supported.

    Parameters
    ----------
    ${PLANAR_PROFILE_CLASS_PARAMETERS}
    ${VERBOSE_PARAMETER}

    Attributes
    ----------
    ${PLANAR_PROFILE_CLASS_ATTRIBUTES}
    """

    def __init__(self,
                 atomgroups,
                 dim=2,
                 zmin=0,
                 zmax=None,
                 binwidth=1,
                 center=False,
                 comgroup=None,
                 sym=False,
                 grouping="atoms",
                 make_whole=True,
                 binmethod="com",
                 output="temperature.dat",
                 concfreq=0,
                 **kwargs):

        super(TemperaturePlanar, self).__init__(
            function=_temperature,
            normalization="number",
            atomgroups=atomgroups,
            dim=dim,
            zmin=zmin,
            zmax=zmax,
            binwidth=binwidth,
            center=center,
            comgroup=comgroup,
            sym=sym,
            grouping=grouping,
            make_whole=make_whole,
            binmethod=binmethod,
            output=output,
            concfreq=concfreq,
            **kwargs)

        if grouping != "atoms":
            raise NotImplementedError(f"Temperature profiles of '{grouping}'"
                                      "are not supported. Use 'atoms' "
                                      "instead.'")


@set_verbose_doc
@set_profile_planar_class_doc
class DensityPlanar(ProfilePlanarBase):
    r"""Compute the partial density profile in a cartesian geometry.

    Calculation are carried out for mass
    (:math:`\rm u \cdot A^{-3}`), number (:math`\rm A^{-3}`) or
    charge (:math:`\rm e \cdot A^{-3}`) density profiles along a certain
    cartesian axes [x,y,z] of the simulation cell. Supported cells can be of
    arbitary shapes and as well fluctuate over time.

    For grouping with respect to molecules, residues etc. the corresponding
    centers (i.e center of mass) using of periodic boundary conditions
    are calculated.
    For these center calculations molecules will be unwrapped/made whole.
    Trajectories containing already whole molecules can be run with
    `make_whole=False` to gain a speedup.
    For grouping with respect to atoms the `make_whole` option is always
    ignored.

    Parameters
    ----------
    ${PLANAR_PROFILE_CLASS_PARAMETERS}
    dens : str {'mass', 'number', 'charge'}
        density type to be calculated
    ${VERBOSE_PARAMETER}

    Attributes
    ----------
    ${PLANAR_PROFILE_CLASS_ATTRIBUTES}
    """

    def __init__(self,
                 atomgroups,
                 dens="mass",
                 dim=2,
                 zmin=0,
                 zmax=None,
                 binwidth=1,
                 center=False,
                 comgroup=None,
                 sym=False,
                 grouping="atoms",
                 make_whole=True,
                 binmethod="com",
                 output="density.dat",
                 concfreq=0,
                 **kwargs):

        super(DensityPlanar, self).__init__(
            function=_density_weights,
            f_kwargs={"dens": dens},
            normalization="volume",
            atomgroups=atomgroups,
            dim=dim,
            zmin=zmin,
            zmax=zmax,
            binwidth=binwidth,
            center=center,
            comgroup=comgroup,
            sym=sym,
            grouping=grouping,
            make_whole=make_whole,
            binmethod=binmethod,
            output=output,
            concfreq=concfreq,
            **kwargs)


@set_verbose_doc
class DensityCylinder(AnalysisBase):
    """Compute partial densities across a cylinder.

    Parameters
    ----------
    atomgroups : list[AtomGroup]
        A list of :class:`~MDAnalysis.core.groups.AtomGroup` for which
        the densities are calculated.
    dens : str
        Density: mass, number, charge, temp
    dim : int
        Dimension for binning (x=0, y=1, z=2)
    center : str
        Perform the binning relative to the center of this selection
        string of teh first AtomGroup. If `None` center of box is used.
    radius : float
        Radius of the cylinder (Å). If None smallest box extension is taken.
    length : float
        Length of the cylinder (Å). If None length of box in the
        binning dimension is taken.
    binwidth : float
        binwidth (nanometer)
    output : str
        Output filename
    concfreq : int
        Default number of frames after which results are calculated
        and files refreshed. If `0` results are only calculated at
        the end of the analysis and not saved by default.
    ${VERBOSE_PARAMETER}

    Attributes
    ----------
    results.r : np.ndarray
        bins
    results.dens_mean : np.ndarray
        calculated densities
    results.dens_mean_sq : np.ndarray
        squared calculated density
    results.dens_std : np.ndarray
        density standard deviation
    results.dens_err : np.ndarray
        density error
    """

    def __init__(self,
                 atomgroups,
                 dens="mass",
                 dim=2,
                 center=None,
                 radius=None,
                 length=None,
                 binwidth=1,
                 output="density_cylinder.dat",
                 concfreq=0,
                 **kwargs):
        super(DensityCylinder, self).__init__(atomgroups,
                                              multi_group=True,
                                              **kwargs)
        self.dim = dim
        self.binwidth = binwidth
        self.center = center
        self.radius = radius
        self.length = length
        self.dens = dens
        self.output = output
        self.concfreq = concfreq

    def _prepare(self):
        if self.dens not in ["mass", "number", "charge", "temp"]:
            raise ValueError(f"Invalid choice for dens: '{self.dens}' "
                             "(choose from 'mass', 'number', "
                             "'charge', 'temp'")

        if self.dens == 'temp':
            profile_str = "temperature"
        else:
            profile_str = f"{self.dens} density"

        logger.info(f"Computing {profile_str} profile "
                    f"along {'XYZ'[self.dim]}-axes.")

        self.odims = np.roll(np.arange(3), -self.dim)[1:]

        if self.center is None:
            logger.info("No center given --> Take from box dimensions.")
            self.centersel = None
            center = self.atomgroups[0].dimensions[:3] / 2
        else:
            self.centersel = self.atomgroups[0].select_atoms(self.center)
            if len(self.centersel) == 0:
                raise RuntimeError("No atoms found in center selection. "
                                   "Please adjust selection!")
            center = self.centersel.center_of_mass()

        logger.info("Initial center at "
                    f"{'XYZ'[self.odims[0]]} = "
                    f"{center[self.odims[0]]:.3f} Å and "
                    f"{'XYZ'[self.odims[1]]} = "
                    f"{center[self.odims[1]]:.3f} Å.")

        if self.radius is None:
            self.radius = self.atomgroups[0].dimensions[self.odims].min() / 2
            logger.info("No radius given --> Take smallest box "
                        f"extension (r={self.radius:.2f} Å).")

        if self.length is None:
            self.length = self.atomgroups[0].dimensions[self.dim]
            logger.info("No length given "
                        f"--> Take length in {'XYZ'[self.dim]}.")

        self.n_bins = int(np.ceil(self.radius / self.binwidth))

        self.density_mean = np.zeros((self.n_bins, self.n_atomgroups))
        self.density_mean_sq = np.zeros((self.n_bins, self.n_atomgroups))

        self._dr = np.ones(self.n_bins) * self.radius / self.n_bins
        self._r_bins = np.arange(self.n_bins) * self._dr + self._dr
        self._delta_r_sq = self._r_bins ** 2 \
            - np.insert(self._r_bins, 0, 0)[0:-1] ** 2  # r_o^2 - r_i^2

        logger.info(f"Using {self.n_bins} bins.")

    def _single_frame(self):
        # calculater center of cylinder.
        if self.center is None:
            center = self.atomgroups[0].dimensions[:3] / 2
        else:
            center = self.centersel.center_of_mass()

        for index, selection in enumerate(self.atomgroups):

            # select cylinder of the given length and radius
            cut = selection.atoms[np.where(
                np.absolute(selection.atoms.positions[:, self.dim]
                            - center[self.dim]) < self.length / 2)[0]]
            cylinder = cut.atoms[np.where(
                np.linalg.norm((cut.atoms.positions[:, self.odims]
                                - center[self.odims]),
                               axis=1) < self.radius)[0]]

            radial_positions = np.linalg.norm(
                (cylinder.atoms.positions[:, self.odims] - center[self.odims]),
                axis=1)

            weights = _weights_legacy(cylinder, self.dens)
            density_ts, _ = np.histogram(radial_positions,
                                         bins=self.n_bins,
                                         range=(0, self.radius),
                                         weights=weights)

            if self.dens == 'temp':
                bincount = np.histogram(radial_positions,
                                        bins=self.n_bins,
                                        range=(0, self.radius))[0]
                self.density_mean[:, index] += density_ts / bincount
                self.density_mean_sq[:, index] += (density_ts / bincount) ** 2
            else:
                self.density_mean[:, index] += density_ts \
                    / (np.pi * self._delta_r_sq * self.length)
                self.density_mean_sq[:, index] += (density_ts
                                                   / (np.pi * self._delta_r_sq
                                                      * self.length)) ** 2

    def _conclude(self):
        self.results.r = (np.copy(self._r_bins) - self._dr / 2)
        self.results.dens_mean = self.density_mean / self._index
        self.results.dens_mean_sq = self.density_mean_sq / self._index

        self.results.dens_std = np.nan_to_num(
            np.sqrt(self.results.dens_mean_sq
                    - self.results.dens_mean ** 2))
        self.results.dens_err = self.results.dens_std / np.sqrt(self._index)

    def save(self):
        """Save results of analysis to file."""
        if self.dens == "mass":
            units = "amu Å^(-3)"
        elif self.dens == "number":
            units = "Å^(-3)"
        elif self.dens == "charge":
            units = "e Å^(-3)"
        elif self.dens == "temp":
            units = "K"

        if self.dens == 'temp':
            profile_type = f"temperature [{units}]"
        else:
            profile_type = f"{self.dens} dens. [{units}]"
        columns = ["positions [Å]"]
        for i, _ in enumerate(self.atomgroups):
            columns.append(f"{profile_type} {i+1}")
        for i, _ in enumerate(self.atomgroups):
            columns.append(f"{profile_type} error {i+1}")

        # save density profile
        self.savetxt(self.output,
                     np.hstack(
                         ((self.results.r[:, np.newaxis]),
                          self.results.dens_mean, self.results.dens_err)),
                     columns=columns)
