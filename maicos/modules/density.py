#!/usr/bin/env python3
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
#
# Copyright (c) 2019 Authors and contributors
# (see the file AUTHORS for the full list of names)
#
# Released under the GNU Public Licence, v2 or any higher version
# SPDX-License-Identifier: GPL-2.0-or-later

import warnings

import numpy as np
from scipy import constants

from .base import MultiGroupAnalysisBase
from ..utils import savetxt, atomgroup_header
from ..decorators import planar_base


def mu(rho, temperature, m):
    """Returns the chemical potential calculated from the density: mu = k_B T log(rho. / m)"""

    # De Broglie (converted to nm)
    db = np.sqrt(constants.h**2 / (2 * np.pi * m * constants.atomic_mass *
                                   constants.Boltzmann * temperature))

    # kT in KJ/mol
    kT = temperature * constants.Boltzmann * constants.Avogadro / constants.kilo

    if np.all(rho > 0):
        return kT * np.log(rho * db**3 / (m * constants.atomic_mass))
    elif np.any(rho == 0):
        return np.float64("-inf")
    else:
        return np.float("nan")


def dmu(rho, drho, temperature):
    """Returns the error of the chemical potential calculated from the density using propagation of uncertainty."""

    if np.all(rho > 0):
        return (drho / rho)
    elif np.any(rho == 0):
        return np.float64("-inf")
    else:
        return np.float("nan")


def weight(selection, dens):
    """Calculates the weights for the histogram depending on the choosen type of density.
        Valid values are `mass`, `number`, `charge` or `temp`."""
    if dens == "mass":
        # amu/nm**3 -> kg/m**3
        return selection.atoms.masses * constants.atomic_mass * 1e27
    elif dens == "number":
        return np.ones(selection.atoms.n_atoms)
    elif dens == "charge":
        return selection.atoms.charges
    elif dens == "temp":
        # ((1 amu * Angstrom^2) / (picoseconds^2)) / Boltzmann constant
        prefac = constants.atomic_mass * 1e4 / constants.Boltzmann
        return ((selection.atoms.velocities**2).sum(axis=1) *
                selection.atoms.masses / 2 * prefac)
    else:
        raise ValueError(
            "`{}` not supported. Use `mass`, `number`, `charge` or `temp`".
            format(dens))


@planar_base()
class density_planar(MultiGroupAnalysisBase):
    """Computes partial densities or temperature profiles across the box.

       :param output (str): Output filename
       :param outfreq (int): Default time after which output files are refreshed (1000 ps).
       :param mu (bool): Calculate the chemical potential
       :param muout (str): Prefix for output filename for chemical potential
       :param temperature (float): temperature (K) for chemical potential
       :param mass (float): atommass if not guessed from topology
       :param zpos (float): position at which the chemical potential will be computed. By default average over box.
       :param dens (str): Density: mass, number, charge, electron

       :returns (dict): * z: bins
                        * deans_mean: calcualted densities
                        * dens_err: density error
                        * mu: chemical potential
                        * dmu: error of chemical potential
     """
    def __init__(self,
                 atomgroups,
                 output="density.dat",
                 outfreq=1000,
                 mu=False,
                 muout="muout.dat",
                 temperature=300,
                 mass=np.nan,
                 zpos=None,
                 dens="mass",
                 # Planar base arguments are necessary for buidling CLI
                 dim=2,
                 binwidth=0.1,
                 center=False,
                 comgroup=None,
                 **kwargs):
        super().__init__(atomgroups, **kwargs)
        self.output = output
        self.outfreq = outfreq
        self.mu = mu
        self.muout = muout
        self.temperature = temperature
        self.mass = mass
        self.zpos = zpos
        self.dens = dens

    def _configure_parser(self, parser):
        parser.add_argument('-o', dest='output')
        parser.add_argument('-dout', dest='outfreq')
        parser.add_argument('-mu', dest='mu')
        parser.add_argument('-muo', dest='muout')
        parser.add_argument('-temp', dest='temperature')
        parser.add_argument('-zpos', dest='zpos')
        parser.add_argument('-dens', dest='dens')

    def _prepare(self):
        if self.dens not in ["mass", "number", "charge", "temp"]:
            raise ValueError(
                "Invalid choice for dens: '{}' (choose from 'mass', "
                "'number', 'charge', 'temp')".format(self.dens))
        if self._verbose:
            if self.dens == 'temp':
                print('Computing temperature profile along {}-axes.'.format(
                    'XYZ'[self.dim]))
            else:
                print('Computing {} density profile along {}-axes.'.format(
                    self.dens, 'XYZ'[self.dim]))

        self.density_mean = np.zeros((self.n_bins, self.n_atomgroups))
        self.density_mean_sq = np.zeros((self.n_bins, self.n_atomgroups))

        if self.mu and self.dens != 'mass':
            raise ValueError(
                "Calculation of the chemical potential is only possible when "
                "mass density is selected")

        if self.mu:
            if self.n_atomgroups != 1:
                with warnings.catch_warnings():
                    warnings.simplefilter('always')
                    warnings.warn(
                        "Performing chemical potential analysis for 1st selection"
                        "group '{}'".format(self.atomgroups[0]))
            self.mass = self.atomgroups[0].total_mass(
            ) / self.atomgroups[0].atoms.n_residues

    def _single_frame(self):
        curV = self._ts.volume / 1000
        dz = self._ts.dimensions[self.dim] / self.n_bins

        for index, selection in enumerate(self.atomgroups):
            bins = np.rint(
                (selection.atoms.positions[:, self.dim] + dz / 2) /
                dz) % self.n_bins
            density_ts = np.histogram(bins,
                                      bins=np.arange(self.n_bins + 1),
                                      weights=weight(selection, self.dens))[0]

            if self.dens == 'temp':
                bincount = np.bincount(bins, minlength=self.n_bins)
                self.density_mean[:, index] += density_ts / bincount
                self.density_mean_sq[:, index] += (density_ts / bincount)**2
            else:
                self.density_mean[:, index] += density_ts / curV * self.n_bins
                self.density_mean_sq[:, index] += (density_ts / curV *
                                                   self.n_bins)**2

        if self._save and self._frame_index % self.outfreq == 0 and self._frame_index > 0:
            self._calculate_results()
            self._save_results()

    def _calculate_results(self):
        self._index = self._frame_index + 1

        self.results["dens_mean"] = self.density_mean / self._index
        self.results["dens_mean_sq"] = self.density_mean_sq / self._index

        self.results["dens_std"] = np.nan_to_num(
            np.sqrt(self.results["dens_mean_sq"] -
                    self.results["dens_mean"]**2))
        self.results["dens_err"] = self.results["dens_std"] / \
            np.sqrt(self._index)

        # chemical potential
        if self.mu:
            if (self.zpos is not None):
                dz = self.Lz / self._index / self.n_bins
                this = (np.rint((self.zpos * 10. + dz / 2.) / dz) %
                        self.n_bins).astype(int)
                if self.center:
                    this += np.rint(self.n_bins / 2).astype(int)
                self.results["mu"] = mu(self.results["dens_mean"][this][0],
                                        self.temperature, self.mass)
                self.results["dmu"] = dmu(self.results["dens_mean"][this][0],
                                          self.results["dens_err"][this][0],
                                          self.temperature)
            else:
                self.results["mu"] = np.mean(
                    mu(self.results["dens_mean"], self.temperature, self.mass))
                self.results["dmu"] = np.mean(
                    dmu(self.results["dens_mean"], self.results["dens_err"],
                        self.temperature))

    def _save_results(self):
        # write header
        if self.dens == "mass":
            units = "kg m^(-3)"
        elif self.dens == "number":
            units = "nm^(-3)"
        elif self.dens == "charge":
            units = "e nm^(-3)"
        elif self.dens == "temp":
            units = "K"

        if self.dens == 'temp':
            columns = "temperature profile [{}]".format(units)
        else:
            columns = "{} density profile [{}]".format(self.dens, units)
        columns += "\nstatistics over {:.1f} picoseconds \npositions [nm]".format(
            self._index * self._universe.trajectory.dt)
        try:
            for group in self.atomgroups:
                columns += "\t" + atomgroup_header(group)
            for group in self.atomgroups:
                columns += "\t" + atomgroup_header(group) + " error"
        except AttributeError:
            with warnings.catch_warnings():
                warnings.simplefilter('always')
                warnings.warn("AtomGroup does not contain resnames."
                              " Not writing residues information to output.")

        # save density profile
        savetxt(self.output,
                np.hstack(
                    ((self.results["z"][:, np.newaxis]/10),
                     self.results["dens_mean"], self.results["dens_err"])),
                header=columns)

        if self.mu:
            # save chemical potential
            savetxt(self.muout,
                    np.hstack((self.results["mu"], self.results["dmu"]))[None],
                    header="μ [kJ/mol]\t μ error [kJ/mol]")


class density_cylinder(MultiGroupAnalysisBase):
    """Computes partial densities across a cylinder of given radius r and length l.

   :param output (str): Output filename
   :param outfreq (int): Default time after which output files are refreshed (1000 ps).
   :param dim (int): Dimension for binning (0=X, 1=Y, 2=Z)
   :param center (str): Perform the binning relative to the center of this selection string of teh first AtomGroup.
                        If None center of box is used.
   :param radius (float): Radius of the cylinder (nm). If None smallest box extension is taken.
   :param binwidth (float): binwidth (nanometer)
   :param length (float): Length of the cylinder (nm). If None length of box in the binning dimension is taken.
   :param dens (str): Density: mass, number, charge, temp

   :returns (dict): * z: bins
                    * deans_mean: calcualted densities
                    * dens_err: density error
    """
    def __init__(self,
                 atomgroups,
                 output="density_cylinder.dat",
                 outfreq=1000,
                 dim=2,
                 center=None,
                 radius=None,
                 binwidth=0.1,
                 length=None,
                 dens="mass",
                 **kwargs):
        super().__init__(atomgroups, **kwargs)
        self.output = output
        self.outfreq = outfreq
        self.dim = dim
        self.binwidth = binwidth
        self.center = center
        self.radius = radius
        self.length = length
        self.dens = dens

    def _configure_parser(self, parser):
        parser.add_argument('-o', dest='output')
        parser.add_argument('-dout', dest='outfreq')
        parser.add_argument('-d', dest='dim')
        parser.add_argument('-center', dest='center')
        parser.add_argument('-r', dest='radius')
        parser.add_argument('-dr', dest='binwidth')
        parser.add_argument('-l', dest='length')
        parser.add_argument('-dens', dest='dens')

    def _prepare(self):

        if self.dens not in ["mass", "number", "charge", "temp"]:
            raise ValueError(
                "Invalid choice for dens: '{}' (choose from 'mass', "
                "'number', 'charge', 'temp')".format(self.dens))

        if self._verbose:
            if self.dens == 'temp':
                print('Computing temperature profile along {}-axes.'.format(
                    'XYZ'[self.dim]))
            else:
                print('Computing radial {} density profile along {}-axes.'.
                      format(self.dens, 'XYZ'[self.dim]))

        self.odims = np.roll(np.arange(3), -self.dim)[1:]

        if self.center is None:
            if self._verbose:
                print("No center given --> Take from box dimensions.")
            self.centersel = None
            center = self.atomgroups[0].dimensions[:3] / 2
        else:
            self.centersel = self.atomgroups[0].select_atoms(self.center)
            if len(self.centersel) == 0:
                raise RuntimeError("No atoms found in center selection. "
                                   "Please adjust selection!")
            center = self.centersel.center_of_mass()

        if self._verbose:
            print("Initial center at {}={:.3f} nm and {}={:.3f} nm.".format(
                'XYZ'[self.odims[0]], center[self.odims[0]] / 10,
                'XYZ'[self.odims[1]], center[self.odims[1]] / 10))

        if self.radius is None:
            self.radius = self.atomgroups[0].dimensions[self.odims].min() / 2
            if self._verbose:
                print(
                    "No radius given --> Take smallest box extension (r={:.2f} nm)."
                    .format(self.radius / 10))
        else:
            self.radius /= 10

        if self.length is None:
            self.length = self.atomgroups[0].dimensions[self.dim]
            if self._verbose:
                print("No length given --> Take length in {}.".format(
                    'XYZ'[self.dim]))
        else:
            self.length /= 10

        self.nbins = int(np.ceil(self.radius / 10 / self.binwidth))

        self.density_mean = np.zeros((self.nbins, self.n_atomgroups))
        self.density_mean_sq = np.zeros((self.nbins, self.n_atomgroups))

        self._dr = np.ones(self.nbins) * self.radius / self.nbins
        self._r_bins = np.arange(self.nbins) * self._dr + self._dr
        self._delta_r_sq = self._r_bins**2 - \
            np.insert(self._r_bins, 0, 0)[0:-1]**2  # r_o^2 - r_i^2

        if self._verbose:
            print("\n")
            print('Using', self.nbins, 'bins.')

    def _single_frame(self):
        # calculater center of cylinder.
        if self.center is None:
            center = self.atomgroups[0].dimensions[:3] / 2
        else:
            center = self.centersel.center_of_mass()

        for index, selection in enumerate(self.atomgroups):

            # select cylinder of the given length and radius
            cut = selection.atoms[np.where(
                np.absolute(selection.atoms.positions[:, self.dim] -
                            center[self.dim]) < self.length / 2)[0]]
            cylinder = cut.atoms[np.where(
                np.linalg.norm((cut.atoms.positions[:, self.odims] -
                                center[self.odims]),
                               axis=1) < self.radius)[0]]

            radial_positions = np.linalg.norm(
                (cylinder.atoms.positions[:, self.odims] - center[self.odims]),
                axis=1)
            bins = np.digitize(radial_positions, self._r_bins)
            density_ts = np.histogram(bins,
                                      bins=np.arange(self.nbins + 1),
                                      weights=weight(cylinder, self.dens))[0]

            if self.dens == 'temp':
                bincount = np.bincount(bins, minlength=self.nbins)
                self.density_mean[:, index] += density_ts / bincount
                self.density_mean_sq[:, index] += (density_ts / bincount)**2
            else:
                self.density_mean[:, index] += density_ts * 1000 / (
                    np.pi * self._delta_r_sq * self.length)
                self.density_mean_sq[:, index] += (
                    density_ts * 1000 /
                    (np.pi * self._delta_r_sq * self.length))**2

        if self._save and self._frame_index % self.outfreq == 0 and self._frame_index > 0:
            self._calculate_results()
            self._save_results()

    def _calculate_results(self):
        self._index = self._frame_index + 1

        self.results["r"] = (np.copy(self._r_bins) - self._dr / 2) / 10
        self.results["dens_mean"] = self.density_mean / self._index
        self.results["dens_mean_sq"] = self.density_mean_sq / self._index

        self.results["dens_std"] = np.nan_to_num(
            np.sqrt(self.results["dens_mean_sq"] -
                    self.results["dens_mean"]**2))
        self.results["dens_err"] = self.results["dens_std"] / np.sqrt(
            self._index)

    def _save_results(self):
        # write header
        if self.dens == "mass":
            units = "kg m^(-3)"
        elif self.dens == "number":
            units = "nm^(-3)"
        elif self.dens == "charge":
            units = "e nm^(-3)"
        elif self.dens == "temp":
            units = "K"

        if self.dens == 'temp':
            columns = "temperature profile [{}]".format(units)
        else:
            columns = "{} density profile [{}]".format(self.dens, units)
        columns += "\nstatistics over {:.1f} picoseconds \npositions [nm]".format(
            self._index * self._universe.trajectory.dt)
        for group in self.atomgroups:
            columns += "\t" + atomgroup_header(group)
        for group in self.atomgroups:
            columns += "\t" + atomgroup_header(group) + " error"

        # save density profile
        savetxt(self.output,
                np.hstack(
                    ((self.results["r"][:, np.newaxis]),
                     self.results["dens_mean"], self.results["dens_err"])),
                header=columns)
