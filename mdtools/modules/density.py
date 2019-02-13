#!/usr/bin/env python3
# coding: utf-8

import os
import sys

import numpy as np
from scipy import constants

from .base import AnalysisBase
from ..utils import savetxt


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


class density_planar(AnalysisBase):
    """Computes partial densities or temperature profiles across the box.
       For group selections use strings in the MDAnalysis selection command style."""

    def __init__(self,
                 atomgroup,
                 output="density",
                 outfreq=1000,
                 dim=2,
                 binwidth=0.1,
                 mu=False,
                 muout="muout",
                 temperature=300,
                 mass=np.nan,
                 zpos=None,
                 dens="mass",
                 groups=['all'],
                 comgroup=None,
                 center=False,
                 **kwargs):
        # Inherit all classes from AnalysisBase
        super(density_planar, self).__init__(atomgroup.universe.trajectory,
                                             **kwargs)

        self.atomgroup = atomgroup
        self.output = output
        self.outfreq = outfreq
        self.dim = dim
        self.binwidth = binwidth
        self.mu = mu
        self.muout = muout
        self.temperature = temperature
        self.mass = mass
        self.zpos = zpos
        self.dens = dens
        self.comgroup = comgroup
        self.center = center

        if not hasattr(groups, "__iter__") and type(groups) not in (str):
            self.groups = [groups]
        else:
            self.groups = groups

    def _configure_parser(self, parser):
        parser.description = self.__doc__
        parser.add_argument(
            '-o',
            dest='output',
            type=str,
            default='density',
            help='Prefix for output filenames')
        parser.add_argument(
            '-dout',
            dest='outfreq',
            type=float,
            default=1000,
            help='Default time after which output files are refreshed (1000 ps).'
        )
        parser.add_argument(
            '-d',
            dest='dim',
            type=int,
            default=2,
            help='dimension for binning (0=X, 1=Y, 2=Z)',
        )
        parser.add_argument(
            '-dz',
            dest='binwidth',
            type=float,
            default=0.1,
            help='binwidth (nanometer)')
        parser.add_argument(
            '-mu',
            dest='mu',
            default=False,
            action='store_true',
            help='Calculate the chemical potential')
        parser.add_argument(
            '-muo',
            dest='muout',
            type=str,
            default='dens',
            help='Prefix for output filename for chemical potential')
        parser.add_argument(
            '-temp',
            dest='temperature',
            type=float,
            default=300,
            help='temperature (K) for chemical potential')
        parser.add_argument(
            '-zpos',
            dest='zpos',
            type=float,
            default=None,
            help=
            'position at which the chemical potential will be computed. By default average over box.'
        )
        parser.add_argument(
            '-dens',
            dest='dens',
            type=str,
            default='mass',
            choices=["mass", "number", "charge", "temp"],
            help='Density')
        parser.add_argument(
            '-gr',
            dest='groups',
            type=str,
            default=['all'],
            nargs='+',
            help='Atoms for which to compute the density profile',
        )
        parser.add_argument(
            '-com',
            dest='comgroup',
            type=str,
            default=None,
            help=
            'Perform the binning relative to the center of mass of the selected group.'
        )
        parser.add_argument(
            '-center',
            dest='center',
            action='store_const',
            const=True,
            default=False,
            help=
            'Perform the binning relative to the center of the (changing) box.')

    def _prepare(self):
        if self._verbose:
            if self.dens == 'temp':
                print('Computing temperature profile along {}-axes.'.format(
                    'XYZ' [self.dim]))
            else:
                print('Computing {} density profile along {}-axes.'.format(
                    self.dens, 'XYZ' [self.dim]))

        self.ngroups = len(self.groups)
        self.nbins = int(
            np.ceil(self.atomgroup.universe.dimensions[self.dim] / 10 /
                    self.binwidth))

        self.density_mean = np.zeros((self.nbins, self.ngroups))
        self.density_mean_sq = np.zeros((self.nbins, self.ngroups))
        self.av_box_length = 0

        if self._verbose:
            print("\nCalcualate profile for the following group(s):")

        self.sel = []

        if self.mu and self.dens != 'mass':
            raise Exception(
                'Calculation of the chemical potential is only possible when mass density is selected'
            )

        if self.mu and len(self.groups) != 1:
            raise Exception(
                'Calculation of the chemical potential is supported for one group only'
            )

        for i, gr in enumerate(self.groups):
            sel = self.atomgroup.select_atoms(gr)
            if self._verbose:
                print("{:>15}: {:>10} atoms".format(gr, sel.n_atoms), end="")
            if sel.n_atoms > 0:
                self.sel.append(sel)
                if self.mu:
                    self.mass = sel.atoms.total_mass() / sel.atoms.n_residues
                print("")
            else:
                print(" - not taken for profile")

        if len(self.sel) == 0:
            raise RuntimeError(
                "No atoms found in selection. Please adjust group selection")

        if self.comgroup is not None:
            self.comsel = self.atomgroup.select_atoms(self.comgroup)
            if self._verbose:
                print("{:>15}: {:>10} atoms".format(self.comgroup,
                                                    self.comsel.n_atoms))
            if self.comsel.n_atoms == 0:
                raise ValueError(
                    "{} does not contain any atoms. Please adjust 'com' selection."
                    .format(gr))
        if self.comgroup is not None:
            self.center = True  # always center when COM
        if self._verbose:
            print("\n")
            print('Using', self.nbins, 'bins.')

    def weight(self, selection):
        """Calculates the weights for the histogram depending on the choosen type of density."""
        if self.dens == "mass":
            # amu in kg -> kg/m^3
            return selection.atoms.masses * 1.66053892
        elif self.dens == "number":
            return np.ones(selection.atoms.n_atoms)
        elif self.dens == "charge":
            return selection.atoms.charges
        elif self.dens == "temp":
            # ((1 amu * (Angstrom^2)) / (picoseconds^2)) / Boltzmann constant = 1.20272362 Kelvin
            return ((selection.atoms.velocities**2).sum(axis=1) *
                    selection.atoms.masses / 2 * 1.20272362)

    def _single_frame(self):
        curV = self._ts.volume / 1000
        self.av_box_length += self._ts.dimensions[self.dim] / 10
        """ center of mass calculation with generalization to periodic systems
        see Bai, Linge; Breen, David (2008). "Calculating Center of Mass in an
        Unbounded 2D Environment". Journal of Graphics, GPU, and Game Tools. 13
        (4): 53–60. doi:10.1080/2151237X.2008.10129266,
        https://en.wikipedia.org/wiki/Center_of_mass#Systems_with_periodic_boundary_conditions
        """
        if self.comgroup is None:
            comshift = 0
        else:
            Theta = self.comsel.positions[:, self.dim] / self._ts.dimensions[
                self.dim] * 2 * np.pi
            Xi = (np.cos(Theta) *
                  self.comsel.masses).sum() / self.comsel.masses.sum()
            Zeta = (np.sin(Theta) *
                    self.comsel.masses).sum() / self.comsel.masses.sum()
            ThetaCOM = np.arctan2(-Zeta, -Xi) + np.pi
            comshift = self._ts.dimensions[self.dim] * (0.5 - ThetaCOM /
                                                        (2 * np.pi))

        dz = self._ts.dimensions[self.dim] / self.nbins

        for index, selection in enumerate(self.sel):
            bins = ((selection.atoms.positions[:, self.dim] + comshift + dz / 2)
                    / dz).astype(int) % self.nbins
            density_ts = np.histogram(
                bins,
                bins=np.arange(self.nbins + 1),
                weights=self.weight(selection))[0]

            bincount = np.bincount(bins, minlength=self.nbins)

            if self.dens == 'temp':
                self.density_mean[:, index] += density_ts / bincount
                self.density_mean_sq[:, index] += (density_ts / bincount)**2
            else:
                self.density_mean[:, index] += density_ts / curV * self.nbins
                self.density_mean_sq[:, index] += (
                    density_ts / curV * self.nbins)**2

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

        dz = self.av_box_length / (self._index * self.nbins)
        if self.center:
            self.results["z"] = np.linspace(
                -self.av_box_length / self._index / 2,
                self.av_box_length / self._index / 2,
                self.nbins,
                endpoint=False) + dz / 2
        else:
            self.results["z"] = np.linspace(
                0, self.av_box_length / self._index, self.nbins,
                endpoint=False) + dz / 2

        # chemical potential
        if self.mu:
            if (self.zpos != None):
                this = (self.zpos / (self.av_box_length / self._index) *
                        self.nbins).astype(int)
                self.results["mu"] = mu(self.results["dens_mean"][this],
                                        self.temperature, self.mass)
                self.results["dmu"] = dmu(self.results["dens_mean"][this],
                                          self.results["dens_err"][this],
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
            self._index * self.atomgroup.universe.trajectory.dt)
        for group in self.groups:
            columns += "\t" + group
        for group in self.groups:
            columns += "\t" + group + " error"

        # save density profile
        savetxt(
            self.output + '.dat',
            np.hstack(((self.results["z"][:, np.newaxis]),
                       self.results["dens_mean"], self.results["dens_err"])),
            header=columns)

        if self.mu:
            # save chemical potential
            savetxt(self.muout + '.dat',
                    np.hstack((self.results["mu"], self.results["dmu"]))[None])
