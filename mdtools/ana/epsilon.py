#!/usr/bin/env python
# coding: utf8

from __future__ import absolute_import, division, print_function

import numpy as np
from MDAnalysis.units import constants, convert

from .base import AnalysisBase
from ..utils import repairMolecules

constants["Boltzman_constant"] = 8.314462159e-3
constants["electric_constant"] = 5.526350e-3

class epsilon_bulk(AnalysisBase):
    """Computes the dipole moment flcutuations and from this the
    dielectric constant. 
    For group selections use strings in the MDAnalysis selection command style"""

    def __init__(self, atomgroup, sel="all", outfreq=100, temperature=300,
                 bpbc=True, output="eps", **kwargs):
        super(epsilon_bulk, self).__init__(atomgroup.universe.trajectory,
                                           **kwargs)
        self.atomgroup = atomgroup
        self.sel = sel
        self.outfreq = 100
        self.temperature = temperature
        self.bpbc = bpbc
        self.output = output
        
    def _configure_parser(self, parser):
        parser.description = self.__doc__
        parser.add_argument('-sel', dest='sel', type=str, default='all',
                            help='Atoms for which to compute the profile')
        parser.add_argument('-dout', dest='outfreq', type=float, default=100,
                            help='Number of frames after which the output is updated.')
        parser.add_argument('-o', dest='output', type=str, default='eps',
                            help='Prefix for output filenames')
        parser.add_argument('-temp', dest='temperature', type=float, default=300,
                            help='temperature (K)')
        parser.add_argument('-nopbcrepair', dest='bpbc', action='store_false',
                            help='do not make broken molecules whole again '
                            + '(only works if molecule is smaller than shortest box vector')

    def _prepare(self):
        self.volume = 0
        self.M = np.zeros(3)
        self.M2 = np.zeros(3)

        self.selection = self.atomgroup.select_atoms(self.sel)
        self.charges = self.selection.charges

        if self._verbose:
            print("There are {} atoms in the selection '{}'.".format(
                self.selection.n_atoms, self.sel))

    def _single_frame(self):
        # Make molecules whole
        if self.bpbc:
            repairMolecules(self.selection)

        self.volume += self.selection.universe.trajectory.ts.volume

        M = np.dot(self.charges, self.selection.positions)
        self.M += M
        self.M2 += M * M

        if self._save and self._frame_index % self.outfreq == 0 and self._frame_index > 0:
            self._calculate_results()
            self._save_results()

    def _calculate_results(self):
        index = self._frame_index + 1

        self.results["M"] = self.M / index
        self.results["M2"] = self.M2 / index
        self.results["volume"] = self.volume / index
        self.results["fluct"] = self.results["M2"] - self.results["M"]**2
        self.results["eps"] = 1 + self.results["fluct"] / (
            convert(constants["Boltzman_constant"], "kJ/mol", "eV")
            * self.temperature * self.results["volume"] * constants["electric_constant"])
        self.results["eps_mean"] = 1 + self.results["fluct"].mean() / (
            convert(constants["Boltzman_constant"], "kJ/mol", "eV")
            * self.temperature * self.results["volume"] * constants["electric_constant"])

    def _conclude(self):
        if self._verbose:
            print(
                "The following averages for the complete trajectory have been calculated:")

            print("")
            for i, d in enumerate("xyz"):
                print(" <M_{}> = {:.4f} eÅ".format(d, self.results["M"][i]))

            print("")
            for i, d in enumerate("xyz"):
                print(" <M_{}²> = {:.4f} (eÅ)²".format(
                    d, self.results["M2"][i]))

            print("")
            print(" <|M|²> = {:.4f} (eÅ)²".format(self.results["M2"].mean()))
            print(" |<M>|² = {:.4f} (eÅ)²".format(
                (self.results["M"]**2).mean()))

            print("")
            print(
                " <|M|²> - |<M>|² = {:.4f} (eÅ)²".format(self.results["fluct"].mean()))

            print("")
            for i, d in enumerate("xyz"):
                print(" ε_{} = {:.2f} ".format(d, self.results["eps"][i]))

            print("")
            print(" ε = {:.2f}".format(self.results["eps_mean"]))
            print("")

    def _save_results(self):
        np.savetxt(self.output + '.dat',
                   np.hstack([self.results["eps_mean"],
                              self.results["eps"]]).T,
                   fmt='%1.2f', header='eps\teps_x\teps_y\teps_z')
                   
class epsilon_planar(AnalysisBase):
    """Calculate the dielectric profile. See Bonthuis et. al., Langmuir 28, vol. 20 (2012) for details."""

    def __init__(self, atomgroup, output="eps", binwidth=0.05, dim=2, zmin=0,
                 zmax=-1, temperature=300, groups=['resname SOL', 'not resname SOL'],
                 outfreq=10000, b2d=True, vac=False, membrane_shift=False, com=False,
                 bpbc=True, **kwself):

        # Inherit all classes from AnalysisBase
        super(epsilon_planar, self).__init__(atomgroup.universe.trajectory,
                                             **kwself)

        self.atomgroup = atomgroup
        self.output = output
        self.binwidth = binwidth
        self.dim = dim
        self.zmin = zmin
        self.zmax = zmax
        self.temperature = temperature
        self.groups = groups
        self.outfreq = outfreq
        self.b2d = b2d
        self.vac = vac
        self.membrane_shift = membrane_shift
        self.com = com
        self.bpbc = bpbc
    
    def _configure_parser(self, parser):
        parser.description = self.__doc__
        parser.add_argument('-dz', dest='binwidth', type=float, default=0.05,
                            help='specify the binwidth [nm]')
        parser.add_argument('-d', dest='dim', type=int, default=2,
                            help='direction normal to the surface (x,y,z=0,1,2, default: z)')
        parser.add_argument('-zmin', dest='zmin', type=float, default=0,
                            help='minimal z-coordinate for evaluation [nm]')
        parser.add_argument('-zmax', dest='zmax', type=float, default=-1,
                            help='maximal z-coordinate for evaluation [nm]')
        parser.add_argument('-temp', dest='temperature', type=float, default=300,
                            help='temperature [K]')
        parser.add_argument('-o', dest='output', type=str, default='eps',
                            help='Prefix for output filenames')
        parser.add_argument('-groups', dest='groups', type=str, nargs='+',
                            default=['resname SOL', 'not resname SOL'],
                            help='atom group selection')
        parser.add_argument('-dout', dest='outfreq', type=float, default=10000,
                            help='Default number of frames after which output files are refreshed (10000)')
        parser.add_argument('-2d', dest='b2d', action='store_const', const=True, default=False,
                            help='use 2d slab geometry')
        parser.add_argument('-vac', dest='vac', action='store_const', const=True, default=False,
                            help='use vacuum boundary conditions instead of metallic (2D only!).')
        parser.add_argument('-sym', dest='bsym', action='store_const', const=True, default=False,
                            help='symmetrize the profiles')
        parser.add_argument('-shift', dest='membrane_shift', action='store_const',
                            const=True, default=False,
                            help='shift system by half a box length (useful for membrane simulations)')
        parser.add_argument('-com', dest='com', action='store_const', const=True, default=False,
                            help='shift system such that the water COM is centered')
        parser.add_argument('-nopbcrepair', dest='bpbc', action='store_false',
                            help='do not make broken molecules whole again (only works if molecule is smaller than shortest box vector')
    
    def _prepare(self):
        if self._verbose:
            print("\nCalcualate profile for the following group(s):")

        self.mysels = []
        for i, gr in enumerate(self.groups):
            self.mysels.append(self.atomgroup.select_atoms(gr))
            if self._verbose:
                print("{:>15}: {:>10} atoms".format(
                    gr, self.mysels[i].n_atoms))
            if self.mysels[i].n_atoms == 0:
                raise RuntimeError(
                    "\n '{}' does not contain any atoms. Please adjust group selection.".format(gr))

        print("\n")

        self.sol = self.atomgroup.select_atoms('resname SOL')

        # Assume a threedimensional universe...
        self.xydims = np.roll(np.arange(3), -self.dim)[1:]
        dz = self.binwidth * 10  # Convert to Angstroms

        if (self.zmax == -1):
            self.zmax = self.atomgroup.universe.dimensions[self.dim]
        else:
            self.zmax *= 10

        self.zmin *= 10
        # CAVE: binwidth varies in NPT !
        self.nbins = int((self.zmax - self.zmin) / dz)

        # Use 10 hardoced blocks for resampling
        self.resample = 10
        self.resample_freq = int(
            np.ceil((self.stop - self.start) / self.resample))

        self.V = 0
        self.Lz = 0
        self.A = np.prod(self.atomgroup.universe.dimensions[self.xydims])

        self.m_par = np.zeros((self.nbins, len(self.groups), self.resample))
        self.mM_par = np.zeros((self.nbins, len(self.groups), self.resample)
                               )  # total fluctuations
        self.mm_par = np.zeros((self.nbins, len(self.groups)))  # self
        self.cmM_par = np.zeros((self.nbins, len(self.groups))
                                )  # collective contribution
        self.cM_par = np.zeros((self.nbins, len(self.groups)))
        self.M_par = np.zeros((self.resample))

        # Same for perpendicular
        self.m_perp = np.zeros((self.nbins, len(self.groups), self.resample))
        self.mM_perp = np.zeros((self.nbins, len(self.groups), self.resample)
                                )  # total fluctuations
        self.mm_perp = np.zeros((self.nbins, len(self.groups)))  # self
        self.cmM_perp = np.zeros((self.nbins, len(self.groups))
                                 )  # collective contribution
        self.cM_perp = np.zeros((self.nbins, len(self.groups))
                                )  # collective contribution
        self.M_perp = np.zeros((self.resample))
        self.M_perp_2 = np.zeros((self.resample))

        if self._verbose:
            print('Using', self.nbins, 'bins.')

    def _single_frame(self):

        if (self.zmax == -1):
            zmax = self._self._ts.dimensions[self.dim]
        else:
            zmax = self.zmax

        if self.membrane_shift:
            # shift membrane
            self._ts.positions[:,self.dim] += self._ts.dimensions[self.dim] / 2
            self._ts.positions[:, self.dim] %= self._ts.dimensions[self.dim]
        if self.com:
            # put water COM into center
            waterCOM = np.sum(
                self.sol.atoms.positions[:, 2] * self.sol.atoms.masses) / self.sol.atoms.masses.sum()
            if self._verbose:
                print("shifting by ", waterCOM)
            self._ts.positions[:,self.dim] += self._ts.dimensions[self.dim] / 2 - waterCOM
            self._ts.positions[:,self.dim] %= self._ts.dimensions[self.dim]

        if self.bpbc:
            # make broken molecules whole again!
            repairMolecules(self.atomgroup)

        dz_frame = self._ts.dimensions[self.dim] / self.nbins

        # precalculate total polarization of the box
        this_M_perp, this_M_par = np.split(
            np.roll(np.dot(self.atomgroup.charges, self.atomgroup.positions), -self.dim), [1])

        # Use polarization density ( for perpendicular component )
        # ========================================================

        # sum up the averages
        self.M_perp[self._frame_index // self.resample_freq] += this_M_perp
        self.M_perp_2[self._frame_index // self.resample_freq] += this_M_perp**2
        for i, sel in enumerate(self.mysels):
            bins = ((sel.atoms.positions[:, self.dim] - self.zmin)
                    / ((zmax - self.zmin) / (self.nbins))).astype(int)
            bins[np.where(bins < 0)] = 0  # put all charges back inside box
            bins[np.where(bins > self.nbins)] = self.nbins
            curQ = np.histogram(bins, bins=np.arange(
                self.nbins + 1), weights=sel.atoms.charges)[0]
            this_m_perp = -np.cumsum(curQ / self.A)
            self.m_perp[:, i, self._frame_index // self.resample_freq] += this_m_perp
            self.mM_perp[:, i, self._frame_index
                         // self.resample_freq] += this_m_perp * this_M_perp
            self.mm_perp[:, i] += this_m_perp * this_m_perp * \
                (self._ts.dimensions[self.dim] / self.nbins) * self.A  # self term
            # collective contribution
            self.cmM_perp[:, i] += this_m_perp * \
                (this_M_perp - this_m_perp * (self.A * dz_frame))
            self.cM_perp[:, i] += this_M_perp - this_m_perp * self.A * dz_frame

        # Use virtual cutting method ( for parallel component )
        # ========================================================
        nbinsx=250  # number of virtual cuts ("many")

        for i, sel in enumerate(self.mysels):
            # Move all z-positions to 'center of charge' such that we avoid monopoles in z-direction
            # (compare Eq. 33 in Bonthuis 2012; we only want to cut in x/y direction)
            chargepos=sel.atoms.positions * \
                np.abs(sel.atoms.charges[:, np.newaxis])
            atomsPerMolecule=sel.n_atoms // sel.n_residues
            centers=sum(chargepos[i::atomsPerMolecule] for i in range(atomsPerMolecule)) \
                / np.abs(sel.residues[0].atoms.charges).sum()
            testpos=sel.atoms.positions
            testpos[:, self.dim]=np.repeat(
                centers[:, self.dim], atomsPerMolecule)
            binsz = (((testpos[:, self.dim] - self.zmin) % self._ts.dimensions[self.dim]) /
                     ((zmax - self.zmin) / self.nbins)).astype(int)

            # Average parallel directions
            for j, direction in enumerate(self.xydims):
                dx = self._ts.dimensions[direction] / nbinsx
                binsx = (sel.atoms.positions[:, direction] /
                         (self._ts.dimensions[direction] / nbinsx)).astype(int)
                # put all charges back inside box
                binsx[np.where(binsx < 0)] = 0
                binsx[np.where(binsx > nbinsx)] = nbinsx
                curQx = np.histogram2d(binsz, binsx, bins=[np.arange(0, self.nbins + 1), np.arange(0, nbinsx + 1)],
                                       weights=sel.atoms.charges)[0]
                curqx = np.cumsum(curQx, axis=1) / (self._ts.dimensions[self.xydims[1 - j]] * (
                    self._ts.dimensions[self.dim] / self.nbins))  # integral over x, so uniself._ts of area
                this_m_par = -curqx.mean(axis=1)

                self.m_par[:, i, self._frame_index //
                           self.resample_freq] += this_m_par
                self.mM_par[:, i, self._frame_index // self.resample_freq] += this_m_par * \
                    this_M_par[j]
                self.M_par[self._frame_index // self.resample_freq] += this_M_par[j]
                self.mm_par[:, i] += this_m_par * \
                    this_m_par * dz_frame * self.A
                # collective contribution
                self.cmM_par[:, i] += this_m_par * \
                    (this_M_par[j] - this_m_par * dz_frame * self.A)
                self.cM_par[:, i] += this_M_par[j] - \
                    this_m_par * dz_frame * self.A

        self.V += self._ts.volume
        self.Lz += self._ts.dimensions[self.dim]

        if self._save and self._frame_index % self.outfreq == 0 and self._frame_index > 0:
            self._calculate_results()
            self._save_results()

    def _calculate_results(self):
        self._index = self._frame_index + 1

        self.results["V"] = self.V / self._index

        cov_perp = self.mM_perp.sum(axis=2) / self._index - \
            self.m_perp.sum(axis=2) / self._index * \
            self.M_perp.sum() / self._index
        dcov_perp = np.sqrt((self.mM_perp.std(axis=2) / self._index * self.resample)**2
                            + (self.m_perp.std(axis=2) / self._index
                             * self.resample * self.M_perp.sum() / self._index)**2 +
                            (self.m_perp.sum(axis=2) / self._index * self.M_perp.std() \
                               / self._index * self.resample)**2) / np.sqrt(self.resample - 1)
        cov_perp_self = self.mm_perp / self._index - \
            (self.m_perp.sum(axis=2) / self._index * self.m_perp.sum(axis=2)
             / self._index * self.A * self.Lz / self.nbins / self._index)
        cov_perp_coll = self.cmM_perp / self._index - \
            self.m_perp.sum(axis=2) / self._index * self.cM_perp / self._index

        var_perp = self.M_perp_2.sum() / self._index - (self.M_perp.sum() / self._index)**2
        dvar_perp = (self.M_perp_2 / self._index - (self.M_perp / self._index)**2).std() \
            / np.sqrt(self.resample - 1)

        cov_par = self.mM_par.sum(axis=2) / self._index - \
            self.m_par.sum(axis=2) / self._index * \
            self.M_par.sum() / self._index
        cov_par_self = self.mm_par / self._index - \
            self.m_par.sum(axis=2) / self._index * (self.m_par.sum(axis=2) *
                                                    self.Lz / self.nbins / self._index * self.A) / self._index
        cov_par_coll = self.cmM_par / self._index - \
            self.m_par.sum(axis=2) / self._index * self.cM_par / self._index
        dcov_par = np.sqrt((self.mM_par.std(axis=2) / self._index * self.resample)**2
                           + (self.m_par.std(axis=2) / self._index
                              * self.resample * self.M_par.sum() / self._index)**2
                           + (self.m_par.sum(axis=2) / self._index * self.M_par.std() / \
                            self._index * self.resample)**2) / np.sqrt(self.resample - 1)

        eps0inv = 1. / 8.854e-12
        pref = (1.6e-19)**2 / 1e-10
        kB = 1.3806488e-23
        beta = 1. / (kB * self.temperature)

        self.results["eps_par"] = beta * eps0inv * pref / 2 * cov_par
        self.results["deps_par"] = beta * eps0inv * pref / 2 * dcov_par
        self.results["eps_par_self"] = beta * eps0inv * pref / 2 * cov_par_self
        self.results["eps_par_coll"] = beta * eps0inv * pref / 2 * cov_par_coll

        if (self.b2d):
            self.results["eps_perp"] = - beta * eps0inv * pref * cov_perp
            self.results["eps_perp_self"] = - beta * eps0inv * pref * cov_perp_self
            self.results["eps_perp_coll"] = - beta * eps0inv * pref * cov_perp_coll
            self.results["deps_perp"] = np.abs(- eps0inv * beta * pref) * dcov_perp
            if (self.vac):
                self.results["eps_perp"] *= 2. / 3.
                self.results["eps_perp_self"] *= 2. / 3.
                self.results["eps_perp_coll"] *= 2. / 3.
                self.results["deps_perp"] *= 2. / 3.

        else:
            self.results["eps_perp"] = (- eps0inv * beta * pref * cov_perp) \
                / (1 + eps0inv * beta * pref / self.results["V"] * var_perp)
            self.results["deps_perp"] = np.abs((- eps0inv * beta * pref) /
                                          (1 + eps0inv * beta * pref / self.results["V"] * var_perp)) * dcov_perp \
                + np.abs((- eps0inv * beta * pref * cov_perp) /
                         (1 + eps0inv * beta * pref / self.results["V"] * var_perp)**2) * dvar_perp

            self.results["eps_perp_self"] = (- eps0inv * beta * pref * cov_perp_self) \
                / (1 + eps0inv * beta * pref / self.results["V"] * var_perp)
            self.results["eps_perp_coll"] = (- eps0inv * beta * pref * cov_perp_coll) \
                / (1 + eps0inv * beta * pref / self.results["V"] * var_perp)

            if (self.zmax == -1):
                self.results["z"] = np.linspace(
                    self.zmin, self.Lz / self._index, len(self.results["eps_par"])) / 10
            else:
                self.results["z"] = np.linspace(
                    self.zmin, self.zmax, len(self.results["eps_par"])) / 10.

    def _save_results(self):


        outdata_perp = np.hstack([self.results["z"][:, np.newaxis], 
                                  self.results["eps_perp"].sum(axis=1)[:, np.newaxis], 
                                  self.results["eps_perp"],
                                  np.linalg.norm(self.results["deps_perp"], axis=1)[:, np.newaxis], 
                                  self.results["deps_perp"],
                                  self.results["eps_perp_self"].sum(axis=1)[:, np.newaxis], 
                                  self.results["eps_perp_coll"].sum(axis=1)[:, np.newaxis],
                                  self.results["eps_perp_self"], 
                                  self.results["eps_perp_coll"]])

        outdata_par = np.hstack([self.results["z"][:, np.newaxis], 
                                 self.results["eps_par"].sum(axis=1)[:, np.newaxis], 
                                 self.results["eps_par"],
                                 np.linalg.norm(self.results["deps_par"], axis=1)[:, np.newaxis], 
                                 self.results["deps_par"],
                                 self.results["eps_par_self"].sum(axis=1)[:, np.newaxis], 
                                 self.results["eps_par_coll"].sum(axis=1)[:, np.newaxis],
                                 self.results["eps_par_self"], 
                                 self.results["eps_par_coll"]])

        if (self.bsym):
            for i in range(len(outdata_par) - 1):
                outdata_par[i + 1] = .5 * \
                    (outdata_par[i + 1] + outdata_par[i + 1][-1::-1])
            for i in range(len(outdata_perp) - 1):
                outdata_perp[i + 1] = .5 * \
                    (outdata_perp[i + 1] + outdata_perp[i + 1][-1::-1])

        header = "statistics over {:.1f} picoseconds".format(
                            self._index * self.atomgroup.universe.trajectory.dt)
        np.savetxt(self.output + '_perp.dat', outdata_perp, header=header)
        np.savetxt(self.output + '_par.dat', outdata_par, header=header)

