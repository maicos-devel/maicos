#!/usr/bin/env python
# coding: utf8

import numpy as np
import scipy.constants

from .base import SingleGroupAnalysisBase, MultiGroupAnalysisBase
from ..utils import FT, iFT, repairMolecules, savetxt

eps0inv = 1. / scipy.constants.epsilon_0
pref = (scipy.constants.elementary_charge)**2 / 1e-10


def Bin(a, bins):
    """Averages array values in bins for easier plotting.
    Note: "bins" array should contain the INDEX (integer) where that bin begins"""

    if np.iscomplex(a).any():
        avg = np.zeros(len(bins), dtype=complex)  # average of data
    else:
        avg = np.zeros(len(bins))

    count = np.zeros(len(bins), dtype=int)
    ic = -1

    for i in range(0, len(a)):
        if i in bins:
            ic += 1  # index for new average
        avg[ic] += a[i]
        count[ic] += 1

    return avg / count


class epsilon_bulk(SingleGroupAnalysisBase):
    """Computes the dipole moment flcutuations and from this the
    dielectric constant.
    For group selections use strings in the MDAnalysis selection command style"""

    def __init__(self,
                 atomgroup,
                 outfreq=100,
                 temperature=300,
                 bpbc=True,
                 output="eps",
                 **kwargs):
        super(epsilon_bulk, self).__init__(atomgroup, **kwargs)
        self.outfreq = 100
        self.temperature = temperature
        self.bpbc = bpbc
        self.output = output

    def _configure_parser(self, parser):
        parser.description = self.__doc__
        parser.add_argument(
            '-dout',
            dest='outfreq',
            type=float,
            default=100,
            help='Number of frames after which the output is updated.')
        parser.add_argument(
            '-o',
            dest='output',
            type=str,
            default='eps',
            help='Prefix for output filenames')
        parser.add_argument(
            '-temp',
            dest='temperature',
            type=float,
            default=300,
            help='temperature (K)')
        parser.add_argument(
            '-nopbcrepair',
            dest='bpbc',
            action='store_false',
            help='do not make broken molecules whole again ' +
            '(only works if molecule is smaller than shortest box vector')

    def _prepare(self):
        self.volume = 0
        self.M = np.zeros(3)
        self.M2 = np.zeros(3)
        self.charges = self.atomgroup.charges

    def _single_frame(self):
        # Make molecules whole
        if self.bpbc:
            repairMolecules(self.atomgroup)

        self.volume += self._ts.volume

        M = np.dot(self.charges, self.atomgroup.positions)
        self.M += M
        self.M2 += M * M

        if self._save and self._frame_index % self.outfreq == 0 and self._frame_index > 0:
            self._calculate_results()
            self._save_results()

    def _calculate_results(self):
        index = self._frame_index + 1
        beta = 1. / (scipy.constants.Boltzmann * self.temperature)

        self.results["M"] = self.M / index
        self.results["M2"] = self.M2 / index
        self.results["volume"] = self.volume / index
        self.results["fluct"] = self.results["M2"] - self.results["M"]**2
        self.results["eps"] = beta * eps0inv * pref * self.results["fluct"] / \
                              self.results["volume"]
        self.results["eps_mean"] = self.results["eps"].mean()

        self.results["eps"] += 1
        self.results["eps_mean"] += 1

    def _conclude(self):
        if self._verbose:
            print(
                "The following averages for the complete trajectory have been calculated:"
            )

            print("")
            for i, d in enumerate("xyz"):
                print(" <M_{}> = {:.4f} eÅ".format(d, self.results["M"][i]))

            print("")
            for i, d in enumerate("xyz"):
                print(" <M_{}²> = {:.4f} (eÅ)²".format(d,
                                                       self.results["M2"][i]))

            print("")
            print(" <|M|²> = {:.4f} (eÅ)²".format(self.results["M2"].mean()))
            print(" |<M>|² = {:.4f} (eÅ)²".format(
                (self.results["M"]**2).mean()))

            print("")
            print(" <|M|²> - |<M>|² = {:.4f} (eÅ)²".format(
                self.results["fluct"].mean()))

            print("")
            for i, d in enumerate("xyz"):
                print(" ε_{} = {:.2f} ".format(d, self.results["eps"][i]))

            print("")
            print(" ε = {:.2f}".format(self.results["eps_mean"]))
            print("")

    def _save_results(self):
        savetxt(
            self.output + '.dat',
            np.hstack([self.results["eps_mean"], self.results["eps"]]).T,
            fmt='%1.2f',
            header='eps\teps_x\teps_y\teps_z')


class epsilon_planar(MultiGroupAnalysisBase):
    """Calculate the dielectric profile. See Bonthuis et. al., Langmuir 28, vol. 20 (2012) for details."""

    def __init__(self,
                 atomgroups,
                 output="eps",
                 binwidth=0.05,
                 dim=2,
                 zmin=0,
                 zmax=-1,
                 temperature=300,
                 outfreq=10000,
                 b2d=False,
                 bsym=False,
                 vac=False,
                 membrane_shift=False,
                 com=False,
                 bpbc=True,
                 **kwself):

        # Inherit all classes from AnalysisBase
        super(epsilon_planar, self).__init__(atomgroups, **kwself)
        self.output = output
        self.binwidth = binwidth
        self.dim = dim
        self.zmin = zmin
        self.zmax = zmax
        self.temperature = temperature
        self.outfreq = outfreq
        self.b2d = b2d
        self.bsym = bsym
        self.vac = vac
        self.membrane_shift = membrane_shift
        self.com = com
        self.bpbc = bpbc

    def _configure_parser(self, parser):
        parser.description = self.__doc__
        parser.add_argument(
            '-dz',
            dest='binwidth',
            type=float,
            default=0.05,
            help='specify the binwidth [nm]')
        parser.add_argument(
            '-d',
            dest='dim',
            type=int,
            default=2,
            help='direction normal to the surface (x,y,z=0,1,2, default: z)')
        parser.add_argument(
            '-zmin',
            dest='zmin',
            type=float,
            default=0,
            help='minimal z-coordinate for evaluation [nm]')
        parser.add_argument(
            '-zmax',
            dest='zmax',
            type=float,
            default=-1,
            help='maximal z-coordinate for evaluation [nm]')
        parser.add_argument(
            '-temp',
            dest='temperature',
            type=float,
            default=300,
            help='temperature [K]')
        parser.add_argument(
            '-o',
            dest='output',
            type=str,
            default='eps',
            help='Prefix for output filenames')
        parser.add_argument(
            '-dout',
            dest='outfreq',
            type=float,
            default=10000,
            help=
            'Default number of frames after which output files are refreshed (10000)'
        )
        parser.add_argument(
            '-2d',
            dest='b2d',
            action='store_const',
            const=True,
            default=False,
            help='use 2d slab geometry')
        parser.add_argument(
            '-vac',
            dest='vac',
            action='store_const',
            const=True,
            default=False,
            help='use vacuum boundary conditions instead of metallic (2D only!).'
        )
        parser.add_argument(
            '-sym',
            dest='bsym',
            action='store_const',
            const=True,
            default=False,
            help='symmetrize the profiles')
        parser.add_argument(
            '-shift',
            dest='membrane_shift',
            action='store_const',
            const=True,
            default=False,
            help=
            'shift system by half a box length (useful for membrane simulations)'
        )
        parser.add_argument(
            '-com',
            dest='com',
            action='store_const',
            const=True,
            default=False,
            help='shift system such that the water COM is centered')
        parser.add_argument(
            '-nopbcrepair',
            dest='bpbc',
            action='store_false',
            help=
            'do not make broken molecules whole again (only works if molecule is smaller than shortest box vector'
        )

    def _prepare(self):
        if self._verbose:
            print("\nCalcualate profile for the following group(s):")

        self.sol = self._universe.select_atoms('resname SOL')

        # Assume a threedimensional universe...
        self.xydims = np.roll(np.arange(3), -self.dim)[1:]
        dz = self.binwidth * 10  # Convert to Angstroms

        if (self.zmax == -1):
            self.zmax = self._universe.dimensions[self.dim]
        else:
            self.zmax *= 10

        self.zmin *= 10
        # CAVE: binwidth varies in NPT !
        self.nbins = int((self.zmax - self.zmin) / dz)

        # Use 10 hardoced blocks for resampling
        self.resample = 10
        self.resample_freq = int(
            np.ceil((self.stopframe - self.startframe) / self.resample))

        self.V = 0
        self.Lz = 0
        self.A = np.prod(self._universe.dimensions[self.xydims])

        self.m_par = np.zeros((self.nbins, len(self.atomgroups), self.resample))
        self.mM_par = np.zeros((self.nbins, len(self.atomgroups),
                                self.resample))  # total fluctuations
        self.mm_par = np.zeros((self.nbins, len(self.atomgroups)))  # self
        self.cmM_par = np.zeros(
            (self.nbins, len(self.atomgroups)))  # collective contribution
        self.cM_par = np.zeros((self.nbins, len(self.atomgroups)))
        self.M_par = np.zeros((self.resample))

        # Same for perpendicular
        self.m_perp = np.zeros((self.nbins, len(self.atomgroups),
                                self.resample))
        self.mM_perp = np.zeros((self.nbins, len(self.atomgroups),
                                 self.resample))  # total fluctuations
        self.mm_perp = np.zeros((self.nbins, len(self.atomgroups)))  # self
        self.cmM_perp = np.zeros(
            (self.nbins, len(self.atomgroups)))  # collective contribution
        self.cM_perp = np.zeros(
            (self.nbins, len(self.atomgroups)))  # collective contribution
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
            self._ts.positions[:, self.dim] += self._ts.dimensions[self.dim] / 2
            self._ts.positions[:, self.dim] %= self._ts.dimensions[self.dim]
        if self.com:
            # put water COM into center
            waterCOM = np.sum(
                self.sol.atoms.positions[:, 2] *
                self.sol.atoms.masses) / self.sol.atoms.masses.sum()
            if self._verbose:
                print("shifting by ", waterCOM)
            self._ts.positions[:, self.dim] += self._ts.dimensions[
                self.dim] / 2 - waterCOM
            self._ts.positions[:, self.dim] %= self._ts.dimensions[self.dim]

        if self.bpbc:
            # make broken molecules whole again!
            repairMolecules(self._universe.atoms)

        dz_frame = self._ts.dimensions[self.dim] / self.nbins

        # precalculate total polarization of the box
        this_M_perp, this_M_par = np.split(
            np.roll(
                np.dot(self._universe.atoms.charges,
                       self._universe.atoms.positions), -self.dim), [1])

        # Use polarization density ( for perpendicular component )
        # ========================================================

        # sum up the averages
        self.M_perp[self._frame_index // self.resample_freq] += this_M_perp
        self.M_perp_2[self._frame_index // self.resample_freq] += this_M_perp**2
        for i, sel in enumerate(self.atomgroups):
            bins = ((sel.atoms.positions[:, self.dim] - self.zmin) / (
                (zmax - self.zmin) / (self.nbins))).astype(int)
            bins[np.where(bins < 0)] = 0  # put all charges back inside box
            bins[np.where(bins >= self.nbins)] = self.nbins - 1
            curQ = np.histogram(
                bins, bins=np.arange(self.nbins + 1),
                weights=sel.atoms.charges)[0]
            this_m_perp = -np.cumsum(curQ / self.A)
            self.m_perp[:, i, self._frame_index //
                        self.resample_freq] += this_m_perp
            self.mM_perp[:, i, self._frame_index //
                         self.resample_freq] += this_m_perp * this_M_perp
            self.mm_perp[:, i] += this_m_perp * this_m_perp * \
                (self._ts.dimensions[self.dim] / self.nbins) * self.A  # self term
            # collective contribution
            self.cmM_perp[:, i] += this_m_perp * \
                (this_M_perp - this_m_perp * (self.A * dz_frame))
            self.cM_perp[:, i] += this_M_perp - this_m_perp * self.A * dz_frame

        # Use virtual cutting method ( for parallel component )
        # ========================================================
        nbinsx = 250  # number of virtual cuts ("many")

        for i, sel in enumerate(self.atomgroups):
            # Move all z-positions to 'center of charge' such that we avoid monopoles in z-direction
            # (compare Eq. 33 in Bonthuis 2012; we only want to cut in x/y direction)
            chargepos = sel.atoms.positions * \
                np.abs(sel.atoms.charges[:, np.newaxis])
            atomsPerMolecule = sel.n_atoms // sel.n_residues
            centers = sum(chargepos[i::atomsPerMolecule] for i in range(atomsPerMolecule)) \
                / np.abs(sel.residues[0].atoms.charges).sum()
            testpos = sel.atoms.positions
            testpos[:, self.dim] = np.repeat(centers[:, self.dim],
                                             atomsPerMolecule)
            binsz = (((testpos[:, self.dim] - self.zmin) %
                      self._ts.dimensions[self.dim]) / (
                          (zmax - self.zmin) / self.nbins)).astype(int)

            # Average parallel directions
            for j, direction in enumerate(self.xydims):
                binsx = (sel.atoms.positions[:, direction] /
                         (self._ts.dimensions[direction] / nbinsx)).astype(int)
                # put all charges back inside box
                binsx[np.where(binsx < 0)] = 0
                binsx[np.where(binsx >= nbinsx)] = nbinsx - 1
                curQx = np.histogram2d(
                    binsz,
                    binsx,
                    bins=[
                        np.arange(0, self.nbins + 1),
                        np.arange(0, nbinsx + 1)
                    ],
                    weights=sel.atoms.charges)[0]
                curqx = np.cumsum(
                    curQx,
                    axis=1) / (self._ts.dimensions[self.xydims[1 - j]] *
                               (self._ts.dimensions[self.dim] / self.nbins)
                              )  # integral over x, so uniself._ts of area
                this_m_par = -curqx.mean(axis=1)

                self.m_par[:, i, self._frame_index //
                           self.resample_freq] += this_m_par
                self.mM_par[:, i, self._frame_index // self.resample_freq] += this_m_par * \
                    this_M_par[j]
                self.M_par[self._frame_index //
                           self.resample_freq] += this_M_par[j]
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
        dcov_perp = np.sqrt(
            (self.mM_perp.std(axis=2) / self._index * self.resample)**2 +
            (self.m_perp.std(axis=2) / self._index * self.resample *
             self.M_perp.sum() / self._index)**2 +
            (self.m_perp.sum(axis=2) / self._index * self.M_perp.std() /
             self._index * self.resample)**2) / np.sqrt(self.resample - 1)
        cov_perp_self = self.mm_perp / self._index - \
            (self.m_perp.sum(axis=2) / self._index * self.m_perp.sum(axis=2)
             / self._index * self.A * self.Lz / self.nbins / self._index)
        cov_perp_coll = self.cmM_perp / self._index - \
            self.m_perp.sum(axis=2) / self._index * self.cM_perp / self._index

        var_perp = self.M_perp_2.sum() / self._index - (
            self.M_perp.sum() / self._index)**2
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
        dcov_par = np.sqrt(
            (self.mM_par.std(axis=2) / self._index * self.resample)**2 +
            (self.m_par.std(axis=2) / self._index * self.resample *
             self.M_par.sum() / self._index)**2 +
            (self.m_par.sum(axis=2) / self._index * self.M_par.std() /
             self._index * self.resample)**2) / np.sqrt(self.resample - 1)

        beta = 1. / (scipy.constants.Boltzmann * self.temperature)

        self.results["eps_par"] = beta * eps0inv * pref / 2 * cov_par
        self.results["deps_par"] = beta * eps0inv * pref / 2 * dcov_par
        self.results["eps_par_self"] = beta * eps0inv * pref / 2 * cov_par_self
        self.results["eps_par_coll"] = beta * eps0inv * pref / 2 * cov_par_coll

        if (self.b2d):
            self.results["eps_perp"] = -beta * eps0inv * pref * cov_perp
            self.results[
                "eps_perp_self"] = -beta * eps0inv * pref * cov_perp_self
            self.results[
                "eps_perp_coll"] = -beta * eps0inv * pref * cov_perp_coll
            self.results["deps_perp"] = np.abs(
                -eps0inv * beta * pref) * dcov_perp
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
            self.results["z"] = np.linspace(self.zmin, self.Lz / self._index,
                                            len(self.results["eps_par"])) / 10
        else:
            self.results["z"] = np.linspace(self.zmin, self.zmax,
                                            len(self.results["eps_par"])) / 10.

    def _save_results(self):
        outdata_perp = np.hstack([
            self.results["z"][:, np.newaxis],
            self.results["eps_perp"].sum(axis=1)[:, np.newaxis],
            self.results["eps_perp"],
            np.linalg.norm(self.results["deps_perp"],
                           axis=1)[:, np.newaxis], self.results["deps_perp"],
            self.results["eps_perp_self"].sum(axis=1)[:, np.newaxis],
            self.results["eps_perp_coll"].sum(axis=1)[:, np.newaxis],
            self.results["eps_perp_self"], self.results["eps_perp_coll"]
        ])
        outdata_par = np.hstack([
            self.results["z"][:, np.newaxis],
            self.results["eps_par"].sum(axis=1)[:, np.newaxis],
            self.results["eps_par"],
            np.linalg.norm(self.results["deps_par"],
                           axis=1)[:, np.newaxis], self.results["deps_par"],
            self.results["eps_par_self"].sum(axis=1)[:, np.newaxis],
            self.results["eps_par_coll"].sum(axis=1)[:, np.newaxis],
            self.results["eps_par_self"], self.results["eps_par_coll"]
        ])

        if (self.bsym):
            for i in range(len(outdata_par) - 1):
                outdata_par[i + 1] = .5 * \
                    (outdata_par[i + 1] + outdata_par[i + 1][-1::-1])
            for i in range(len(outdata_perp) - 1):
                outdata_perp[i + 1] = .5 * \
                    (outdata_perp[i + 1] + outdata_perp[i + 1][-1::-1])

        header = "statistics over {:.1f} picoseconds".format(
            self._index * self._universe.trajectory.dt)
        savetxt(self.output + '_perp.dat', outdata_perp, header=header)
        savetxt(self.output + '_par.dat', outdata_par, header=header)


class dielectric_spectrum(SingleGroupAnalysisBase):
    """This module, given molecular dynamics trajectory data, produces a
        .txt file containing the complex dielectric function as a function of the (linear, not radial -
        i.e. nu or f, rather than omega) frequency, along with the associated standard deviations.
        The algorithm is based on the Fluctuation Dissipation Relation (FDR):\
        chi(f) = -1/(3 V k_B T epsilon_0) FT{theta(t) <P(0) dP(t)/dt>}.\
        By default, the polarization trajectory, time series array and the average system volume are
        saved in the working directory, and the data are reloaded from these files if they are present.
        Lin-log and log-log plots of the susceptibility are also produced by default."""

    # TODO set up script to calc spectrum at intervals while calculating polarization
    # for very big-data trajectories

    # TODO merge with molecular version?

    def __init__(self,
                 atomgroup,
                 temperature=300,
                 output="",
                 use=None,
                 segs=20,
                 df=None,
                 noplots=False,
                 plotformat="pdf",
                 ymin=None,
                 bins=200,
                 binafter=20,
                 nobin=False,
                 **kwargs):
        # Inherit all classes from AnalysisBase
        super(dielectric_spectrum, self).__init__(atomgroup, **kwargs)
        self.temperature = temperature
        self.output = output
        self.use = use
        self.segs = segs
        self.df = df
        self.noplots = noplots
        self.plotformat = plotformat
        self.ymin = ymin
        self.bins = bins
        self.binafter = binafter
        self.nobin = nobin

    def _configure_parser(self, parser):
        parser.description = self.__doc__
        parser.add_argument(
            "-recalc",
            action="store_true",
            help=
            "Forces to recalculate the polarization, regardless if it is already present."
        )
        parser.add_argument(
            '-temp',
            dest='temperature',
            type=float,
            default=300,
            help='Reference temperature.')
        parser.add_argument(
            "-o",
            dest="output",
            default="",
            help="Prefix for the output files.")
        parser.add_argument(
            "-segs",
            type=int,
            default=20,
            help="Sets the number of segments the trajectory is broken into.")
        parser.add_argument(
            "-df",
            type=float,
            help=
            "The desired frequency spacing in THz. This determines the minimum "
            + "frequency about which there is data. Overrides -segs option.")
        parser.add_argument(
            "-noplots",
            action="store_true",
            help="Prevents plots from being generated.")
        parser.add_argument(
            "-plotformat",
            default="pdf",
            choices=["png", "pdf", "ps", "eps", "svg"],
            help="Allows the user to choose the format of generated plots.")
        parser.add_argument(
            "-ymin",
            type=float,
            help="Manually sets the minimum lower bound for the log-log plot.")
        parser.add_argument(
            "-bins",
            type=int,
            default=200,
            help="Determines the number of bins used for data averaging;" +
            "(this parameter sets the upper limit)." +
            "The data are by default binned logarithmically. " +
            "This helps to reduce noise, particularly in" +
            "the high-frequency domain, and also prevents plot files from being too large."
        )
        parser.add_argument(
            "-binafter",
            type=int,
            default=20,
            help=
            "The number of low-frequency data points that are left unbinned.")
        parser.add_argument(
            "-nobin",
            action="store_true",
            help=
            "Prevents the data from being binned altogether. This can result in very large plot files and errors."
        )

    def _prepare(self):
        if len(self.output) > 0:
            self.output += "_"

        self.Nframes = (self.stopframe - self.startframe) // self.step
        self.dt = self._trajectory.dt * self.step
        self.V = 0
        self.P = np.zeros((self.Nframes, 3))

    def _single_frame(self):
        self.V += self._ts.volume
        repairMolecules(self.atomgroup)
        self.P[self._frame_index, :] = np.dot(self.atomgroup.charges,
                                              self.atomgroup.positions)

    def _calculate_results(self):

        self.results["t"] = self._trajectory.dt * np.arange(
            self.startframe, self.stopframe, self.step)

        self.results["V"] = self.V
        self.results["V"] *= 1e-3 / (self._frame_index + 1)

        self.results["P"] = self.P
        # MDAnalysis gives units of Å, we use nm
        self.results["P"] /= 10

        # Find a suitable number of segments if it's not specified:
        if self.df is not None:
            self.segs = np.max([int(self.Nframes * self.dt * self.df), 2])

        self.seglen = int(self.Nframes / self.segs)

        # Prefactor for susceptibility:
        pref = scipy.constants.e * scipy.constants.e * 1e9 / \
            (3 * self.results["V"] * scipy.constants.k
             * self.temperature * scipy.constants.epsilon_0)

        if self._verbose:  # Susceptibility and errors:
            print('Calculating susceptibilty and errors...')

        # if t too short to simply truncate
        if len(self.results["t"]) < 2 * self.seglen:
            self.results["t"] = np.append(
                self.results["t"],
                self.results["t"] + self.results["t"][-1] + self.dt)

        # truncate t array (it's automatically longer than 2 * seglen)
        self.results["t"] = self.results["t"][:2 * self.seglen]
        # get freqs
        self.results["nu"] = FT(
            self.results["t"],
            np.append(self.results["P"][:self.seglen, 0],
                      np.zeros(self.seglen)))[0]
        # susceptibility
        self.results["susc"] = np.zeros(self.seglen, dtype=complex)
        # std deviation of susceptibility
        self.results["dsusc"] = np.zeros(self.seglen, dtype=complex)
        # susceptibility for current seg
        ss = np.zeros((2 * self.seglen), dtype=complex)

        # loop over segs
        for s in range(0, self.segs):
            if self._verbose:
                print('\rSegment {0} of {1}'.format(s + 1, self.segs), end='')
            ss = 0 + 0j

            # loop over x, y, z
            for self._i in range(0, len(self.results["P"][0, :])):
                FP = FT(
                    self.results["t"],
                    np.append(
                        self.results["P"][s * self.seglen:(s + 1) *
                                          self.seglen, self._i],
                        np.zeros(self.seglen)), False)
                ss += FP.real * FP.real + FP.imag * FP.imag

            ss *= self.results["nu"] * 1j

            # Get the real part by Kramers Kronig
            ss.real = iFT(
                self.results["t"], 1j * np.sign(self.results["nu"]) * FT(
                    self.results["nu"], ss, False), False).imag

            if s == 0:
                self.results["susc"] += ss[self.seglen:]

            else:
                ds = ss[self.seglen:] - \
                    (self.results["susc"] / s)
                self.results["susc"] += ss[self.seglen:]
                dif = ss[self.seglen:] - \
                    (self.results["susc"] / (s + 1))
                ds.real *= dif.real
                ds.imag *= dif.imag
                # variance by Welford's Method
                self.results["dsusc"] += ds

        self.results["dsusc"].real = np.sqrt(self.results["dsusc"].real)
        self.results["dsusc"].imag = np.sqrt(self.results["dsusc"].imag)

        # 1/2 b/c it's the full FT, not only half-domain
        self.results["susc"] *= pref / (2 * self.seglen * self.segs * self.dt)
        self.results["dsusc"] *= pref / (2 * self.seglen * self.segs * self.dt)

        # Discard negative-frequency data; contains the same information as positive regime:
        # Now nu represents positive f instead of omega
        self.results["nu"] = self.results["nu"][self.seglen:] / (2 * np.pi)

        if self._verbose:
            print('Length of segments:    {0} frames, {1:.0f} ps'.format(
                self.seglen, self.seglen * self.dt))
            print('Frequency spacing:    ~ {0:.5f} THz'.format(
                self.segs / (self.Nframes * self.dt)))

        # Bin data if there are too many points:
        if not (self.nobin or self.seglen <= self.bins):
            bins = np.logspace(
                np.log(self.binafter) / np.log(10),
                np.log(len(self.results["susc"])) / np.log(10),
                self.bins - self.binafter + 1).astype(int)
            bins = np.unique(np.append(np.arange(self.binafter), bins))[:-1]

            self.results["nu_binned"] = Bin(self.results["nu"], bins)
            self.results["susc_binned"] = Bin(self.results["susc"], bins)
            self.results["dsusc_binned"] = Bin(self.results["dsusc"], bins)

            if self._verbose:
                print('Binning data above datapoint {0} in log-spaced bins'.
                      format(self.binafter))
                print('Binned data consists of {0} datapoints'.format(
                    len(self.results["susc"])))
        elif self._verbose:
            # data is binned
            print('Not binning data: there are {0} datapoints'.format(
                len(self.results["susc"])))

    def _save_results(self):
        np.save(self.output + 'tseries.npy', self.results["t"])

        with open(self.output + 'V.txt', "w") as Vfile:
            Vfile.write(str(self.results["V"]))

        np.save(self.output + 'P_tseries.npy', self.results["P"])

        suscfilename = self.output + 'susc.txt'
        savetxt(
            suscfilename,
            np.transpose([
                self.results["nu"], self.results["susc"].real,
                self.results["dsusc"].real, self.results["susc"].imag,
                self.results["dsusc"].imag
            ]),
            delimiter='\t',
            header='freq\tsusc\'\tstd_dev_susc\'\t-susc\'\'\tstd_dev_susc\'\'')

        if self._verbose:
            print('Susceptibility data saved as ' + suscfilename)

        if not (self.nobin or self.seglen <= self.bins):

            suscfilename = self.output + 'susc_binned.txt'
            savetxt(
                suscfilename,
                np.transpose([
                    self.results["nu_binned"], self.results["susc_binned"].real,
                    self.results["dsusc_binned"].real,
                    self.results["susc_binned"].imag,
                    self.results["dsusc_binned"].imag
                ]),
                delimiter='\t',
                header=
                'freq\tsusc\'\tstd_dev_susc\'\t-susc\'\'\tstd_dev_susc\'\'')

            if self._verbose:
                print('Binned susceptibility data saved as ' + suscfilename)

    def _conclude(self):
        if self.noplots and self._verbose:
            print('User specified not to generate plots -- finished :)')

        else:
            if self._verbose:
                print('Generating plots...')

            import matplotlib.pyplot as plt

            plt.rc('text', usetex=True)
            plt.rc('font', family='serif')

            # Colors/alpha values/labels/params for plotting
            col1 = 'royalblue'
            col2 = 'crimson'
            curve = 0.9
            shade = 0.15
            lw = 1.0
            nuBuf = 1.4  # buffer factor for extra room in the x direction
            cp = '$\chi^{{\prime}}$'
            cpp = '$\chi^{{\prime \prime}}$'
            width = 3.5  # inches

            def my_plot(binned=False):
                element = binned * "_binned"

                fig, ax = plt.subplots(1, figsize=[width, width / np.sqrt(2)])
                ax.set_ylabel('$\chi$')
                ax.set_xlabel('$\\nu$ [THz]')
                ax.set_xlim(self.results["nu"][1] / nuBuf,
                            self.results["nu"][-1] * nuBuf)
                ax.set_xscale('log')
                ax.set_yscale(yscale)

                ax.fill_between(
                    self.results["nu" + element][1:],
                    self.results["susc" + element].real[1:] -
                    self.results["dsusc" + element].real[1:],
                    self.results["susc" + element].real[1:] +
                    self.results["dsusc" + element].real[1:],
                    color=col2,
                    alpha=shade)
                ax.fill_between(
                    self.results["nu" + element][1:],
                    self.results["susc" + element].imag[1:] -
                    self.results["dsusc" + element].imag[1:],
                    self.results["susc" + element].imag[1:] +
                    self.results["dsusc" + element].imag[1:],
                    color=col1,
                    alpha=shade)

                ax.plot(
                    self.results["nu" + element][:2],
                    self.results["susc" + element].real[:2],
                    color=col2,
                    alpha=curve,
                    linestyle=':',
                    linewidth=lw)
                ax.plot(
                    self.results["nu" + element][:2],
                    self.results["susc" + element].imag[:2],
                    color=col1,
                    alpha=curve,
                    linestyle=':',
                    linewidth=lw)
                ax.plot(
                    self.results["nu" + element][1:],
                    self.results["susc" + element].real[1:],
                    color=col2,
                    alpha=curve,
                    label=cp,
                    linewidth=lw)
                ax.plot(
                    self.results["nu" + element][1:],
                    self.results["susc" + element].imag[1:],
                    color=col1,
                    alpha=curve,
                    label=cpp,
                    linewidth=lw)

                if self._i == 0 and (self.ymin is not None):
                    plt.set_ylim(ymin=self.ymin)
                ax.legend(loc='best', frameon=False)
                fig.tight_layout(pad=0.1)
                fig.savefig(plotname, format=self.plotformat)

            if self.nobin or self.seglen <= self.bins:
                binned = False
            else:
                binned = True

            yscale = 'log'
            plotname = self.output + 'susc_log.' + self.plotformat
            my_plot(binned)  # log-log

            yscale = 'linear'
            plotname = self.output + 'susc_linlog.' + self.plotformat
            my_plot(binned)  # lin-log

            plt.close('all')

            if self._verbose:
                print('Susceptibility plots generated -- finished :)')
