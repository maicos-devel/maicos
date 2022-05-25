#!/usr/bin/env python3
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
#
# Copyright (c) 2022 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the GNU Public Licence, v3 or any higher version
# SPDX-License-Identifier: GPL-3.0-or-later
r"""Tools for computing structure properties.

The structure modules of MAICoS allow for calculating Small-Angle X-ray
Scattering (SAXS) scattering intensities, scattering intensities using the
Debye equation, and dipolar order parameters from molecular simulation
trajectory files.
"""

import logging
import os
import subprocess
import tempfile

import MDAnalysis as mda
import numpy as np

from .. import tables
from ..decorators import make_whole, set_planar_class_doc, set_verbose_doc
from ..lib import sfactor
from ..utils import check_compound, savetxt
from .base import AnalysisBase, PlanarBase


logger = logging.getLogger(__name__)


def compute_form_factor(q, atom_type):
    """Calculate the form factor for the given element for given q (1/nm).

    Handles united atom types like CH4 etc ...
    """
    element = tables.atomtypes[atom_type]

    if element == "CH1":
        form_factor = compute_form_factor(q, "C") + compute_form_factor(q, "H")
    elif element == "CH2":
        form_factor = compute_form_factor(
            q, "C") + 2 * compute_form_factor(q, "H")
    elif element == "CH3":
        form_factor = compute_form_factor(
            q, "C") + 3 * compute_form_factor(q, "H")
    elif element == "CH4":
        form_factor = compute_form_factor(
            q, "C") + 4 * compute_form_factor(q, "H")
    elif element == "NH1":
        form_factor = compute_form_factor(q, "N") + compute_form_factor(q, "H")
    elif element == "NH2":
        form_factor = compute_form_factor(
            q, "N") + 2 * compute_form_factor(q, "H")
    elif element == "NH3":
        form_factor = compute_form_factor(
            q, "N") + 3 * compute_form_factor(q, "H")
    else:
        form_factor = tables.CM_parameters[element].c
        # factor of 10 to convert from 1/nm to 1/Angstroms
        q2 = (q / (4 * np.pi * 10))**2
        for i in range(4):
            form_factor += tables.CM_parameters[element].a[i] * \
                np.exp(-tables.CM_parameters[element].b[i] * q2)

    return form_factor


@set_verbose_doc
class Saxs(AnalysisBase):
    """Compute SAXS scattering intensities.

    The q vectors are binned by their length using a binwidth given by -dq.
    Using the -nobin option the raw intensity for each q_{i,j,k} vector
    is saved using. Note that this only works reliable using constant
    box vectors! The possible scattering vectors q can be restricted by a
    miminal and maximal angle with the z-axis. For 0 and 180 all possible
    vectors are taken into account. For the scattering factor the structure
    factor is multiplied by a atom type specific form factor based on
    Cromer-Mann parameters. By using the -sel option atoms can be selected
    for which the profile is calculated. The selection uses the
    MDAnalysis selection commands.

    Parameters
    ----------
    atomgroup : AtomGroup
       Atomgroup on which the analysis is executed
    noboindata : bool
        Do not bin the data. Only works reliable for NVT!
    startq : float
        Starting q (1/nm)
    endq : float
        Ending q (1/nm)
    dq : float
        binwidth (1/nm)
    mintheta float
        Minimal angle (°) between the q vectors and the z-axis.
    maxtheta : float
        Maximal angle (°) between the q vectors and the z-axis.
    output : str
        Output filename
    concfreq : int
        Default number of frames after which results are calculated
        and files refreshed. If `0` results are only calculated at
        the end of the analysis and not saved by default.
    ${VERBOSE_PARAMETER}

    Attributes
    ----------
    results.q : np.ndarray
        length of binned q-vectors
    results.q_indices : np.ndarray
        Miller indices of q-vector (only if noboindata==True)
    results.scat_factor : np.ndarray
        Scattering intensities
    """

    def __init__(self,
                 atomgroup,
                 nobin=False,
                 startq=0,
                 endq=60,
                 dq=0.05,
                 mintheta=0,
                 maxtheta=180,
                 output="sq.dat",
                 concfreq=0,
                 **kwargs):
        super(Saxs, self).__init__(atomgroup, **kwargs)
        self.nobindata = nobin
        self.startq = startq
        self.endq = endq
        self.dq = dq
        self.mintheta = mintheta
        self.maxtheta = maxtheta
        self.output = output
        self.concfreq = concfreq

    def _prepare(self):

        self.mintheta = min(self.mintheta, self.maxtheta)
        self.maxtheta = max(self.mintheta, self.maxtheta)

        if self.mintheta < 0:
            logger.info(f"mintheta = {self.mintheta}° < 0°: "
                        "Set mininmal angle to 0°.")
            self.mintheta = 0
        if self.maxtheta > 180:
            logger.info(f"maxtheta = {self.maxtheta}° > 180°: "
                        "Set maximal angle to 180°.")
            self.maxtheta = np.pi

        self.mintheta *= np.pi / 180
        self.maxtheta *= np.pi / 180

        self.groups = []
        self.atom_types = []
        logger.info("\nMap the following atomtypes:")
        for atom_type in np.unique(self.atomgroup.types).astype(str):
            try:
                element = tables.atomtypes[atom_type]
            except KeyError:
                raise RuntimeError(f"No suitable element for '{atom_type}' "
                                   f"found. You can add '{atom_type}' "
                                   "together with a suitable element "
                                   "to 'share/atomtypes.dat'.")
            if element == "DUM":
                continue
            self.groups.append(
                self.atomgroup.select_atoms("type {}*".format(atom_type)))
            self.atom_types.append(atom_type)

            logger.info("{:>14} --> {:>5}".format(atom_type, element))

        if self.nobindata:
            self.box = np.diag(
                mda.lib.mdamath.triclinic_vectors(
                    self._universe.dimensions)) / 10
            self.q_factor = 2 * np.pi / self.box
            self.maxn = np.ceil(self.endq / self.q_factor).astype(int)
            self.S_array = np.zeros(list(self.maxn) + [len(self.groups)])
        else:
            self.n_bins = int(np.ceil((self.endq - self.startq) / self.dq))
            self.struct_factor = np.zeros([self.n_bins, len(self.groups)])

    def _single_frame(self):
        # Convert everything to cartesian coordinates.
        box = np.diag(mda.lib.mdamath.triclinic_vectors(self._ts.dimensions))
        for i, t in enumerate(self.groups):
            positions = t.atoms.positions - box * \
                np.round(t.atoms.positions / box)  # minimum image

            q_ts, S_ts = sfactor.compute_structure_factor(
                np.double(positions / 10), np.double(box / 10), self.startq,
                self.endq, self.mintheta, self.maxtheta)

            S_ts *= compute_form_factor(q_ts, self.atom_types[i])**2

            if self.nobindata:
                self.S_array[:, :, :, i] += S_ts
            else:
                q_ts = q_ts.flatten()
                S_ts = S_ts.flatten()
                nonzeros = np.where(S_ts != 0)[0]

                q_ts = q_ts[nonzeros]
                S_ts = S_ts[nonzeros]

                struct_ts = np.histogram(q_ts,
                                         bins=self.n_bins,
                                         range=(self.startq, self.endq),
                                         weights=S_ts)[0]
                with np.errstate(divide='ignore', invalid='ignore'):
                    struct_ts /= np.histogram(q_ts,
                                              bins=self.n_bins,
                                              range=(self.startq, self.endq))[0]
                self.struct_factor[:, i] += np.nan_to_num(struct_ts)

        if self.concfreq and self._frame_index % self.concfreq == 0 \
                and self._frame_index > 0:
            self._conclude()
            self.save()

    def _conclude(self):
        self._index = self._frame_index + 1
        if self.nobindata:
            self.results.scat_factor = self.S_array.sum(axis=3)
            self.results.q_indices = np.array(
                list(np.ndindex(tuple(self.maxn))))
            self.results.q = np.linalg.norm(self.results.q_indices
                                            * self.q_factor[np.newaxis, :],
                                            axis=1)
        else:
            q = np.arange(self.startq, self.endq, self.dq) + 0.5 * self.dq
            nonzeros = np.where(self.struct_factor[:, 0] != 0)[0]
            scat_factor = self.struct_factor[nonzeros]

            self.results.q = q[nonzeros]
            self.results.scat_factor = scat_factor.sum(axis=1)

        self.results.scat_factor /= (self._index * self.atomgroup.n_atoms)

    def save(self):
        """Save the current profiles to a file."""
        if self.nobindata:
            out = np.hstack([
                self.results.q[:, np.newaxis], self.results.q_indices,
                self.results.scat_factor.flatten()[:, np.newaxis]
                ])
            nonzeros = np.where(out[:, 4] != 0)[0]
            out = out[nonzeros]
            argsort = np.argsort(out[:, 0])
            out = out[argsort]

            boxinfo = "box_x = {0:.3f} nm, box_y = {1:.3f} nm, " \
                      "box_z = {2:.3f} nm\n".format(*self.box)
            savetxt(self.output, out,
                    header=boxinfo + "q (1/nm)\tq_i\t q_j \t "
                                     "q_k \tS(q) (arb. units)", fmt='%.4e')
        else:
            savetxt(self.output,
                    np.vstack([self.results.q,
                               self.results.scat_factor]).T,
                    header="q (1/nm)\tS(q) (arb. units)",
                    fmt='%.4e')


@set_verbose_doc
class Debye(AnalysisBase):
    """Calculate scattering intensities using the Debye equation.

    Parameters
    ----------
    atomgroup : AtomGroup
       Atomgroup on which the analysis is executed
    outfreq float :
        Number of frames after which the output is updated.
    output : str
        Output filename
    startq : float
        Starting q (1/nm)
    endq : float
        Ending q (1/nm)
    dq : float
        binwidth (1/nm)
    sinc : bool
        Apply sinc damping
    debyer : str
        Path to the debyer executable
    ${VERBOSE_PARAMETER}

    Attributes
    ----------
    results.q : np.ndarray
        length of binned q-vectors
    results.scat_factor : np.ndarray
        Scattering intensities
    """

    def __init__(self,
                 atomgroup,
                 startq=0,
                 endq=60,
                 dq=0.05,
                 sinc=False,
                 debyer="debyer",
                 output="sq.dat",
                 concfreq=0,
                 **kwargs):
        super(Debye, self).__init__(atomgroup, **kwargs)
        self.startq = startq
        self.endq = endq
        self.dq = dq
        self.sinc = sinc
        self.debyer = debyer
        self.output = output
        self.concfreq = concfreq

    def _configure_parser(self, parser):
        parser.add_argument('-dout', dest='outfreq')
        parser.add_argument('-sq', dest='output')
        parser.add_argument('-startq', dest='startq')
        parser.add_argument('-endq', dest='endq')
        parser.add_argument(
            '-dq',
            dest='dq',
            )
        parser.add_argument('-sinc', dest='sinc')
        parser.add_argument('-d', dest='debyer')

    def _prepare(self):

        # Convert 1/nm to 1/Å
        self.startq /= 10
        self.endq /= 10
        self.dq /= 10

        self.atomgroup = self.atomgroup.select_atoms(
            "not name DUM and not name MW")

        # Create an extra list for the atom names. This is necessary since it
        # is not possible to efficently add axtra atoms to a MDAnalysis
        # universe, necessary for the hydrogens in united atom forcefields.

        self.atom_names = self.atomgroup.n_atoms * ['']

        for i, atom_type in enumerate(self.atomgroup.types.astype(str)):
            element = tables.atomtypes[atom_type]

            # add hydrogens in the case of united atom forcefields
            if element in ["CH1", "CH2", "CH3", "CH4", "NH", "NH2", "NH3"]:
                self.atom_names[i] = element[0]
                for _h in range(int(element[-1])):
                    self.atom_names.append("H")
                    # add an extra atom to universe. It got the wrong
                    # type but we only need the position, since we maintain
                    # our own atom type list.
                    self.atomgroup += self.atomgroup.atoms[i]
            else:
                self.atom_names[i] = element

        # create tmp directory for saving datafiles
        self._tmp = tempfile.mkdtemp()

        self._OUT = open(os.devnull, 'w')

        try:
            subprocess.call(self.debyer, stdout=self._OUT, stderr=self._OUT)
        except FileNotFoundError:
            raise RuntimeError("{}: command not found".format(self.debyer))

        logger.info(f"{self._tmp} is the tempory directory for all files.")

    def _writeXYZ(self, filename):
        """Write the positions of the current frame to the given xyz file."""
        write = mda.coordinates.XYZ.XYZWriter(filename,
                                              n_atoms=len(self.atom_names),
                                              atoms=self.atom_names)

        ts = self._trajectory.ts.copy_slice(self.atomgroup.indices)
        write.write_next_timestep(ts)
        write.close()

    def _single_frame(self):

        # convert coordinates in a rectengular box
        box = np.diag(mda.lib.mdamath.triclinic_vectors(self._ts.dimensions))
        self.atomgroup.positions \
            = self.atomgroup.positions - box \
            * np.round(self.atomgroup.positions / box)  # minimum image

        self._writeXYZ("{}/{}.xyz".format(self._tmp, self._frame_index))

        ref_q = 4 * np.pi / np.min(box)
        if ref_q > self.startq:
            self.startq = ref_q

        command = "-x -f {0} -t {1} -s {2} -o {3}/{4}.dat " \
                  "-a {5} -b {6} -c {7} -r {8} " \
                  "{3}/{4}.xyz".format(round(self.startq, 3),
                                       self.endq, self.dq, self._tmp,
                                       self._frame_index, box[0],
                                       box[1], box[2],
                                       np.min(box) / 2.001)

        command += self.sinc * " --sinc"

        subprocess.call("{} {}".format(self.debyer, command),
                        stdout=self._OUT,
                        stderr=self._OUT,
                        shell=True)

        if self.concfreq and self._frame_index % self.concfreq == 0 \
                and self._frame_index > 0:
            self._conclude()
            self.save()

    def _conclude(self):
        datfiles = [f for f in os.listdir(self._tmp) if f.endswith(".dat")]

        s_tmp = np.loadtxt("{}/{}".format(self._tmp, datfiles[0]))
        for f in datfiles[1:]:
            s_tmp = np.vstack(
                [s_tmp, np.loadtxt("{}/{}".format(self._tmp, f))])

        n_bins = int(np.ceil((self.endq - self.startq) / self.dq))
        q = np.arange(self.startq, self.endq, self.dq) + 0.5 * self.dq

        s_out = np.histogram(s_tmp[:, 0],
                             bins=n_bins,
                             range=(self.startq, self.endq),
                             weights=s_tmp[:, 1])[0]

        nonzeros = np.where(s_out != 0)[0]

        self.results.q = 10 * q[nonzeros]
        self.results.scat_factor = s_out[nonzeros] / len(datfiles)

        # Remove temp files at the end of the analysis
        if self.n_frames == self._frame_index + 1:
            for f in os.listdir(self._tmp):
                os.remove("{}/{}".format(self._tmp, f))

            os.rmdir(self._tmp)

    def save(self):
        """Save result."""
        savetxt(self.output,
                np.vstack([self.results.q, self.results.scat_factor]).T,
                header="q (1/A)\tS(q)_tot (arb. units)",
                fmt='%.8e')


@set_verbose_doc
@set_planar_class_doc
@make_whole()
class Diporder(PlanarBase):
    """Calculate dipolar order parameters.

    Calculations include the projected dipole density
    P_0⋅ρ(z)⋅cos(θ[z]), the dipole orientation cos(θ[z]), the squared dipole
    orientation cos²(Θ[z]) and the number density ρ(z).

    Parameters
    ----------
    atomgroup : AtomGroup
       Atomgroup on which the analysis is executed
    ${PLANAR_CLASS_PARAMETERS}
    sym : bool
        symmetrize the profiles
    binmethod : str
        binning method: center of mass (COM) or center of charge (COC)
    ${MAKE_WHOLE_PARAMETER}
    output : str
        Output filename
    concfreq : int
        Default number of frames after which results are calculated
        and files refreshed. If `0` results are only calculated at the
        end of the analysis and not saved by default.
    ${VERBOSE_PARAMETER}

    Attributes
    ----------
    ${PLANAR_CLASS_ATTRIBUTES}
    results.P0 : np.ndarray
        P_0⋅ρ(z)⋅cos(θ[z]) [e/nm²]
    results.cos_theta : np.ndarray
        cos(θ[z])
    results.cos_2_theta : np.ndarray
        cos²(Θ[z])
    results.rho : np.ndarray
        ρ(z) [1/nm³]
    """

    def __init__(self,
                 atomgroup,
                 dim=2,
                 zmin=0,
                 zmax=None,
                 binwidth=0.1,
                 center=False,
                 comgroup=None,
                 sym=False,
                 binmethod='COM',
                 make_whole=True,
                 output="diporder.dat",
                 concfreq=0,
                 **kwargs):
        super(Diporder, self).__init__(atomgroups=atomgroup,
                                       dim=dim,
                                       zmin=zmin,
                                       zmax=zmax,
                                       binwidth=binwidth,
                                       center=center,
                                       comgroup=comgroup,
                                       **kwargs)
        self.sym = sym
        self.binmethod = binmethod
        self.make_whole = make_whole
        self.concfreq = concfreq
        self.output = output

    def _prepare(self):
        """Set things up before the analysis loop begins."""
        super(Diporder, self)._prepare()
        self.binmethod = self.binmethod.upper()
        if self.binmethod not in ["COM", "COC"]:
            raise ValueError('Unknown binning method: {}'.format(
                self.binmethod))

        # Check if all residues are identical.
        # Choose first residue as reference.
        residue_names = [ag.names for ag in self.atomgroup.split("residue")]
        for names in residue_names[1:]:
            if len(residue_names[0]) != len(names) and np.all(
                    residue_names[0] != names):
                raise ValueError("Not all residues are identical. Please adjust"
                                 "selection.")

        # Assume a threedimensional universe...
        self.xydims = np.roll(np.arange(3), -self.dim)[1:]
        self.P0 = np.zeros(self.n_bins)
        self.cos_theta = np.zeros(self.n_bins)
        self.cos_2_theta = np.zeros(self.n_bins)
        self.rho = np.zeros(self.n_bins)
        self.bin_count = np.zeros(self.n_bins)

        # unit normal vector
        self.unit = np.zeros(3)
        self.unit[self.dim] += 1

    def _single_frame(self):
        super(Diporder, self)._single_frame()
        dz_frame = self._ts.dimensions[self.dim] / self.n_bins

        chargepos = self.atomgroup.positions \
            * self.atomgroup.charges[:, np.newaxis]
        dipoles = self.atomgroup.accumulate(
            chargepos, compound=check_compound(self.atomgroup))
        dipoles /= 10  # convert to e nm

        if self.binmethod == 'COM':
            # Calculate the centers of the objects (i.e. Molecules)
            bin_positions = self.atomgroup.center_of_mass(
                compound=check_compound(self.atomgroup))
        elif self.binmethod == 'COC':
            bin_positions = self.atomgroup.center(
                weights=np.abs(self.atomgroup.charges),
                compound=check_compound(self.atomgroup))

        bincount = np.histogram(bin_positions[:,self.dim],
                                bins=self.n_bins,
                                range=(self.zmin, self.zmax))[0]
        self.bin_count += bincount
        A = np.prod(self._ts.dimensions[self.xydims])

        self.P0 += np.histogram(
            bin_positions[:, self.dim],
            bins=self.n_bins,
            range=(self.zmin, self.zmax),
            weights=np.dot(dipoles, self.unit))[0] / (A * dz_frame / 1e3)
        self.cos_theta += np.histogram(
            bin_positions[:, self.dim],
            bins=self.n_bins,
            range=(self.zmin, self.zmax),
            weights=np.dot(
                dipoles / np.linalg.norm(dipoles, axis=1)[:, np.newaxis],
                self.unit))[0]
        self.cos_2_theta += np.histogram(
            bin_positions[:, self.dim],
            bins=self.n_bins,
            range=(self.zmin, self.zmax),
            weights=np.dot(
                dipoles / np.linalg.norm(dipoles, axis=1)[:, np.newaxis],
                self.unit)**2)[0]
        self.rho += bincount / (A * dz_frame / 1e3)  # convert to 1/nm^3

        if self.concfreq and self._frame_index % self.concfreq == 0 \
                and self._frame_index > 0:
            self._conclude()
            self.save()

    def _conclude(self):
        """Calculate the results.

        Called at the end of the run() method to before the _conclude function.
        Can also be called during a run to update the results during
        processing.
        """
        super(Diporder, self)._conclude()
        self._index = self._frame_index + 1
        self.results.P0 = self.P0 / self._index
        self.results.rho = self.rho / self._index

        with np.errstate(divide='ignore', invalid='ignore'):
            self.results.cos_theta = np.nan_to_num(
                self.cos_theta / self.bin_count)
            self.results.cos_2_theta = np.nan_to_num(
                self.cos_2_theta / self.bin_count)

        if self.sym:
            for i in range(len(self.results.z) - 1):
                self.results.z[i + 1] \
                    = .5 * (self.results.z[i + 1]
                            + self.results.z[i + 1][-1::-1])
                self.results.diporder[i + 1] \
                    = .5 * (self.results.diporder[i + 1]
                            + self.results.diporder[i + 1][-1::-1])

    def save(self):
        """Save results to a file."""
        header = "z [nm]\t"
        header += "P_0*rho(z)*cos(Theta[z]) [e/nm^2]\t"
        header += "cos(theta(z))\t"
        header += "cos^2(theta(z))\t"
        header += "rho(z) [1/nm^3]"

        savetxt(self.output,
                np.vstack([
                    self.results.z, self.results.P0,
                    self.results.cos_theta, self.results.cos_2_theta,
                    self.results.rho
                    ]).T,
                header=header)
