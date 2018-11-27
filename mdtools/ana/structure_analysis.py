#!/usr/bin/env python
# coding: utf-8

from __future__ import absolute_import, division, print_function

import os
import subprocess
import sys
import tempfile
import math

import numba as nb
import numpy as np
import MDAnalysis as mda

from .base import AnalysisBase
from .. import tables


def compute_form_factor(q, atom_type):
    """Calculates the form factor for the given element for given q (1/nm).
       Handles united atom types like CH4 etc ..."""
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


@nb.jit(
    nb.types.UniTuple(nb.float32[:, :, :],
                      2)(nb.float32[:, :], nb.float32[:], nb.float32,
                         nb.float32, nb.float32, nb.float32),
    nopython=True,
    nogil=True,
    parallel=True)
def compute_structure_factor(positions, boxdimensions, start_q, end_q, mintheta,
                             maxtheta):
    """Calculates S(|q|) for all possible q values. Returns the q values as well as the scattering factor."""

    maxn = [0, 0, 0]
    q_factor = np.zeros(3, dtype=nb.float32)

    n_atoms = positions.shape[0]
    for i in range(3):
        q_factor[i] = 2 * np.pi / boxdimensions[i]
        maxn[i] = math.ceil(end_q / q_factor[i])

    S_array = np.zeros((maxn[0], maxn[1], maxn[2]), dtype=nb.float32)
    q_array = np.zeros((maxn[0], maxn[1], maxn[2]), dtype=nb.float32)

    for i in nb.prange(maxn[0]):
        qx = i * q_factor[0]

        for j in range(maxn[1]):
            qy = j * q_factor[1]

            for k in range(maxn[2]):
                if (i + j + k != 0):
                    qz = k * q_factor[2]
                    qrr = math.sqrt(qx * qx + qy * qy + qz * qz)
                    theta = math.acos(qz / qrr)

                    if (qrr >= start_q and qrr <= end_q and
                            theta >= mintheta and theta <= maxtheta):
                        q_array[i, j, k] = qrr

                        sin = 0
                        cos = 0
                        for l in range(n_atoms):
                            qdotr = positions[l, 0] * qx + \
                                positions[l, 1] * qy + positions[l, 2] * qz
                            sin += math.sin(qdotr)
                            cos += math.cos(qdotr)

                        S_array[i, j, k] += sin * sin + cos * cos

    return (q_array, S_array)


class saxs(AnalysisBase):
    """Computes SAXS scattering intensities S(q) for all atom types from the given trajectory.
    The q vectors are binned
    by their length using a binwidth given by -dq. Using the -nobin option
    the raw intensity for each q_{i,j,k} vector
    is saved using. Note that this only works reliable using constant box vectors!
    The possible scattering vectors q can be restricted by a miminal and maximal angle with the z-axis.
    For 0 and 180 all possible vectors are taken into account.
    For the scattering factor the structure fator is multiplied by a atom type specific form factor
    based on Cromer-Mann parameters. By using the -sel option atoms can be selected for which the
    profile is calculated. The selection uses the MDAnalysis selection commands."""

    def __init__(self, atomgroup, sel="all", outfreq=100, output="sq", nobin=False,
                startq=0, endq=60, dq=0.05, mintheta=0, maxtheta=180, **kwargs):
       # Inherit all classes from AnalysisBase
        super(saxs, self).__init__(atomgroup.universe.trajectory, **kwargs)

        self.atomgroup = atomgroup
        self.sel = sel
        self.outfreq = outfreq
        self.output = output
        self.nobindata = nobin
        self.startq = startq
        self.endq = endq
        self.dq = dq
        self.mintheta = mintheta
        self.maxtheta = maxtheta

    def _configure_parser(self, parser):
        parser.description = self.__doc__
        parser.add_argument(
            '-sel',
            dest='sel',
            type=str,
            default='all',
            help='Atoms for which to compute the profile',
        )
        parser.add_argument(
            '-dout',
            dest='outfreq',
            type=float,
            default=100,
            help='Number of frames after which the output is updated.')
        parser.add_argument(
            '-sq',
            dest='output',
            type=str,
            default='sq',
            help='Prefix/Path for output file')
        parser.add_argument(
            '-startq',
            dest='startq',
            type=float,
            default=0,
            help='Starting q (1/nm)')
        parser.add_argument(
            '-endq',
            dest='endq',
            type=float,
            default=60,
            help='Ending q (1/nm)')
        parser.add_argument(
            '-dq', dest='dq', type=float, default=0.05, help='binwidth (1/nm)')
        parser.add_argument(
            '-mintheta',
            dest='mintheta',
            type=float,
            default=0,
            help='Minimal angle (°) between the q vectors and the z-axis.')
        parser.add_argument(
            '-maxtheta',
            dest='maxtheta',
            type=float,
            default=180,
            help='Maximal angle (°) between the q vectors and the z-axis.')
    def _prepare(self):

        self.mintheta = min(self.mintheta, self.maxtheta)
        self.maxtheta = max(self.mintheta, self.maxtheta)

        if self.mintheta < 0 and self._verbose:
            print("mintheta = {}° < 0°: Set mininmal angle to 0°.".format(
                self.mintheta))
            self.mintheta = 0
        if self.maxtheta > 180 and self._verbose:
            print("maxtheta = {}° > 180°: Set maximal angle to 180°.".format(
                self.maxtheta))
            self.maxtheta = np.pi

        self.mintheta *= np.pi / 180
        self.maxtheta *= np.pi / 180

        self.selection = self.atomgroup.select_atoms(self.sel)

        if self.selection.n_atoms == 0:
            raise RuntimeError("Selection does not contain any atoms.")

        if self._verbose:
            print("\nSelection '{}' contains {} atoms.".format(
                self.sel, self.selection.n_atoms))

        self.groups = []
        self.atom_types = []
        if self._verbose:
            print("\nMap the following atomtypes:")
        for atom_type in np.unique(self.selection.atoms.types).astype(str):
            try:
                element = tables.atomtypes[atom_type]
            except KeyError:
                raise RuntimeError(
                    "No suitable element for '{0}' found. You can add '{0}' together with a suitable element to 'share/atomtypes.dat'."
                    .format(atom_type))
            if element == "DUM":
                continue
            self.groups.append(
                self.atomgroup.select_atoms("type {}*".format(atom_type)))
            self.atom_types.append(atom_type)

            if self._verbose:
                print("{:>14} --> {:>5}".format(atom_type, element))

        if self._verbose:
            print("")

        if self.nobindata:
            self.box = np.diag(mda.lib.mdamath.triclinic_vectors(self.selection.universe.dimensions)) / 10
            self.q_factor = 2 * np.pi / self.box
            self.maxn = np.ceil(self.endq / self.q_factor).astype(int)
            self.S_array = np.zeros(list(self.maxn) + [len(self.groups)])
        else:
            self.nbins = int(np.ceil((self.endq - self.startq) / self.dq))
            self.struct_factor = np.zeros([self.nbins, len(self.groups)])

    def _single_frame(self):
        for i, t in enumerate(self.groups):
            # convert everything to cartesian coordinates
            box = np.diag(
                mda.lib.mdamath.triclinic_vectors(self._ts.dimensions))
            positions = t.atoms.positions - box * \
                np.round(t.atoms.positions / box)  # minimum image

            q_ts, S_ts = compute_structure_factor(positions / 10, box / 10,
                                                  self.startq, self.endq,
                                                  self.mintheta, self.maxtheta)

            S_ts *= compute_form_factor(q_ts, self.atom_types[i])**2

            if self.nobindata:
                self.S_array[:, :, :, i] += S_ts
            else:
                q_ts = q_ts.flatten()
                S_ts = S_ts.flatten()
                nonzeros = np.where(S_ts != 0)[0]

                q_ts = q_ts[nonzeros]
                S_ts = S_ts[nonzeros]

                bins = ((q_ts - self.startq) /
                        ((self.endq - self.startq) / self.nbins)).astype(int)
                struct_ts = np.histogram(bins, bins=np.arange(self.nbins + 1),
                                         weights=S_ts)[0]
                with np.errstate(divide='ignore', invalid='ignore'):
                    struct_ts /= np.bincount(bins, minlength=self.nbins)
                self.struct_factor[:, i] += np.nan_to_num(struct_ts)

        if self._save and self._frame_index % self.outfreq == 0 and self._frame_index > 0:
            self._calculate_results()
            self._save_results()

    def _calculate_results(self):
        self._index = self._frame_index + 1
        if self.nobindata:
            self.results["scat_factor"] = self.S_array.sum(axis=3)
            self.results["q_indices"] = np.array(list(np.ndindex(tuple(self.maxn))))
            self.results["q"] = np.linalg.norm(self.results["q_indices"] * self.q_factor[np.newaxis,:],
                                               axis=1)
        else:
            q = np.arange(self.startq, self.endq, self.dq) + 0.5 * self.dq
            nonzeros = np.where(self.struct_factor[:, 0] != 0)[0]
            scat_factor = self.struct_factor[nonzeros]

            self.results["q"] = q[nonzeros]
            self.results["scat_factor"] = scat_factor.sum(axis=1)

        self.results["scat_factor"] /= (self._index * self.selection.n_atoms)

    def _save_results(self):
        """Saves the current profiles to a file."""

        if self.nobindata:
            out = np.hstack([self.results["q"][:,np.newaxis],
                             self.results["q_indices"],
                             self.results["scat_factor"].flatten()[:,np.newaxis]])
            nonzeros = np.where(out[:, 4] != 0)[0]
            out = out[nonzeros]
            argsort = np.argsort(out[:, 0])
            out = out[argsort]

            boxinfo = "box_x = {0:.3f} nm, box_y = {1:.3f} nm, box_z = {2:.3f} nm\n".format(*self.box)
            np.savetxt(self.output + '.dat', out,
                        header=boxinfo + "q (1/nm)\tq_i\t q_j \t q_k \tS(q) (arb. units)",
                        fmt='%.4e')
        else:
            np.savetxt(self.output + '.dat',
                       np.vstack([self.results["q"], self.results["scat_factor"]]).T,
                       header="q (1/nm)\tS(q) (arb. units)",
                       fmt='%.4e')


class debye(AnalysisBase):
    """Calculates scattering intensities using the debye equation.
       By using the -sel option atoms can be selected for which the
       profile is calculated. For group selections use strings in the MDAnalysis selection command style."""

    def __init__(self,
                 atomgroup,
                 sel="all",
                 outfreq=100,
                 output="sq",
                 startq=0,
                 endq=60,
                 dq=0.05,
                 sinc=False,
                 debyer="debyer",
                 **kwargs):
        # Inherit all classes from AnalysisBase
        super(debye, self).__init__(atomgroup.universe.trajectory, **kwargs)

        self.atomgroup = atomgroup
        self.sel = sel
        self.outfreq = outfreq
        self.output = output
        self.startq = startq
        self.endq = endq
        self.dq = dq
        self.sinc = sinc
        self.debyer = debyer

    def _configure_parser(self, parser):
        parser.description = self.__doc__
        parser.add_argument(
            '-sel',
            dest='sel',
            type=str,
            default='all',
            help='Atoms for which to compute the profile',
        )
        parser.add_argument(
            '-dout',
            dest='outfreq',
            type=float,
            default=100,
            help='Number of frames after which the output is updated.')
        parser.add_argument(
            '-sq',
            dest='output',
            type=str,
            default='sq',
            help='Prefix/Path for output file')
        parser.add_argument(
            '-startq',
            dest='startq',
            type=float,
            default=0,
            help='Starting q (1/nm)')
        parser.add_argument(
            '-endq',
            dest='endq',
            type=float,
            default=60,
            help='Ending q (1/nm)')
        parser.add_argument(
            '-dq', dest='dq', type=float, default=0.05, help='binwidth (1/nm)')
        parser.add_argument(
            '-sinc',
            dest='sinc',
            action='store_true',
            help='apply sinc damping')
        parser.add_argument(
            '-d',
            dest='debyer',
            type=str,
            default="debyer",
            help='path to the debyer executable')

    def _prepare(self):

        # Convert 1/nm to 1/Å
        self.startq /= 10
        self.endq /= 10
        self.dq /= 10

        self.selection = self.atomgroup.select_atoms(
            self.sel + " and not name DUM and not name MW")

        if self._verbose:
            print("Selection '{}' contains {} atoms.\n".format(
                self.sel, self.selection.n_atoms))
        if self.selection.n_atoms == 0:
            raise RuntimeError("Selection does not contain any atoms.")

        # Create an extra list for the atom names.
        # This is necessary since it is not possible to efficently add axtra atoms to
        # a MDAnalysis universe, necessary for the hydrogens in united atom forcefields.

        self.atom_names = self.selection.n_atoms * ['']

        for i, atom_type in enumerate(self.selection.types.astype(str)):
            element = tables.atomtypes[atom_type]

            # add hydrogens in the case of united atom forcefields
            if element in ["CH1", "CH2", "CH3", "CH4", "NH", "NH2", "NH3"]:
                self.atom_names[i] = element[0]
                for h in range(int(element[-1])):
                    self.atom_names.append("H")
                    # add a extra atom to universe. It got the wrong type but we only
                    # need the position, since we maintain our own atom type list.
                    self.selection += self.selection.atoms[i]
            else:
                self.atom_names[i] = element

        # create tmp directory for saving datafiles
        self._tmp = tempfile.mkdtemp()

        self._OUT = open(os.devnull, 'w')

        try:
            subprocess.call(self.debyer, stdout=self._OUT, stderr=self._OUT)
        except FileNotFoundError:
            raise RuntimeError("{}: command not found".format(self.debyer))

        if self._verbose:
            print("{} is the tempory directory for all files.\n".format(
                self._tmp))

    def _writeXYZ(self, filename):
        """Writes the positions of the current frame to the given xyz file"""
        write = mda.coordinates.XYZ.XYZWriter(
            filename, n_atoms=len(self.atom_names), atoms=self.atom_names)

        ts = self.selection.universe.trajectory.ts.copy_slice(
            self.selection.atoms.indices)
        write.write_next_timestep(ts)
        write.close()

    def _single_frame(self):

        # convert coordinates in a rectengular box
        box = np.diag(mda.lib.mdamath.triclinic_vectors(self._ts.dimensions))
        self.selection.atoms.positions = self.selection.atoms.positions \
            - box * np.round(self.selection.atoms.positions /
                             box)  # minimum image

        self._writeXYZ("{}/{}.xyz".format(self._tmp, self._frame_index))

        ref_q = 4 * np.pi / np.min(box)
        if ref_q > self.startq:
            self.startq = ref_q

        command = "-x -f {0} -t {1} -s {2} -o {3}/{4}.dat -a {5} -b {6} -c {7} -r {8} {3}/{4}.xyz".format(
            round(self.startq, 3), self.endq, self.dq, self._tmp,
            self._frame_index, box[0], box[1], box[2],
            np.min(box) / 2.001)

        command += self.sinc * " --sinc"

        subprocess.call(
            "{} {}".format(self.debyer, command),
            stdout=self._OUT,
            stderr=self._OUT,
            shell=True)

        if self._save and self._frame_index % self.outfreq == 0 and self._frame_index > 0:
            self._calculate_results()
            self._save_results()

    def _calculate_results(self):
        datfiles = [f for f in os.listdir(self._tmp) if f.endswith(".dat")]

        s_tmp = np.loadtxt("{}/{}".format(self._tmp, datfiles[0]))
        for f in datfiles[1:]:
            s_tmp = np.vstack([s_tmp, np.loadtxt("{}/{}".format(self._tmp, f))])

        nbins = int(np.ceil((self.endq - self.startq) / self.dq))
        q = np.arange(self.startq, self.endq, self.dq) + 0.5 * self.dq

        bins = ((s_tmp[:, 0] - self.startq) / (
            (self.endq - self.startq) / nbins)).astype(int)
        s_out = np.histogram(
            bins, bins=np.arange(nbins + 1), weights=s_tmp[:, 1])[0]

        nonzeros = np.where(s_out != 0)[0]

        self.results["q"] = 10 * q[nonzeros]
        self.results["scat_factor"] = s_out[nonzeros] / len(datfiles)

    def _conclude(self):
        for f in os.listdir(self._tmp):
            os.remove("{}/{}".format(self._tmp, f))

        os.rmdir(self._tmp)

    def _save_results(self):
        np.savetxt(
            self.output + '.dat',
            np.vstack([self.results["q"], self.results["scat_factor"]]).T,
            header="q (1/A)\tS(q)_tot (arb. units)",
            fmt='%.8e')
