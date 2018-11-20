#!/usr/bin/env python
# coding: utf-8

from __future__ import absolute_import, division, print_function

import os
import subprocess
import sys
import tempfile

import numpy as np
import MDAnalysis as mda

from .base import AnalysisBase
from .. import sharePath

class debye(AnalysisBase):
    """Calculates scattering intensities using the debye equation.
       By using the -sel option atoms can be selected for which the
       profile is calculated. For group selections use strings in the MDAnalysis selection command style."""

    def __init__(self, atomgroup, sel="all", outfreq=100, output="sq",
                 startq=0, endq=6, dq=0.02, sinc=False, debyer="debyer", **kwargs):
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
        parser.add_argument('-sel', dest='sel', type=str, default='all',
                            help='Atoms for which to compute the profile', )
        parser.add_argument('-dout', dest='outfreq', type=float, default='100',
                            help='Number of frames after which the output is updated.')
        parser.add_argument('-sq', dest='output', type=str, default='sq',
                            help='Prefix/Path for output file')
        parser.add_argument('-startq', dest='startq', type=float, default=0,
                            help='Starting q (1/Å)')
        parser.add_argument('-endq', dest='endq', type=float, default=6,
                            help='Ending q (1/Å)')
        parser.add_argument('-dq', dest='dq', type=float, default=0.02,
                            help='binwidth (1/Å)')
        parser.add_argument('-sinc', dest='sinc', action='store_true',
                            help='apply sinc damping')
        parser.add_argument('-d', dest='debyer', type=str, default="debyer",
                            help='path to the debyer executable')

    def _prepare(self):

        type_dict = {}
        with open(os.path.join(sharePath, "atomtypes.dat")) as f:
            for line in f:
                if line[0] != '#':
                    elements = line.split()
                    type_dict[elements[0]] = elements[1]


        self.selection = self.atomgroup.select_atoms(self.sel + " and not name DUM and not name MW")

        if self._verbose:
            print("Selection '{}' contains {} atoms.\n".format(self.sel, self.selection.n_atoms))
        if self.selection.n_atoms == 0:
            raise RuntimeError("Selection does not contain any atoms.")

        # Create an extra list for the atom names.
        # This is necessary since it is not possible to efficently add axtra atoms to
        # a MDAnalysis universe, necessary for the hydrogens in united atom forcefields.

        self.atom_names = self.selection.n_atoms * ['']

        for i, atom_type in enumerate(self.selection.types.astype(str)):
            element = type_dict[atom_type]

            # add hydrogens in the case of united atom forcefields
            if element in ["CH1", "CH2", "CH3", "CH4", "NH", "NH2", "NH3"]:
                self.atom_names[i] = element[0]
                for h in range(int(element[-1])):
                    self.atom_names.append("H")
                    # add a extra atom to universe. It got the wrong type but we only
                    # need the position, since we maintain our own atom type list.
                    sel += sel.atoms[i]
            else:
                self.atom_names[i] = element

        # create tmp directory for saving datafiles
        self._tmp = tempfile.mkdtemp()

        self._OUT = open(os.devnull, 'w')

        if self._verbose:
            print("{} is the tempory directory for all files.\n".format(self._tmp))

    def _writeXYZ(self, filename):
        """Writes the positions of the current frame to the given xyz file"""
        write = mda.coordinates.XYZ.XYZWriter(filename,
                                              n_atoms=len(self.atom_names),
                                              atoms=self.atom_names)

        ts = self.selection.universe.trajectory.ts.copy_slice(self.selection.atoms.indices)
        write.write_next_timestep(ts)
        write.close()

    def _single_frame(self):

        # convert coordinates in a rectengular box
        box = np.diag(mda.lib.mdamath.triclinic_vectors(self._ts.dimensions))
        self.selection.atoms.positions = self.selection.atoms.positions \
            - box * np.round(self.selection.atoms.positions / box)  # minimum image

        self._writeXYZ("{}/{}.xyz".format(self._tmp, self._frame_index))

        ref_q = 4 * np.pi / np.min(box)
        if ref_q > self.startq:
            self.startq = ref_q

        command = "-x -f {0} -t {1} -s {2} -o {3}/{4}.dat -a {5} -b {6} -c {7} -r {8} {3}/{4}.xyz".format(
            round(self.startq, 3), self.endq, self.dq, self._tmp, self._frame_index,
            box[0], box[1], box[2], np.min(box) / 2.001)

        command += self.sinc * " --sinc"

        subprocess.call("{} {}".format(self.debyer, command),
                        stdout=self._OUT, stderr=self._OUT, shell=True)

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

        bins = ((s_tmp[:, 0] - self.startq) /
                ((self.endq - self.startq) / nbins)).astype(int)
        s_out = np.histogram(bins, bins=np.arange(
            nbins + 1), weights=s_tmp[:, 1])[0]

        nonzeros = np.where(s_out != 0)[0]

        self.results["q"] = q[nonzeros]
        self.results["I"] = s_out[nonzeros] / len(datfiles)

    def _conclude(self):
        for f in os.listdir(self._tmp):
            os.remove("{}/{}".format(self._tmp, f))

        os.rmdir(self._tmp)

    def _save_results(self):
        np.savetxt(self.output + '.dat',
                   np.vstack([self.results["q"], self.results["I"]]).T,
                   header="q (1/A)\tS(q)_tot (arb. units)", fmt='%.8e')
