#!/usr/bin/env python
# coding: utf-8

from __future__ import division, print_function

import argparse
import os
import subprocess
import sys
import tempfile

import MDAnalysis as mda
import numpy as np
from scipy.stats import binned_statistic

from . import initilize_universe, print_frameinfo
from .. import sharePath, initilize_parser

#========== PARSER ===========
#=============================
parser = initilize_parser(add_traj_arguments=True)
parser.description = """
    A python interface to the Deyer library. By using the -sel option atoms can be selected for which the
    profile is calculated. The selection uses the MDAnalysis selection commands found here:
    http://www.mdanalysis.org/docs/documentation_pages/selections.html
    The system can be replicated with the -nbox option. The system is than stacked multiplie times on itself. No
    replication is done be default."""
parser.add_argument('-sel',   dest='sel',         type=str,     default='all',
                    help='Atoms for which to compute the profile', )
parser.add_argument('-dout',  dest='outfreq',     type=float,   default='100',
                    help='Number of frames after which the output is updated.')
parser.add_argument('-sq',    dest='output',      type=str,
                    default='./sq',                 help='Prefix/Path for output file')
parser.add_argument('-startq', dest='startq',      type=float,
                    default=0,                      help='Starting q (1/Å)')
parser.add_argument('-endq',  dest='endq',        type=float,
                    default=6,                     help='Ending q (1/Å)')
parser.add_argument('-dq',    dest='dq',          type=float,
                    default=0.02,                   help='binwidth (1/Å)')
parser.add_argument('-d',     dest='debyer',      type=str,
                    default="~/repos/debyer/debyer/debyer", help='path to the debyer executable')
parser.add_argument('-v',     dest='verbose',     action='store_true',
                    help='Be loud and noisy.')


def output():
    """averages over all dat file and removes them"""
    datfiles = [f for f in os.listdir(args.tmp) if f.endswith(".dat")]

    s_tmp = np.loadtxt("{}/{}".format(args.tmp, datfiles[0]))
    for f in datfiles[1:]:
        s_tmp = np.vstack([s_tmp, np.loadtxt("{}/{}".format(args.tmp, f))])

    nbins = int(np.ceil((args.endq - args.startq) / args.dq))
    q = np.arange(args.startq, args.endq, args.dq) + 0.5 * args.dq

    bins = ((s_tmp[:, 0] - args.startq) /
            ((args.endq - args.startq) / nbins)).astype(int)
    s_out = np.histogram(bins, bins=np.arange(
        nbins + 1), weights=s_tmp[:, 1])[0]

    nonzeros = np.where(s_out != 0)[0]

    np.savetxt(args.output + '.dat',
               np.vstack([q[nonzeros], s_out[nonzeros] / len(datfiles)]).T,
               header="q (1/A)\tS(q)_tot (arb. units)", fmt='%.8e')


def cleanup():
    """Cleans up temporal file directory."""

    for f in os.listdir(args.tmp):
        os.remove("{}/{}".format(args.tmp, f))

    os.rmdir(args.tmp)


type_dict = {}
with open(os.path.join(sharePath, "atomtypes.dat")) as f:
    for line in f:
        if line[0] != '#':
            elements = line.split()
            type_dict[elements[0]] = elements[1]


def writeXYZ(filename, obj, atom_names):
    """Writes the positions of the given MDAnalysis object to the given file"""
    write = mda.coordinates.XYZ.XYZWriter(filename,
                                          n_atoms=len(atom_names),
                                          atoms=atom_names)

    ts = obj.universe.trajectory.ts.copy_slice(obj.indices)
    write.write_next_timestep(ts)
    write.close()

#======= PREPERATIONS =======
#============================


def main(firstarg=2):
    global args

    args = parser.parse_args(args=sys.argv[firstarg:])
    u = initilize_universe(args)

    sel = u.select_atoms(args.sel + " and not name DUM and not name MW")

    print("Selection '{}' contains {} atoms.\n".format(args.sel, sel.n_atoms))
    if sel.n_atoms == 0:
        sys.exit("Exiting since selection does not contain any atoms.")

    # Create an extra list for the atom names.
    # This is necessary since it is not possible to efficently add axtra atoms to
    # a MDAnalysis universe, necessary for the hydrogens in united atom forcefields.

    atom_names = sel.n_atoms * ['']

    for i, atom_type in enumerate(sel.types.astype(str)):
        element = type_dict[atom_type]

        # add hydrogens in the case of united atom forcefields
        if element in ["CH1", "CH2", "CH3", "CH4", "NH", "NH2", "NH3"]:
            atom_names[i] = element[0]
            for h in range(int(element[-1])):
                atom_names.append("H")
                # add a extra atom to universe. It got the wrong type but we only
                # need the position, since we maintain our own atom type list.
                sel += sel.atoms[i]
        else:
            atom_names[i] = element

    # create tmp directory for saving datafiles
    args.tmp = tempfile.mkdtemp()

    if args.verbose:
        OUT = None
        print("{} is the tempory directory for all files.".format(args.tmp))
    else:
        OUT = open(os.devnull, 'w')

    args.frame = 0
    for ts in u.trajectory[args.beginframe:args.endframe + 1:args.skipframes]:

        # convert coordinates in a rectengular box
        box = np.diag(mda.lib.mdamath.triclinic_vectors(ts.dimensions))
        sel.atoms.positions = sel.atoms.positions \
            - box * np.round(sel.atoms.positions / box)  # minimum image

        writeXYZ("{}/{}.xyz".format(args.tmp, args.frame),
                 sel.atoms, atom_names)

        ref_q = 4 * np.pi / np.min(box)
        if ref_q > args.startq:
            args.startq = ref_q

        command = "-x -f {0} -t {1} -s {2} -o {3}/{4}.dat -a {5} -b {6} -c {7} -r {8} {3}/{4}.xyz".format(
            round(args.startq, 3), args.endq, args.dq, args.tmp, args.frame,
            box[0], box[1], box[2], np.min(box) / 2.001)

        subprocess.call("{} {}".format(args.debyer, command),
                        stdout=OUT, stderr=OUT, shell=True)

        args.frame += 1
        print_frameinfo(ts, args.frame)
        # call for output
        if args.frame % args.outfreq == 0 and args.frame > 0:
            output()

    output()
    cleanup()
    print("\n")


if __name__ == "__main__":
    main(firstarg=1)
