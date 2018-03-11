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

from .. import sharePath

#========== PARSER ===========
#=============================
parser = argparse.ArgumentParser(description="""
    An interface to the Deyer library. By using the -sel option atoms can be selected for which the
    profile is calculated. The selection uses the MDAnalysis selection commands found here:
    http://www.mdanalysis.org/docs/documentation_pages/selections.html
    The system can be replicated with the -nbox option. The system is than stacked multiplie times on itself. No
    replication is done be default.""", prog="mdtools debyer",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-s',     dest='topology',    type=str,
                    default='topol.tpr',            help='the topolgy file')
parser.add_argument('-f',     dest='trajectory',  type=str,     default=[
                    'traj.xtc'], nargs='+', help='A single or multiple trajectory files.')
parser.add_argument('-sel',   dest='sel',         type=str,     default='all',
                    help='Atoms for which to compute the profile', )
parser.add_argument('-b',     dest='begin',       type=float,   default=0,
                    help='First frame (ps) to read from trajectory')
parser.add_argument('-e',     dest='end',         type=float,   default=None,
                    help='Last frame (ps) to read from trajectory')
parser.add_argument('-skip',  dest='skipframes',  type=int,
                    default=1,                      help='Evaluate every Nth frames')
parser.add_argument('-dout',  dest='outfreq',     type=float,   default='100',
                    help='Number of frames after which the output is updated.')
parser.add_argument('-sq',    dest='output',      type=str,
                    default='./sq',                 help='Prefix/Path for output file')
parser.add_argument('-startq', dest='startq',      type=float,
                    default=0,                      help='Starting q (1/A)')
parser.add_argument('-endq',  dest='endq',        type=float,
                    default=60,                     help='Ending q (1/A)')
parser.add_argument('-dq',    dest='dq',          type=float,
                    default=0.02,                   help='binwidth (1/A)')
parser.add_argument('-d',     dest='debyer',      type=str,
                    default="~/repos/debyer/debyer/debyer", help='path to the debyer executable')
parser.add_argument('-v',     dest='verbose',     action='store_true',
                    help='Be loud and noisy.')


def cleanup():
    """averages over all dat file and removes them"""
    datfiles = [f for f in os.listdir(tmp) if f.endswith(".dat")]
    xyzfiles = [f for f in os.listdir(tmp) if f.endswith(".xyz")]

    for i, f in enumerate(datfiles):
        path = "{}/{}".format(tmp, f)
        if i == 0:
            s_tmp = np.loadtxt(path)
        else:
            s_tmp = np.vstack([s_tmp, np.loadtxt(path)])

        os.remove(path)
        os.remove("{}/{}".format(tmp, xyzfiles[i]))

    if frames > args.outfreq:
        s_prev = np.loadtxt("{}.dat".format(args.output))
        s_prev[:, 1] *= (frames - len(datfiles))  # weighting for average
        s_tmp = np.vstack([s_tmp, s_prev])

    s_out = binned_statistic(
        s_tmp[:, 0], s_tmp[:, 1], bins=nbins, range=(args.startq, args.endq))[0]
    s_out = np.nan_to_num(s_out)

    nonzeros = np.where(s_out != 0)[0]

    np.savetxt(args.output + '.dat',
               np.vstack([q[nonzeros], s_out[nonzeros]]).T,
               header="q (1/A)\tS(q)_tot (arb. units)", fmt='%.8e')


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
    args = parser.parse_args(args=sys.argv[firstarg:])

    print("Loading trajectory...\n")
    u = mda.Universe(args.topology, args.trajectory)
    sel = u.select_atoms(args.sel)

    if sel.n_atoms == 0:
        sys.exit("Exiting since selection does not contain any atoms.")

    # Create an extra list for the atom names.
    # This is necessary since it is not possible to efficently add axtra atoms to
    # a MDAnalysis universe, necessary for the hydrogens in united atom forcefields.

    atom_names = sel.n_atoms * ['']

    for i, atom_type in enumerate(sel.types.astype(str)):
        element = type_dict[atom_type]

        if element is not "DUM":
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

    #sel = sel.atoms.select_atoms("not name 'DUM'")

    startq = args.startq
    dt = u.trajectory.dt

    begin = int(args.begin // dt)
    if args.end != None:
        end = int(args.end // dt)
    else:
        end = int(u.trajectory.totaltime // u.trajectory.dt)

    if begin > end:
        print("Start time is larger than end time!")

    maxnframes = -begin
    if begin > 0:
        maxnframes += end

    maxnframes /= args.skipframes

    # create tmp directory for saving datafiles
    tmp = tempfile.mkdtemp()

    if args.verbose:
        FNULL = None
        print("{} is the tempory directory for all files.".format(tmp))
    else:
        FNULL = open(os.devnull, 'w')

    nbins = int(np.ceil((args.endq - args.startq) / args.dq))
    q = np.arange(args.startq, args.endq, args.dq) + 0.5 * args.dq
    shift = np.zeros(3)

    frames = 0
    for ts in u.trajectory[begin:end + 1:args.skipframes]:

        # convert coordinates in a rectengular box
        box = np.diag(mda.lib.mdamath.triclinic_vectors(ts.dimensions))
        sel.atoms.positions = sel.atoms.positions \
            - box * np.round(sel.atoms.positions / box)  # minimum image

        writeXYZ("{}/{}.xyz".format(tmp, frames), sel.atoms, atom_names)

        ref_q = 4 * np.pi / np.min(box)
        if ref_q > args.startq:
            startq = ref_q

        command = "-x -f {0} -t {1} -s {2} -o {3}/{4}.dat -a {5} -b {6} -c {7} -r {8} {3}/{4}.xyz".format(
            round(startq, 3), args.endq, args.dq, tmp, frames,
            box[0], box[1], box[2], np.min(box) / 2)

        subprocess.call("{} {}".format(args.debyer, command),
                        stdout=FNULL, stderr=FNULL, shell=True)

        frames += 1
        if (frames < 100):
            print("\rEvaluating frame: {:>12} time: {:>12} ps".format(
                frames, round(ts.time)), end="")
        elif (frames < 1000 and frames % 10 == 1):
            print("\rEvaluating frame: {:>12} time: {:>12} ps".format(
                frames, round(ts.time)), end="")
        elif (frames % 250 == 1):
            print("\rEvaluating frame: {:>12} time: {:>12} ps".format(
                frames, round(ts.time)), end="")
        # call for output
        if (frames % args.outfreq == 0) and frames < maxnframes:
            cleanup()
        sys.stdout.flush()

    cleanup()
    os.rmdir(tmp)
    print("\n")


if __name__ == "__main__":
    main(firstarg=1)
