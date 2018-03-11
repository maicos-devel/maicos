#!/usr/bin/env python

from __future__ import division, print_function

import argparse
import os
import sys

import MDAnalysis as mda
import numpy as np
from scipy.stats import binned_statistic

import sfactor

from .. import sharePath

#========== PARSER ===========
#=============================
parser = argparse.ArgumentParser(description="""
    Computes SAXS scattering intensities for all atom types from the given trajectory.
    For the scattering factor the structure fator is multiplied by a atom type specific form factor
    based on Cromer-Mann parameters. By using the -sel option atoms can be selected for which the
    profile is calculated. The selection uses the MDAnalysis selection commands found here:
    http://www.mdanalysis.org/docs/documentation_pages/selections.html""",
    prog = "mdtools epsilon_bulk", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-s',     dest='topology',    type=str,
                    default='topol.tpr',            help='the topolgy file')
parser.add_argument('-f',     dest='trajectory',  type=str,   default=[
                    'traj.xtc'], nargs='+', help='A single or multiple trajectory files.')
parser.add_argument('-sel',   dest='sel',         type=str,   default='all',
                    help='Atoms for which to compute the profile', )
parser.add_argument('-b',     dest='begin',       type=float, default=0,
                    help='First frame (ps) to read from trajectory')
parser.add_argument('-e',     dest='end',         type=float, default=None,
                    help='Last frame (ps) to read from trajectory')
parser.add_argument('-skip',  dest='skipframes',  type=int,
                    default=1,                      help='Evaluate every Nth frames')
parser.add_argument('-dout',  dest='outfreq',     type=float, default='100',
                    help='Number of frames after which the output is updated.')
parser.add_argument('-sq',    dest='output',      type=str,
                    default='./sq',                 help='Prefix/Path for output file')
parser.add_argument('-startq', dest='startq',      type=float,
                    default=0,                      help='Starting q (1/nm)')
parser.add_argument('-endq',  dest='endq',        type=float,
                    default=60,                     help='Ending q (1/nm)')
parser.add_argument('-dq',    dest='dq',          type=float,
                    default=0.05,                   help='binwidth (1/nm)')

#======== DEFINITIONS ========
#=============================


def output():
    """Saves the current profiles to a file."""
    nonzeros = np.where(struct_factor[:, 0] != 0)[0]
    scat_factor = struct_factor[nonzeros]
    wave_vectors = q[nonzeros]

    scat_factor = scat_factor.sum(axis=1) / frames

    np.savetxt(args.output + '.dat',
               np.vstack([wave_vectors, scat_factor]).T,
               header="q (1/nm)\tS(q)_tot (arb. units)", fmt='%.8e')


def compute_form_factor(q, atom_type):
    """Calculates the form factor for the given element. Correctly handles united atom types like CH4 etc..."""
    element = type_dict[atom_type]

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
        form_factor = CM_parameters[element].c
        # factor of 10 to convert from 1/nm to 1/Angstroms
        q2 = (q / (4 * np.pi * 10))**2
        for i in range(4):
            form_factor += CM_parameters[element].a[i] * \
                np.exp(-CM_parameters[element].b[i] * q2)

    return form_factor


type_dict = {}
with open(os.path.join(sharePath, "atomtypes.dat")) as f:
    for line in f:
        if line[0] != '#':
            elements = line.split()
            type_dict[elements[0]] = elements[1]

CM_parameters = {}
with open(os.path.join(sharePath, "sfactor.dat")) as f:
    for line in f:
        if line[0] != '#':
            elements = line.split()
            CM_parameters[elements[0]] = type('CM_parameter', (object,), {})()
            CM_parameters[elements[0]].a = np.array(
                elements[2:6], dtype=np.double)
            CM_parameters[elements[0]].b = np.array(
                elements[6:10], dtype=np.double)
            CM_parameters[elements[0]].c = float(elements[10])


#=========== MAIN ===========
#============================
def main(firstarg=2):
    args = parser.parse_args(args=sys.argv[firstarg:])

    print("Loading trajectory...\n")
    u = mda.Universe(args.topology, args.trajectory)
    sel = u.select_atoms(args.sel)

    groups = []
    atom_types = []
    print("\nMap the following atomtypes:")
    for atom_type in np.unique(sel.atoms.types).astype(str):
        try:
            element = type_dict[atom_type]
        except KeyError:
            sys.exit(
                "No suitable element for '{0}' found. You can add '{0}' together with a suitable element to 'share/atomtypes.dat'.".format(atom_type))

        if element == "DUM":
            continue
        groups.append(u.select_atoms("type {}*".format(atom_type)))
        atom_types.append(atom_type)
        print("{:>14} --> {:>5}".format(atom_type, element))

    print("\n")
    dt = u.trajectory.dt

    begin = int(args.begin // dt)
    if args.end != None:
        end = int(args.end // dt)
    else:
        end = int(u.trajectory.totaltime // u.trajectory.dt)

    if begin > end:
        print("Start time is larger than end time!")

    nbins = int(np.ceil((args.endq - args.startq) / args.dq))
    q = np.arange(args.startq, args.endq, args.dq) + 0.5 * args.dq
    struct_factor = np.zeros([nbins, len(groups)])
    frames = 0

    #======== MAIN LOOP =========
    #============================
    for ts in u.trajectory[begin:end + 1:args.skipframes]:
        for i, t in enumerate(groups):

            # convert everything to cartesian coordinates
            box = np.diag(mda.lib.mdamath.triclinic_vectors(ts.dimensions))
            positions = t.atoms.positions - box * \
                np.round(t.atoms.positions / box)  # minimum image

            q_ts, S_ts = sfactor.compute_structure_factor(
                np.double(positions / 10), np.double(box / 10), args.startq, args.endq)

            q_ts = np.asarray(q_ts).flatten()
            S_ts = np.asarray(S_ts).flatten()
            nonzeros = np.where(S_ts != 0)[0]

            q_ts = q_ts[nonzeros]
            S_ts = S_ts[nonzeros]

            S_ts *= compute_form_factor(q_ts, atom_types[i])**2

            struct_ts = binned_statistic(
                q_ts, S_ts, bins=nbins, range=(args.startq, args.endq))[0]
            struct_factor[:, i] += np.nan_to_num(struct_ts)

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
        if (frames % args.outfreq == 0):
            output()
        sys.stdout.flush()

    output()
    print("\n")


if __name__ == "__main__":
    main(firstarg=1)
