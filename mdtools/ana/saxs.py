#!/usr/bin/env python
# coding: utf-8

from __future__ import absolute_import, division, print_function

import math
import os
import sys

import MDAnalysis as mda
import numpy as np

import numba as nb

from . import initilize_universe, print_frameinfo
from .. import initilize_parser, sharePath

# ========== PARSER ===========
# =============================
parser = initilize_parser(add_traj_arguments=True)
parser.description = """
    Computes SAXS scattering intensities for all atom types from the given trajectory. The possible scattering
    vectors q can be restricted by a miminal and maximal angle with the z-axis. For 0 and 180 all possible vectors
    are taken into account.
    For the scattering factor the structure fator is multiplied by a atom type specific form factor
    based on Cromer-Mann parameters. By using the -sel option atoms can be selected for which the
    profile is calculated. The selection uses the MDAnalysis selection commands found here:
    http://www.mdanalysis.org/docs/documentation_pages/selections.html"""
parser.add_argument('-sel',   dest='sel',         type=str,   default='all',
                    help='Atoms for which to compute the profile', )
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
parser.add_argument('-mintheta', dest='mintheta',      type=float,
                    default=0,                      help='Minimal angle (°) between the q vectors and the z-axis.')
parser.add_argument('-maxtheta',  dest='maxtheta',        type=float,
                    default=180,         help='Maximal angle (°) between the q vectors and the z-axis.')

# ======== DEFINITIONS ========
# =============================


def output(q, struct_factor, n_atoms):
    """Saves the current profiles to a file."""
    nonzeros = np.where(struct_factor[:, 0] != 0)[0]
    scat_factor = struct_factor[nonzeros]
    wave_vectors = q[nonzeros]

    scat_factor = scat_factor.sum(axis=1) / (args.frame * n_atoms)

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


@nb.jit(nb.types.UniTuple(nb.float32[:, :, :], 2)(nb.float32[:, :], nb.float32[:], nb.float32, nb.float32, nb.float32, nb.float32),
        nopython=True, nogil=True, parallel=True)
def compute_structure_factor(positions, boxdimensions,  start_q, end_q, mintheta, maxtheta):
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


# =========== MAIN ===========
# ============================
def main(firstarg=2, DEBUG=False):
    global args

    args = parser.parse_args(args=sys.argv[firstarg:])

    args.mintheta = min(args.mintheta, args.maxtheta)
    args.maxtheta = max(args.mintheta, args.maxtheta)

    if args.mintheta < 0:
        print("mintheta = {}° < 0°: Set mininmal angle to 0°.".format(args.mintheta))
        args.mintheta = 0
    if args.maxtheta > 180:
        print("maxtheta = {}° > 180°: Set maximal angle to 180°.".format(args.maxtheta))
        args.maxtheta = np.pi

    args.mintheta *= np.pi / 180
    args.maxtheta *= np.pi / 180

    u = initilize_universe(args)
    sel = u.select_atoms(args.sel)

    print("\nSelection '{}' contains {} atoms.".format(args.sel, sel.n_atoms))
    if sel.n_atoms == 0:
        sys.exit("Exiting since selection does not contain any atoms.")

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

    args.nbins = int(np.ceil((args.endq - args.startq) / args.dq))
    q = np.arange(args.startq, args.endq, args.dq) + 0.5 * args.dq
    struct_factor = np.zeros([args.nbins, len(groups)])

    # ======== MAIN LOOP =========
    # ============================
    for args.frame, ts in enumerate(u.trajectory[args.beginframe:args.endframe:args.skipframes]):
        print_frameinfo(ts, args.frame)

        for i, t in enumerate(groups):

            # convert everything to cartesian coordinates
            box = np.diag(mda.lib.mdamath.triclinic_vectors(ts.dimensions))
            positions = t.atoms.positions - box * \
                np.round(t.atoms.positions / box)  # minimum image

            q_ts, S_ts = compute_structure_factor(positions / 10, box / 10,
                                                  args.startq, args.endq,
                                                  args.mintheta, args.maxtheta)

            q_ts = q_ts.flatten()
            S_ts = S_ts.flatten()
            nonzeros = np.where(S_ts != 0)[0]

            q_ts = q_ts[nonzeros]
            S_ts = S_ts[nonzeros]

            S_ts *= compute_form_factor(q_ts, atom_types[i])**2

            bins = ((q_ts - args.startq) /
                    ((args.endq - args.startq) / args.nbins)).astype(int)
            struct_ts = np.histogram(bins, bins=np.arange(args.nbins + 1),
                                     weights=S_ts)[0]
            with np.errstate(divide='ignore', invalid='ignore'):
                struct_ts /= np.bincount(bins, minlength=args.nbins)
            struct_factor[:, i] += np.nan_to_num(struct_ts)

        # call for output
        if (int(ts.time) % args.outfreq == 0 and ts.time - args.begin >= args.outfreq):
            output(q, struct_factor, sel.atoms.n_atoms)

    output(q, struct_factor, sel.atoms.n_atoms)
    print("\n")


    if DEBUG:
        # Inject local variables into global namespace for debugging.
        for key, value in locals().items():
            globals()[key] = value

if __name__ == "__main__":
    main(firstarg=1)
