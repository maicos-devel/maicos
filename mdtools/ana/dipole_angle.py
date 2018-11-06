#!/usr/bin/env python
# coding: utf-8

# Mandatory imports
from __future__ import absolute_import, division, print_function

import argparse
import sys

import MDAnalysis as mda
import numpy as np

from . import initilize_universe, print_frameinfo
from .. import initilize_parser
from ..utils import repairMolecules

# ========== PARSER ===========
# =============================
parser = initilize_parser(add_traj_arguments=True)
parser.description = """Calculates the timeseries of the dipole moment wit an axis."""

# Custom arguments
parser.add_argument('-d', dest='dim', type=int,
                    default=2, help='direction normal to the surface (x,y,z=0,1,2, default: z)')
parser.add_argument('-sel', dest='sel', type=str,
                    help='atom group selection', default='resname SOL')
parser.add_argument('-dout', dest='outfreq', type=float,
                    default='10000', help='Default number of frames after which output files are refreshed (10000)')
parser.add_argument('-o',   dest='output',      type=str,
                    default='dipangle', help='Prefix for output filenames')

# ========== MAIN ============
# ============================


def output(t, cos_theta_i, cos_theta_ii, cos_theta_ij):
    np.savetxt("{}.dat".format(args.output),
               np.vstack([t, cos_theta_i, cos_theta_ii, cos_theta_ij]).T,
               header="t\t<cos(θ_i)>\t<cos(θ_i)cos(θ_i)>\t<cos(θ_i)cos(θ_j)>",
               fmt='%.5e')


def main(firstarg=2, DEBUG=False):
    global args

    # parse the arguments and saves them in an args object
    args = parser.parse_args(args=sys.argv[firstarg:])

    # the MDAnalysis universe given by the user for analysis
    u = initilize_universe(args)

    sol = u.select_atoms(args.sel)
    
    n_residues = sol.residues.n_residues
    atomsPerMolecule = sol.n_atoms // n_residues

    # unit normal vector
    unit = np.zeros(3)
    unit[args.dim] += 1

    dt = args.dt * args.skipframes

    t = (np.arange(args.beginframe, args.endframe) - args.beginframe) * dt
    cos_theta_i = np.empty(args.n_frames)
    cos_theta_ii = np.empty(args.n_frames)
    cos_theta_ij = np.empty(args.n_frames)

    # ======== MAIN LOOP =========
    # ============================
    for args.frame, ts in enumerate(u.trajectory[args.beginframe:args.endframe:args.skipframes]):
        print_frameinfo(ts, args.frame)

        # make broken molecules whole again!
        repairMolecules(u)

        chargepos = sol.atoms.positions * sol.atoms.charges[:, np.newaxis]
        dipoles = np.sum(list(chargepos[i::atomsPerMolecule]
                              for i in range(atomsPerMolecule)), axis=0)

        cos_theta = np.dot(dipoles, unit) / np.linalg.norm(dipoles, axis=1)
        matrix = np.outer(cos_theta, cos_theta)

        trace = matrix.trace()
        cos_theta_i[args.frame] = cos_theta.mean()
        cos_theta_ii[args.frame] = trace / n_residues
        cos_theta_ij[args.frame] = (matrix.sum() - trace) / (n_residues**2 - n_residues)

        if (args.frame % args.outfreq == 0 and args.frame >= args.outfreq):
            output(t, cos_theta_i, cos_theta_ii, cos_theta_ij)

    print("\n")
    output(t, cos_theta_i, cos_theta_ii, cos_theta_ij)

    if DEBUG:
        # Inject local variables into global namespace for debugging.
        for key, value in locals().items():
            globals()[key] = value


if __name__ == "__main__":
    main(firstarg=1)
