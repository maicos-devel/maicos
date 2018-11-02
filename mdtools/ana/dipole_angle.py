#!/usr/bin/env python
# coding: utf-8

# ========== DESCRIPTION ===========
# This is an example for an analysis script. To use this
# script do the following steps:
# 1. Copy it to the "mdtsools/ana" folder and add your code.
# 2. Choose an unique name and add <"analysis_example": "ana">
#    to the apps dictionary in "mdtools/__main__.py".
# 3. OPTIONAL: Add bash completion commands to "mdtools/share/mdtools_completion.bash".
# ==================================

# Mandatory imports
from __future__ import absolute_import, division, print_function

import argparse
import os
import sys

import MDAnalysis as mda
import numpy as np

from . import initilize_universe, print_frameinfo
from .. import initilize_parser

# Custom import modules
import time

# ========== PARSER ===========
# =============================
# parser object will already contain options for
# the topology, trajectory, begin, end, skipped frames and the box dimenions
parser = initilize_parser(add_traj_arguments=True)
parser.description = """Calculates the timeseries of the dipole moment wit an axis."""

# Custom arguments
parser.add_argument('-d', dest='dim', type=int,
                    default=2, help='direction normal to the surface (x,y,z=0,1,2, default: z)')
parser.add_argument('-sel', dest='sel', type=str, help='atom group selection', default='resname SOL')
parser.add_argument('-dout', dest='outfreq', type=float,
                    default='10000', help='Default number of frames after which output files are refreshed (10000)')
parser.add_argument('-o',   dest='output',      type=str,
                    default='dipangle', help='Prefix for output filenames')

# ========== MAIN ============
# ============================

def output(cos_theta):
    np.save(args.output, cos_theta)

def main(firstarg=2, DEBUG=False):
    global args

    # parse the arguments and saves them in an args object
    args = parser.parse_args(args=sys.argv[firstarg:])

    # the MDAnalysis universe given by the user for analysis
    u = initilize_universe(args)

    sol = u.select_atoms(args.sel)
    atomsPerMolecule = sol.n_atoms // sol.n_residues

    # unit normal vector
    unit = np.zeros(3)
    unit[args.dim] += 1

    cos_theta = np.empty(args.n_frames)

    # ======== MAIN LOOP =========
    # ============================
    for args.frame, ts in enumerate(u.trajectory[args.beginframe:args.endframe:args.skipframes]):
        print_frameinfo(ts, args.frame)
        
        chargepos = sol.atoms.positions * sol.atoms.charges[:, np.newaxis]
        dipoles = np.sum(list(chargepos[i::atomsPerMolecule] for i in range(atomsPerMolecule)), axis=0)

        cos_theta[args.frame] = (np.dot(dipoles, unit) / np.linalg.norm(dipoles)).mean()
        
        if (args.frame % args.outfreq == 0 and args.frame >= args.outfreq):
            output(cos_theta)

        print("\n")
        output(cos_theta)
        
    if DEBUG:
        # Inject local variables into global namespace for debugging.
        for key, value in locals().items():
            globals()[key] = value

if __name__ == "__main__":
    main(firstarg=1)
