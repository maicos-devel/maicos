#!/usr/bin/env python
# coding: utf-8

# Mandatory imports
from __future__ import absolute_import, division, print_function

import argparse
import os
import sys

import MDAnalysis as mda
import numpy as np

from . import initilize_universe, print_frameinfo
from .. import initilize_parser


# ========== PARSER ===========
# =============================
# parser object will already contain options for
# the topology, trajectory, begin, end, skipped frames and the box dimenions
parser = initilize_parser(add_traj_arguments=True)
parser.description = """Calculates the timeseries for the molecular center of 
mass translational and rotational kinetic energy."""

parser.add_argument('-o',   dest='output',      type=str,
                    default='ke', help='Prefix for output filenames')    

def output(t, E_kin, E_cm):
    # Call for output
    trans = E_cm / 2
    rot = (E_kin - E_cm) / 2
    
    np.savetxt("{}.dat".format(args.output),
                np.vstack([t, trans, rot]).T, fmt='%.8e',
                header="t / ps \t E_kin^trans \t E_kin^rot")
                
# ========== MAIN ============
# ============================
def main(firstarg=2, DEBUG=False):
    # Not essential but nice to use args also in custom functions without passing
    # explicitly
    global args

    # parse the arguments and saves them in an args object
    args = parser.parse_args(args=sys.argv[firstarg:])

    # the MDAnalysis universe given by the user for analysis
    u = initilize_universe(args)

    seg_masses = []
    for seg in u.segments:
        seg_masses.append(u.segments[0].residues.masses)

    # Total kinetic energy
    E_kin = np.empty(args.n_frames)
    
    # Molecular center of mass kinetic energy
    E_cm  = np.empty(args.n_frames)
    
    dt = args.dt * args.skipframes
    t = (np.arange(args.beginframe, args.endframe) - args.beginframe) * dt
    # ======== MAIN LOOP =========
    # ============================
    for args.frame, ts in enumerate(u.trajectory[args.beginframe:args.endframe:args.skipframes]):
        print_frameinfo(ts, args.frame)
        
        massvel = u.atoms.velocities * u.atoms.masses[:, np.newaxis]

        E_kin[args.frame] = np.dot(u.atoms.masses, np.linalg.norm(u.atoms.velocities,axis=1)**2)
        
        for j, seg in enumerate(u.segments):
            
            atomsPerMolecule = seg.atoms.n_atoms // seg.atoms.n_residues
            massvel = seg.atoms.velocities * seg.atoms.masses[:, np.newaxis]
            v_cm = np.sum(list(massvel[i::atomsPerMolecule] for i in range(atomsPerMolecule)), axis=0)
            v_cm /= seg_masses[j][:,np.newaxis]
        
            E_cm[args.frame] += np.dot(seg_masses[j], np.linalg.norm(v_cm, axis=1)**2)

    print("\n")
    output(t, E_kin, E_cm)

    if DEBUG:
        # Inject local variables into global namespace for debugging.
        for key, value in locals().items():
            globals()[key] = value

if __name__ == "__main__":
    main(firstarg=1)
