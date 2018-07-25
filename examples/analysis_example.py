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
parser.description = """Description for my awesome analysis script."""

# Custom arguments
parser.add_argument('-o',   dest='output',      type=str,
                    default='ouput file', help='Prefix for output filenames')
parser.add_argument('-T',   dest='temperetaure',      type=float,
                    default=300, help='Reference temperature')


# Custom functions
def foo(bar=None):
    """A function that prints the temperature and returns the argument."""
    print(args.temperature)
    return bar

# ========== MAIN ============
# ============================


def main(firstarg=2, DEBUG=False):
    # Not essential but nice to use args also in custo functions without passing
    # explicitly
    global args

    # parse the arguments and saves them in an args object
    args = parser.parse_args(args=sys.argv[firstarg:])

    # the MDAnalysis universe given by the user for analysis
    u = initilize_universe(args)

    # Custom variables needed for evaluation and initial calculations if needed
    Volume = 0

    # ======== MAIN LOOP =========
    # ============================
    t_0 = time.clock()
    args.frame = 0
    print("\rEvaluating frame: {:>12} time: {:>12} ps".format(
        args.frame, round(u.trajectory.time)), end="")
    for ts in u.trajectory[args.beginframe:args.endframe:args.skipframes]:

        # Calculations done in every frame
        Volume += ts.volume

        args.frame += 1
        print_frameinfo(ts, args.frame)

    t_end = time.clock()
    print("\n")

    # Final calculations i.e. printing informations and call for output
    print("Calculation took {:.2f} seconds.".format(t_end - t_0))
    print("Average volume of the simulation box {:.2f} Å".format(
        Volume / args.frame))

    if DEBUG:
        # Inject local variables into global namespace for debugging.
        for key, value in locals().items():
            globals()[key] = value

if __name__ == "__main__":
    main(firstarg=1)
