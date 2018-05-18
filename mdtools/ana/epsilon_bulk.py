#!/usr/bin/env python3
# coding: utf8

from __future__ import absolute_import, division, print_function

import argparse
import os
import sys

import MDAnalysis
import numpy as np

from . import initilize_universe, pbctools, print_frameinfo
from .. import initilize_parser

parser = initilize_parser(add_traj_arguments=True)
parser.description = """
          Computes the dipole moment flcutuations and from this the
          dielectric constant. The selection uses the MDAnalysis selection commands found here:
          http://www.mdanalysis.org/docs/documentation_pages/selections.html"""
parser.add_argument('-sel',         dest='sel',         type=str,     default='all',
                    help='Atoms for which to compute the profile', )
parser.add_argument('-dout',        dest='outfreq',     type=float,   default='100',
                    help='Number of frames after which the output is updated.')
parser.add_argument('-o',           dest='output',      type=str,
                    default='eps',                  help='Prefix for output filenames')
parser.add_argument('-temp',        dest='temperature', type=float,
                    default=300,                    help='temperature (K)')
parser.add_argument('-nopbcrepair', dest='bpbc',        action='store_false',
                    help='do not make broken molecules whole again (only works if molecule is smaller than shortest box vector')


def output(M, M2, V, verbose=False):

    epsilon_0 = 5.526350e-3  # ElementaryCharge (Angstroms Volts)^-1
    kb = 8.6173324e-5  # electronVolts Kelvins^-1
    T = args.temperature  # Kelvins

    fluct = M2 - M * M
    eps = fluct / (kb * T * V * epsilon_0)
    eps_mean = fluct.mean() / (kb * T * V * epsilon_0)

    if verbose:
        print("The following averages for the complete trajectory have been calculated:")

        print("")
        for i, d in enumerate("xyz"):
            print(" <M_{}> = {:.4f} eÅ".format(d, M[i]))

        print("")
        for i, d in enumerate("xyz"):
            print(" <M_{}²> = {:.4f} (eÅ)²".format(d, M2[i]))

        print("")
        print(" <|M|²> = {:.4f} (eÅ)²".format(M2.mean()))
        print(" |<M>|² = {:.4f} (eÅ)²".format((M * M).mean()))

        print("")
        print(" <|M|²> - |<M>|² = {:.4f} (eÅ)²".format(fluct.mean()))

        print("")
        for i, d in enumerate("xyz"):
            print(" ε_{} = {:.2f} ".format(d, eps[i]))

        print("")
        print(" ε = {:.2f}".format(eps_mean))
        print("")

    np.savetxt(args.output + '.dat', np.hstack([eps_mean, eps]).T,
               fmt='%1.2f', header='eps\teps_x\teps_y\teps_z')


def main(firstarg=2):
    global args

    args = parser.parse_args(args=sys.argv[firstarg:])
    u = initilize_universe(args)

    s = u.select_atoms(args.sel)
    print("There are {} atoms in the selection '{}'.".format(
        s.atoms.n_atoms, args.sel))

    M = np.zeros(3)
    M2 = np.zeros(3)
    V = 0

    # ======== MAIN LOOP =========
    # ============================
    args.frame = 0
    print("\rEvaluating frame: {:>12} time: {:>12} ps".format(
        args.frame, round(u.trajectory.time)), end="")
    for ts in u.trajectory[args.beginframe:args.endframe:args.skipframes]:

        if args.bpbc:
            pbctools.repairMolecules(u)

        M_ts = np.dot(s.atoms.charges, s.atoms.positions)
        M += M_ts
        M2 += M_ts * M_ts
        V += ts.volume

        args.frame += 1
        print_frameinfo(ts, args.frame)
        if (int(ts.time) % args.outfreq == 0 and ts.time - args.begin >= args.outfreq):
            output(M / args.frame, M2 / args.frame, V / args.frame)

    print("\n")
    output(M / args.frame, M2 / args.frame, V / args.frame, verbose=True)


if __name__ == "__main__":
    main(firstarg=1)
