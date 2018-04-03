#!/usr/bin/env python
# coding: utf-8

from __future__ import print_function, division

import argparse
import sys

import MDAnalysis
import numpy as np

import pbctools

from . import initilize_universe, print_frameinfo
from .. import initilize_parser

# parse command line options

parser = initilize_parser(add_traj_arguments=True)
parser.description="""Calculate dipolar order parameters."""
parser.add_argument('-dz', dest='binwidth', type=float,
                    default=0.01, help='specify the binwidth [nm]')
parser.add_argument('-d', dest='dim', type=int,
                    default=2, help='direction normal to the surface (x,y,z=0,1,2, default: z)')
parser.add_argument('-o', dest='output', type=str,
                    default='diporder', help='Prefix for output filenames')
parser.add_argument('-sel', dest='sel', type=str,
                    help='atom group selection',
                    default='resname SOL')
parser.add_argument('-dout', dest='outfreq', type=float,
                    default='10000', help='Default number of frames after which output files are refreshed (10000)')
parser.add_argument('-sym', dest='bsym', action='store_const',
                    const=True, default=False,
                    help='symmetrize the profiles')
parser.add_argument('-shift', dest='membrane_shift', action='store_const',
                    const=True, default=False,
                    help='shift system by half a box length (useful for membrane simulations)')
parser.add_argument('-com', dest='com', action='store_const',
                    const=True, default=False,
                    help='shift system such that the water COM is centered')
parser.add_argument('-bin', dest='binmethod', type=str,
                    default='COM',
                    help='binning method: center of Mass (COM), center of charge (COC) or oxygen position (OXY)')


def output(diporder, av_box_length):
    av_box_length_ts = av_box_length / args.frame / 10

    z = np.linspace(-av_box_length_ts / 2, av_box_length_ts / 2, len(diporder),
                    endpoint=False) + av_box_length_ts / args.nbins / 2

    outdata = np.vstack(
        [z, diporder[:, 0] / args.frame, diporder[:, 1] / args.frame, diporder[:, 2] / args.frame]).T

    if (args.bsym):
        for i in range(len(outdata) - 1):
            outdata[i + 1] = .5 * (outdata[i + 1] + outdata[i + 1][-1::-1])

    np.savetxt(args.output + '.dat', outdata,
               header="z\tP_0 rho(z) cos(Theta(z))\tcos(Theta(z))\trho(z)")

    return


def main(firstarg=2):
    global args

    args = parser.parse_args(args=sys.argv[firstarg:])
    u = initilize_universe(args)

    sol = u.select_atoms(args.sel)
    atomsPerMolecule = sol.n_atoms // sol.n_residues

    # Assume a threedimensional universe...
    xydims = np.roll(np.arange(3), -args.dim)[1:]
    dz = args.binwidth * 10  # Convert to Angstroms
    # CAVE: binwidth varies in NPT !
    args.nbins = int(u.dimensions[args.dim] / dz)

    '''
    data structure:
        diporder saves the (summed) projected polarization density:
            0 P_0 rho(z) cos(Theta(z))
            1 cos(Theta(z))
            2 rho(z)
    '''

    diporder = np.zeros((args.nbins, 3))
    av_box_length = 0

    # unit normal vector
    unit = np.zeros(3)
    unit[args.dim] += 1

    print('Using', args.nbins, 'bins.')

    args.frame = 0
    print("Evaluating frame: ", u.trajectory.frame, "\ttime: ", int(u.trajectory.time), end="")

    for ts in u.trajectory[args.beginframe:args.endframe + 1:args.skipframes]:

        if args.membrane_shift:
            # shift membrane
            ts.positions[:, args.dim] += ts.dimensions[args.dim] / 2
            ts.positions[:, args.dim] %= ts.dimensions[args.dim]
        if args.com:
            # put water COM into center
            waterCOM = np.sum(
                sol.atoms.positions[:, 2] * sol.atoms.masses) / sol.atoms.masses.sum()
            ts.positions[:, args.dim] += ts.dimensions[args.dim] / 2 - waterCOM
            ts.positions[:, args.dim] %= ts.dimensions[args.dim]

        # make broken molecules whole again!
        pbctools.repairMolecules(u)

        A = np.prod(ts.dimensions[xydims])
        dz_frame = ts.dimensions[args.dim] / args.nbins

        chargepos = sol.atoms.positions * sol.atoms.charges[:, np.newaxis]
        dipoles = np.sum(chargepos[i::atomsPerMolecule] for i in range(
            atomsPerMolecule)) / 10  # convert to e nm

        if args.binmethod == 'COM':
            # Calculate the centers of the objects ( i.e. Molecules )
            masses = np.sum(sol.atoms.masses[i::atomsPerMolecule]
                            for i in range(atomsPerMolecule))
            masspos = sol.atoms.positions * sol.atoms.masses[:, np.newaxis]
            coms = np.sum(masspos[i::atomsPerMolecule] for i in range(
                atomsPerMolecule)) / masses[:, np.newaxis]
            bins = ((coms[:, args.dim] %
                     ts.dimensions[args.dim]) / dz_frame).astype(int)
        elif args.binmethod == 'COC':
            abschargepos = sol.atoms.positions * \
                np.abs(sol.atoms.charges)[:, np.newaxis]
            charges = np.sum(np.abs(sol.atoms.charges)[
                             i::atomsPerMolecule] for i in range(atomsPerMolecule))
            cocs = np.sum(abschargepos[i::atomsPerMolecule] for i in range(
                atomsPerMolecule)) / charges[:, np.newaxis]
            bins = ((cocs[:, args.dim] %
                     ts.dimensions[args.dim]) / dz_frame).astype(int)
        elif args.binmethod == 'OXY':
            bins = ((sol.atoms.positions[::3, args.dim] %
                     ts.dimensions[args.dim]) / dz_frame).astype(int)
        else:
            raise ValueError('Unknown binning method: %s' % args.binmethod)

        bincount = np.bincount(bins, minlength=args.nbins)

        diporder[:, 0] += np.histogram(bins, bins=np.arange(args.nbins + 1),
                                       weights=np.dot(dipoles, unit))[0] / (A * dz_frame / 1e3)
        with np.errstate(divide='ignore', invalid='ignore'):
            diporder[:, 1] += np.nan_to_num(np.histogram(bins, bins=np.arange(args.nbins + 1),
                                                         weights=np.dot(dipoles / np.linalg.norm(dipoles, axis=1)[:, np.newaxis], unit))[0] / bincount)
        diporder[:, 2] += bincount / (A * dz_frame / 1e3)

        av_box_length += ts.dimensions[args.dim]

        args.frame += 1
        print_frameinfo(ts, args.frame)
        # call for output
        if (args.frame % args.outfreq == 0 and args.frame >= args.outfreq):
            output(diporder, av_box_length)

    print('\n')
    output(diporder, av_box_length)


if __name__ == "__main__":
    main(firstarg=1)
