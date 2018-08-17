#!/usr/bin/env python
# coding: utf-8

from __future__ import absolute_import, division, print_function

import os
import sys

import MDAnalysis as mda
import numpy as np

from . import initilize_universe, print_frameinfo
from .. import initilize_parser

# ========== PARSER ===========
# =============================
parser = initilize_parser(add_traj_arguments=True)
parser.description = """
    Computes partial densities across a cylinder of given radius r and length l.
    For group selections use strings in the MDAnalysis selection command style
    found here:
    https://pythonhosted.org/MDAnalysis/documentation_pages/selections.html"""
parser.add_argument('-o', dest='output', type=str,
                    default='density_cylinder', help='Prefix for output filenames')
parser.add_argument('-dout', dest='outfreq', type=float,
                    default='1000', help='Default time after which output files are refreshed (1000 ps).')
parser.add_argument('-d', dest='dim', type=int,
                    help='dimension for binning (0=X, 1=Y, 2=Z)', default=2)
parser.add_argument('-center', dest='center', type=str,
                    default=None, help="Perform the binning relative to the center of this group. If None center of box is used.")
parser.add_argument('-r', dest='radius', type=float,
                    default=None, help='Radius of the cylinder (nm). If None smallest box extension is taken.')
parser.add_argument('-dr', dest='binwidth', type=float,
                    default=1, help='binwidth (nm).')
parser.add_argument('-l', dest='length', type=float,
                    default=None, help='Length of the cylinder (nm). If None length of box in the binning dimension is taken.')
parser.add_argument('-dens', dest='density',     type=str,   default='mass',
                    choices=["mass", "number", "charge", "temp"], help='Density')
parser.add_argument('-gr', dest='groups', type=str, nargs='+', default=['all'],
                    help='Atoms for which to compute the density profile')


# ======== DEFINITIONS ========
# =============================

def output(density_mean, density_mean_sq):
    """Saves the current profiles to a file."""
    dens_mean = density_mean / \
        (np.pi * args.delta_r_sq[:, np.newaxis] * args.length * args.frame)
    dens_mean_sq = density_mean_sq / \
        ((np.pi * args.delta_r_sq[:, np.newaxis]
          * args.length)**2 * args.frame)

    dens_std = np.nan_to_num(np.sqrt(dens_mean_sq - dens_mean**2))
    dens_err = dens_std / np.sqrt(args.frame)

    # write header
    if args.density == "mass":
        units = "kg m^(-3)"
    elif args.density == "number":
        units = "nm^(-3)"
    elif args.density == "charge":
        units = "e nm^(-3)"

    columns = "{} density profile [{}]".format(args.density, units)
    columns += "\nstatistics over {:.1f} picoseconds \npositions [nm]".format(
        args.frame * args.dt)
    for group in args.groups:
        columns += "\t" + group
    for group in args.groups:
        columns += "\t" + group + " error"

    np.savetxt(args.output + '.dat',
               np.hstack(((args.r[:, np.newaxis]) / 10,
                          dens_mean * 1000, dens_err * 1000)),
               header=columns)


def weight(selection):
    """Calculates the weights for the histogram depending on the choosen type of density."""
    if args.density == "mass":
        # amu in kg -> kg/m^3
        return selection.atoms.masses * 1.66053892
    elif args.density == "number":
        return np.ones(selection.atoms.n_atoms)
    elif args.density == "charge":
        return selection.atoms.charges

# ========== MAIN ============
# ============================


def main(firstarg=2, DEBUG=False):
    global args

    args = parser.parse_args(args=sys.argv[firstarg:])
    u = initilize_universe(args)

    print (
        'Computing radial {} density profile along {}-axes.'.format(args.density, 'XYZ'[args.dim]))

    dim = args.dim
    odims = np.roll(np.arange(3), -dim)[1:]

    if args.center != None:
        center = u.select_atoms(args.center).center_of_mass()
    else:
        print("No center given --> Take from box dimensions.")
        center = u.dimensions[:3] / 2

    print("Initial center at {}={:.3f} nm and {}={:.3f} nm.".format(
        'XYZ'[odims[0]], center[odims[0]]/10, 'XYZ'[odims[1]], center[odims[1]]/10))

    if args.radius != None:
        args.radius /= 10
    else:
        args.radius = u.dimensions[odims].min() / 2
        print("No radius given --> Take smallest box extension (r={:.2f} nm).".format(args.radius/10))

    if args.length != None:
        args.length /= 10
    else:
        print("No length given --> Take length in {}.".format('XYZ'[dim]))
        args.length = u.dimensions[dim]

    ngroups = len(args.groups)
    nbins = int(np.ceil(args.radius / args.binwidth / 1))
    dr = np.ones(nbins) * args.radius / nbins
    r_bins = np.arange(nbins) * dr + dr
    args.delta_r_sq = r_bins**2 - \
        np.insert(r_bins, 0, 0)[0:-1]**2  # r_o^2 - r_i^2
    args.r = np.copy(r_bins) - dr / 2

    density_mean = np.zeros((nbins, ngroups))
    density_mean_sq = np.zeros((nbins, ngroups))

    print("\nCalcualate profile for the following group(s):")
    sel = []
    for i, gr in enumerate(args.groups):
        sel.append(u.select_atoms(gr))
        print("{:>15}: {:>10} atoms".format(gr, sel[i].n_atoms))
        if sel[i].n_atoms == 0:
            sys.exit(
                "\n Error: {} does not contain any atoms. Please adjust '-gr' selection.".format(gr))
    print("\n")

    print('Using', nbins, 'bins.')

    # ======== MAIN LOOP =========
    # ============================
    args.frame = 0
    print("\rEvaluating frame: {:>12} time: {:>12} ps".format(
        args.frame, round(u.trajectory.time)), end="")
    for ts in u.trajectory[args.beginframe:args.endframe:args.skipframes]:

        # calculater center of cylinder.
        if args.center != None:
            center = u.select_atoms(args.center).center_of_mass()
        else:
            center = u.dimensions[:3] / 2

        for index, selection in enumerate(sel):

            # select cylinder of the given length and radius
            cut = selection.atoms[np.where(np.absolute(
                selection.atoms.positions[:, dim] - center[dim]) < args.length / 2)[0]]
            cylinder = cut.atoms[np.where(np.linalg.norm(
                (cut.atoms.positions[:, odims] - center[odims]), axis=1) < args.radius)[0]]

            radial_positions = np.linalg.norm(
                (cylinder.atoms.positions[:, odims] - center[odims]), axis=1)
            bins = np.digitize(radial_positions, r_bins)
            density_ts = np.histogram(bins, bins=np.arange(
                nbins + 1), weights=weight(cylinder))[0]

            density_mean[:, index] += density_ts
            density_mean_sq[:, index] += density_ts**2

        args.frame += 1
        print_frameinfo(ts, args.frame)
        # call for output
        if (int(ts.time) % args.outfreq == 0 and ts.time - args.begin >= args.outfreq):
            output(density_mean, density_mean_sq)

    output(density_mean, density_mean_sq)
    print("\n")


    if DEBUG:
        # Inject local variables into global namespace for debugging.
        for key, value in locals().items():
            globals()[key] = value

if __name__ == "__main__":
    main(firstarg=1)
