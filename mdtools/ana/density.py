#!/usr/bin/env python
# coding: utf-8

from __future__ import division, print_function, absolute_import

import argparse
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
    Computes partial densities or tempertaure profiles across the box.
    For group selections use strings in the MDAnalysis selection command style
    found here:
    https://pythonhosted.org/MDAnalysis/documentation_pages/selections.html"""
parser.add_argument('-o',   dest='output',      type=str,
                    default='density',              help='Prefix for output filenames')
parser.add_argument('-dout', dest='outfreq',     type=float, default='1000',
                    help='Default time after which output files are refreshed (1000 ps).')
parser.add_argument('-d',   dest='dim',         type=int,   default=2,
                    help='dimension for binning (0=X, 1=Y, 2=Z)', )
parser.add_argument('-dz',  dest='binwidth',    type=float,
                    default=1,                      help='binwidth (nanometer)')
parser.add_argument('-muo', dest='muout',       type=str,   default='dens',
                    help='Prefix for output filename for chemical potential')
parser.add_argument('-temp', dest='temperature', type=float, default=300,
                    help='temperature (K) for chemical potential')
parser.add_argument('-zpos', dest='zpos',        type=float, default=None,
                    help='position at which the chemical potential will be computed. By default average over box.')
parser.add_argument('-dens', dest='density',     type=str,   default='mass',
                    help='Density: mass, number, charge, temp')
parser.add_argument('-gr',  dest='groups',      type=str,   default=[
                    'all'], nargs='+',      help='Atoms for which to compute the density profile', )

# ======== DEFINITIONS ========
# =============================


def output(density_mean, density_mean_sq, av_box_length):
    """Averages the profile and saves the current profiles to a file."""

    dens_mean = density_mean / args.frame
    dens_mean_sq = density_mean_sq / args.frame

    dens_std = np.nan_to_num(np.sqrt(dens_mean_sq - dens_mean**2))
    dens_err = dens_std / np.sqrt(args.frame)

    dz = av_box_length / (args.frame * args.nbins)
    z = np.linspace(0, av_box_length / args.frame,
                    args.nbins, endpoint=False) + dz / 2

    # write header
    if args.density == "mass":
        units = "kg m^(-3)"
    elif args.density == "number":
        units = "nm^(-3)"
    elif args.density == "charge":
        units = "e nm^(-3)"
    elif args.density == "temp":
        units = "K"

    if args.density == 'temp':
        columns = "temperature profile [{}]".format(units)
    else:
        columns = "{} density profile [{}]".format(args.density, units)
    columns += "\nstatistics over {:.1f} picoseconds \npositions [nm]".format(
        args.frame * args.dt)
    for group in args.groups:
        columns += "\t" + group
    for group in args.groups:
        columns += "\t" + group + " error"

    # save density profile
    np.savetxt(args.output + '.dat',
               np.hstack(((z[:, np.newaxis]), dens_mean, dens_err)),
               header=columns)

    # save chemcial potential
    if (args.zpos != None):
        this = (args.zpos / (av_box_length / args.frame)
                * args.nbins).astype(int)
        np.savetxt(args.muout + '.dat',
                   np.hstack((mu(dens_mean[this]), dmu(dens_mean[this], dens_err[this])))[None])
    else:
        np.savetxt(args.muout + '.dat',
                   np.array((np.mean(mu(dens_mean, args.temperature)),
                             np.mean(dmu(dens_mean, dens_err, args.temperature))))[None])


def weight(selection):
    """Calculates the weights for the histogram depending on the choosen type of density."""
    if args.density == "mass":
        # amu in kg -> kg/m^3
        return selection.atoms.masses * 1.66053892
    elif args.density == "number":
        return np.ones(selection.atoms.n_atoms)
    elif args.density == "charge":
        return selection.atoms.charges
    elif args.density == "temp":
        # ((1 amu * (Angstrom^2)) / (picoseconds^2)) / Boltzmann constant = 1.20272362 Kelvin
        return ((selection.atoms.velocities**2).sum(axis=1) * selection.atoms.masses / 2 * 1.20272362)


def mu(rho, temperature):
    """Returns the chemical potential calculated from the density: mu = k_B T log(rho.)"""
    # db = 1.00394e-1  # De Broglie (converted to nm)
    kT = 0.00831446215 * temperature
    if np.all(rho > 0):
        return kT * np.log(rho)
    elif np.any(rho == 0):
        return np.float64("-inf")
    else:
        return np.float("nan")


def dmu(rho, drho, temperature):
    """Returns the error of the chemical potential calculated from the density using propagation of uncertainty."""
    kT = 0.00831446215 * temperature
    if np.all(rho > 0):
        return (kT / rho * drho)
    elif np.any(rho == 0):
        return np.float64("-inf")
    else:
        return np.float("nan")


# ========== MAIN ============
# ============================

def main(firstarg=2):
    global args

    args = parser.parse_args(args=sys.argv[firstarg:])
    u = initilize_universe(args)

    if args.density not in ["mass", "number", "charge", "temp"]:
        parser.error(
            'Unknown density type {}. Valid are mass, number, charge, temp'.format(args.density))

    if args.density == 'temp':
        print('Computing temperature profile along {}-axes.'.format('XYZ'[args.dim]))
    else:
        print('Computing {} density profile along {}-axes.'.format(args.density, 'XYZ'[args.dim]))

    ngroups = len(args.groups)
    args.nbins = int(np.ceil(u.dimensions[args.dim] / 10 / args.binwidth))

    density_mean = np.zeros((args.nbins, ngroups))
    density_mean_sq = np.zeros((args.nbins, ngroups))
    av_box_length = 0

    print("\nCalcualate profile for the following group(s):")
    sel = []
    for i, gr in enumerate(args.groups):
        sel.append(u.select_atoms(gr))
        print("{:>15}: {:>10} atoms".format(gr, sel[i].n_atoms))
        if sel[i].n_atoms == 0:
            sys.exit(
                "\n Error: {} does not contain any atoms. Please adjust '-gr' selection.".format(gr))
    print("\n")

    print('Using', args.nbins, 'bins.')

    # ======== MAIN LOOP =========
    # ============================
    args.frame = 0
    print("\rEvaluating frame: {:>12} time: {:>12} ps".format(
        args.frame, round(u.trajectory.time)), end="")
    for ts in u.trajectory[args.beginframe:args.endframe:args.skipframes]:
        curV = ts.volume / 1000
        av_box_length += u.dimensions[args.dim] / 10

        for index, selection in enumerate(sel):
            bins = (selection.atoms.positions[:, args.dim] /
                    (u.dimensions[args.dim] / args.nbins)).astype(int) % args.nbins
            density_ts = np.histogram(bins, bins=np.arange(
                args.nbins + 1), weights=weight(selection))[0]

            bincount = np.bincount(bins, minlength=args.nbins)

            if args.density == 'temp':
                density_mean[:, index] += density_ts / bincount
                density_mean_sq[:, index] += (density_ts / bincount)**2
            else:
                density_mean[:, index] += density_ts / curV * args.nbins
                density_mean_sq[:,
                                index] += (density_ts / curV * args.nbins)**2

        args.frame += 1
        print_frameinfo(ts, args.frame)
        # call for output
        if (int(ts.time) % args.outfreq == 0 and ts.time - args.begin >= args.outfreq):
            output(density_mean, density_mean_sq, av_box_length)

    output(density_mean, density_mean_sq, av_box_length)
    print("\n")


if __name__ == "__main__":
    main(firstarg=1)
