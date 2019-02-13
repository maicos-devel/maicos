#!/usr/bin/env python3
# coding: utf-8

import os
import sys

import MDAnalysis as mda
import numpy as np
from scipy.optimize import curve_fit

from . import initilize_universe, pbctools, print_frameinfo
from .. import initilize_parser
from ..utils import savetxt

# ========== PARSER ===========
# =============================
parser = initilize_parser(add_traj_arguments=True)
parser.description = """Mean Velocity analysis.
Reads in coordinates and velocities from a trajectory and calculates a
velocity profile along a given axis. The obtained profile is averaged over the 4
symmetric slab halfs.Error bars are estimated via block averaging as described in [1].

[1] Hess, B. Determining the shear viscosity of model liquids from molecular
dynamics simulations. The Journal of Chemical Physics 116, 209-217 (2002)."""
parser.add_argument(
    '-o',
    dest='output',
    type=str,
    default='com',
    help='Suffix for output filenames')
parser.add_argument(
    '-dout',
    dest='outfreq',
    type=float,
    default='1000',
    help='Default time after which output files are refreshed (1000 ps).')
parser.add_argument(
    '-d',
    dest='dim',
    type=int,
    default=2,
    help='dimension for position binning (0=X, 1=Y, 2=Z)',
)
parser.add_argument(
    '-dv',
    dest='vdim',
    type=int,
    default=0,
    help='dimension for velocity binning (0=X, 1=Y, 2=Z)',
)
parser.add_argument(
    '-nbins', dest='nbins', type=int, default=200, help='number of bins')
parser.add_argument(
    '-gr',
    dest='groups',
    type=str,
    default='all',
    help=
    'Atoms for which to compute the velocity profile. This must contain only one moleculetype.',
)
parser.add_argument(
    '-nblock',
    dest='nblock',
    type=int,
    default=10,
    help=
    'maximum number of blocks for block averaging error estimate; 1 results in standard error'
)

# ======== DEFINITIONS ========
# =============================


def blockee(data):
    ee = []
    for i in range(0, int(np.log2(args.nblock)) - 1):
        bs = 2**i
        numb = args.nblock // bs
        blocks = np.vstack([
            np.mean(data[:, bs * i:bs * (i + 1)], axis=1) for i in range(numb)
        ]).T
        ee.append([
            bs * args.dt * args.skipframes * args.blockfreq,
            np.std(blocks, axis=1) / np.sqrt(numb - 1)
        ])
    return ee


def fitfn(x, alpha, tau1, tau2, pref):
    return pref * (alpha * tau1 * (1 + tau1 / x * (np.exp(-x / tau1) - 1)) +
                   (1 - alpha) * tau2 * (1 + tau2 / x *
                                         (np.exp(-x / tau2) - 1)))


def output(L, av_vel, av_vel_sq, binframes, isFinal=False):
    """Averages the profile and saves the current profiles to a file."""
    minframes = .01 * args.frame  # minimum number of frames where a water should be present
    avL = L / args.frame / 10  # in nm
    dz = (avL) / (args.nbins)
    symz = np.arange(0, avL / 4 - dz / 2, dz) + dz / 2
    z = np.arange(0, avL - dz / 2, dz) + dz / 2
    v = np.sum(av_vel[np.sum(binframes, axis=1) > minframes], axis=1) / \
        np.sum(binframes[np.sum(binframes, axis=1) > minframes], axis=1)
    dv = np.sqrt(
        av_vel_sq[np.sum(binframes, axis=1) > minframes] /
        np.sum(binframes[np.sum(binframes, axis=1) > minframes], axis=1) - v**2
    ) / np.sqrt(
        np.sum(binframes[np.sum(binframes, axis=1) > minframes], axis=1) - 1)

    # make use of the symmetry
    symvel = (av_vel[:args.nbins // 4] -
              av_vel[args.nbins // 4:2 * args.nbins // 4][::-1] -
              av_vel[2 * args.nbins // 4:3 * args.nbins // 4] +
              av_vel[3 * args.nbins // 4:][::-1])
    symvel_sq = (av_vel_sq[:args.nbins // 4] +
                 av_vel_sq[args.nbins // 4:2 * args.nbins // 4][::-1] +
                 av_vel_sq[2 * args.nbins // 4:3 * args.nbins // 4] +
                 av_vel_sq[3 * args.nbins // 4:][::-1])
    symbinframes = (binframes[:args.nbins // 4] +
                    binframes[args.nbins // 4:2 * args.nbins // 4][::-1] +
                    binframes[2 * args.nbins // 4:3 * args.nbins // 4] +
                    binframes[3 * args.nbins // 4:][::-1])
    vsym = np.sum(symvel[np.sum(symbinframes, axis=1) > minframes], axis=1) / \
        np.sum(symbinframes[np.sum(symbinframes, axis=1) > minframes], axis=1)
    dvsym = np.sqrt(
        symvel_sq[np.sum(symbinframes, axis=1) > minframes] /
        np.sum(symbinframes[np.sum(symbinframes, axis=1) > minframes], axis=1) -
        vsym**2
    ) / np.sqrt(
        np.sum(symbinframes[np.sum(symbinframes, axis=1) > minframes], axis=1) -
        1)

    header = "statistics over {:.1f} picoseconds".format(args.frame * args.dt)
    savetxt(
        'vel_sym_' + args.output + '.dat',
        np.vstack((symz[np.sum(symbinframes, axis=1) > minframes], vsym,
                   dvsym)).T,
        header=header)
    savetxt(
        'vel_' + args.output + '.dat',
        np.vstack((z[np.sum(binframes, axis=1) > minframes], v, dv)).T,
        header=header)
    if isFinal and int(np.log2(args.nblock)) > 1:
        bee = blockee(np.nan_to_num(av_vel / binframes))
        ee_out = np.vstack(np.hstack((bee[i])) for i in range(len(bee)))

        prefs = 2 * (av_vel_sq[np.sum(binframes, axis=1) > minframes] / np.sum(
            binframes[np.sum(binframes, axis=1) > minframes], axis=1) - v**
                     2) / (args.frame * args.dt * args.skipframes
                          )  # 2 sigma^2 / T, (A16) in [1]
        ees = []
        count = 0
        params = []
        for i in range(args.nbins):
            if np.sum(binframes[i]) > minframes:
                pref = prefs[count]

                def modfitfn(x, alpha, tau1, tau2):
                    return fitfn(x, alpha, tau1, tau2, pref)

                [alpha, tau1, tau2], pcov = curve_fit(
                    modfitfn,
                    ee_out[:, 0], (ee_out[:, i + 1])**2,
                    bounds=([0, 0, 0], [1, np.inf, np.inf]),
                    p0=[.99, .001, .01],
                    max_nfev=1e5)
                # (A.17) in [1]
                errest = np.sqrt(pref * (alpha * tau1 + (1 - alpha) * tau2))
                ees.append(errest)
                params.append([pref, alpha, tau1, tau2])
                count += 1

        savetxt(
            'vel_' + args.output + '.dat',
            np.vstack((z[np.sum(binframes, axis=1) > minframes], v,
                       np.array(ees), dv)).T)
        savetxt(
            'errest_' + args.output + '.dat',
            np.concatenate(
                (ee_out[:, 0].reshape(len(ee_out), 1),
                 (ee_out[:, 1:])[:, np.sum(binframes, axis=1) > minframes]),
                axis=1),
            header='z ' + ' '.join(
                map(str, z[np.sum(binframes, axis=1) > minframes])))
        savetxt('errparams_' + args.output + '.dat', np.array(params))

        # Same for symmetrized
        bee = blockee(np.nan_to_num(symvel / symbinframes))
        ee_out = np.vstack(np.hstack((bee[i])) for i in range(len(bee)))

        prefs = 2 * (
            symvel_sq[np.sum(symbinframes, axis=1) > minframes] / np.sum(
                symbinframes[np.sum(symbinframes, axis=1) > minframes], axis=1)
            - vsym**2) / (args.frame * args.dt * args.skipframes
                         )  # 2 sigma^2 / T, (A16) in [1]
        ees = []
        count = 0
        for i in range(args.nbins // 4):
            if np.sum(symbinframes[i]) > minframes:
                pref = prefs[count]

                def modfitfn(x, alpha, tau1, tau2):
                    return fitfn(x, alpha, tau1, tau2, pref)

                [alpha, tau1, tau2], pcov = curve_fit(
                    modfitfn,
                    ee_out[:, 0], (ee_out[:, i + 1])**2,
                    bounds=([0, 0, 0], [1, np.inf, np.inf]),
                    p0=[.9, 1e3, 1e4],
                    max_nfev=1e5)
                # (A.17) in [1]
                errest = np.sqrt(pref * (alpha * tau1 + (1 - alpha) * tau2))
                ees.append(errest)
                count += 1

        savetxt(
            'vel_sym_' + args.output + '.dat',
            np.vstack((symz[np.sum(symbinframes, axis=1) > minframes], vsym,
                       np.array(ees))).T)
        savetxt(
            'errest_sym_' + args.output + '.dat',
            np.concatenate(
                (ee_out[:, 0].reshape(len(ee_out), 1),
                 (ee_out[:, 1:])[:, np.sum(symbinframes, axis=1) > minframes]),
                axis=1),
            header='z ' + ' '.join(
                map(str, symz[np.sum(symbinframes, axis=1) > minframes])))
        savetxt('errparams_sym_' + args.output + '.dat', np.array(params))


# ========== MAIN ============
# ============================


def main(firstarg=2, DEBUG=False):
    global args
    args = parser.parse_args(args=sys.argv[firstarg:])
    u = initilize_universe(args)

    if (args.nbins % 2 != 0):
        sys.exit(
            "Number of bins %d can't be divided by 4 (for making use of symmetry)! Quitting..."
        )

    evalframes = (args.endframe - args.beginframe) // args.skipframes
    args.blockfreq = evalframes // args.nblock
    skipinitialframes = evalframes % args.nblock  # skip from initial, not end

    print("\nCalcualate profile for the following group:")
    sol = u.select_atoms(args.groups)
    print("{:>15}: {:>10} atoms".format(args.groups, sol.n_atoms))
    if sol.n_atoms == 0:
        sys.exit(
            "\n Error: {} does not contain any atoms. Please adjust '-gr' selection."
            .format(args.groups))

    av_vel = np.zeros((args.nbins, args.nblock))
    av_vel_sq = np.zeros((args.nbins))
    # count frame only to velocity if existing
    binframes = np.zeros((args.nbins, args.nblock))
    L = 0

    print('Using', args.nbins, 'bins.')

    # ======== MAIN LOOP =========
    # ============================
    for args.frame, ts in enumerate(
            u.trajectory[args.beginframe:args.endframe:args.skipframes]):
        print_frameinfo(ts, args.frame)
        pbctools.repairMolecules(sol)

        L += u.dimensions[args.dim]

        # Do the solvent velocity profile
        # -------------------------------
        seg = sol.segments
        # Calculate the centers of the objects ( i.e. Molecules )
        atomsPerMolecule = seg.n_atoms // seg.n_residues
        masses = np.sum(seg.atoms.masses[i::atomsPerMolecule]
                        for i in range(atomsPerMolecule))
        masspos = seg.atoms.positions[:, args.dim] * seg.atoms.masses
        coms = np.sum(masspos[i::atomsPerMolecule]
                      for i in range(atomsPerMolecule)) / masses
        massvels = seg.atoms.velocities[:, args.vdim] * seg.atoms.masses
        comvels = np.sum(massvels[i::atomsPerMolecule]
                         for i in range(atomsPerMolecule)) / masses

        bins = (coms /
                (u.dimensions[args.dim] / args.nbins)).astype(int) % args.nbins
        bincount = np.bincount(bins, minlength=args.nbins)
        with np.errstate(divide='ignore', invalid='ignore'):
            curvel = np.nan_to_num(
                np.histogram(
                    bins, bins=np.arange(0, args.nbins + 1), weights=comvels)[0]
                / bincount)  # mean velocity in this bin, zero if empty

        # add velocities to the average and convert to (m/s)
        av_vel[:, args.frame // args.blockfreq] += curvel * 100.
        av_vel_sq[:] += (curvel * 100.)**2
        # only average velocities if bin is not empty
        binframes[:, args.frame // args.blockfreq] += bincount > 0

        # call for output
        if (int(ts.time) % args.outfreq == 0 and
                ts.time - args.begin >= args.outfreq):
            output(L, av_vel, av_vel_sq, binframes)

    output(L, av_vel, av_vel_sq, binframes, isFinal=True)
    print("\n")

    if DEBUG:
        # Inject local variables into global namespace for debugging.
        for key, value in locals().items():
            globals()[key] = value


if __name__ == "__main__":
    main(firstarg=1)
