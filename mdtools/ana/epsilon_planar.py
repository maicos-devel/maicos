#!/usr/bin/env python
# coding: utf8

from __future__ import division, print_function

# parse command line options
import argparse
import sys

import MDAnalysis
import numpy as np

import pbctools
from . import initilize_universe, print_frameinfo
from .. import initilize_parser

parser = initilize_parser(add_traj_arguments=True)
parser.description = """Calculate the dielectric profile.
        See Bonthuis et. al., Langmuir 28, vol. 20 (2012) for details."""
parser.add_argument('-dz', dest='binwidth', type=float,
                    default=0.05, help='specify the binwidth [nm]')
parser.add_argument('-d', dest='dim', type=int,
                    default=2, help='direction normal to the surface (x,y,z=0,1,2, default: z)')
parser.add_argument('-zmin', dest='zmin', type=float,
                    default=0, help='minimal z-coordinate for evaluation [nm]')
parser.add_argument('-zmax', dest='zmax', type=float,
                    default=-1, help='maximal z-coordinate for evaluation [nm]')
parser.add_argument('-temp', dest='temperature', type=float,
                    default=300, help='temperature [K]')
parser.add_argument('-o', dest='output', type=str,
                    default='eps', help='Prefix for output filenames')
parser.add_argument('-groups', dest='groups', type=str, nargs='+',
                    help='atom group selection',
                    default=['resname SOL', 'not resname SOL'])
parser.add_argument('-dout', dest='outfreq', type=float,
                    default='10000', help='Default number of frames after which output files are refreshed (10000)')
parser.add_argument('-2d', dest='b2d', action='store_const',
                    const=True, default=False,
                    help='use 2d slab geometry')
parser.add_argument('-vac', dest='vac', action='store_const',
                    const=True, default=False,
                    help='use vacuum boundary conditions instead of metallic (2D only!).')
parser.add_argument('-sym', dest='bsym', action='store_const',
                    const=True, default=False,
                    help='symmetrize the profiles')
parser.add_argument('-shift', dest='membrane_shift', action='store_const',
                    const=True, default=False,
                    help='shift system by half a box length (useful for membrane simulations)')
parser.add_argument('-com', dest='com', action='store_const',
                    const=True, default=False,
                    help='shift system such that the water COM is centered')
parser.add_argument('-nopbcrepair', dest='bpbc', action='store_false',
                    help='do not make broken molecules whole again (only works if molecule is smaller than shortest box vector')


def output(V, Lz, A, m_par, mM_par, mm_par, cmM_par, cM_par, M_par, m_perp, mM_perp, mm_perp, cmM_perp, cM_perp, M_perp, M_perp_2):
    avV = V / args.frame

    cov_perp = mM_perp.sum(axis=2) / args.frame - \
        m_perp.sum(axis=2) / args.frame * M_perp.sum() / args.frame
    dcov_perp = np.sqrt((mM_perp.std(axis=2) / args.frame * args.resample)**2
                        + (m_perp.std(axis=2) / args.frame *
                           args.resample * M_perp.sum() / args.frame)**2
                        + (m_perp.sum(axis=2) / args.frame * M_perp.std() / args.frame * args.resample)**2) / np.sqrt(args.resample - 1)
    cov_perp_self = mm_perp / args.frame - \
        (m_perp.sum(axis=2) / args.frame * m_perp.sum(axis=2) /
         args.frame * A * Lz / args.nbins / args.frame)
    cov_perp_coll = cmM_perp / args.frame - \
        m_perp.sum(axis=2) / args.frame * cM_perp / args.frame

    var_perp = M_perp_2.sum() / args.frame - (M_perp.sum() / args.frame)**2
    dvar_perp = (M_perp_2 / args.frame - (M_perp / args.frame)
                 ** 2).std() / np.sqrt(args.resample - 1)

    cov_par = mM_par.sum(axis=2) / args.frame - \
        m_par.sum(axis=2) / args.frame * M_par.sum() / args.frame
    cov_par_self = mm_par / args.frame - \
        m_par.sum(axis=2) / args.frame * (m_par.sum(axis=2)
                                     * Lz / args.nbins / args.frame * A) / args.frame
    cov_par_coll = cmM_par / args.frame - m_par.sum(axis=2) / args.frame * cM_par / args.frame
    dcov_par = np.sqrt((mM_par.std(axis=2) / args.frame * args.resample)**2
                       + (m_par.std(axis=2) / args.frame *
                          args.resample * M_par.sum() / args.frame)**2
                       + (m_par.sum(axis=2) / args.frame * M_par.std() / args.frame * args.resample)**2) / np.sqrt(args.resample - 1)

    eps0inv = 1. / 8.854e-12
    pref = (1.6e-19)**2 / 1e-10
    kB = 1.3806488e-23
    beta = 1. / (kB * args.temperature)

    eps_par = beta * eps0inv * pref / 2 * cov_par
    deps_par = beta * eps0inv * pref / 2 * dcov_par
    eps_par_self = beta * eps0inv * pref / 2 * cov_par_self
    eps_par_coll = beta * eps0inv * pref / 2 * cov_par_coll

    if (args.b2d):
        eps_perp = - beta * eps0inv * pref * cov_perp
        eps_perp_self = - beta * eps0inv * pref * cov_perp_self
        eps_perp_coll = - beta * eps0inv * pref * cov_perp_coll
        deps_perp = np.abs(- eps0inv * beta * pref) * dcov_perp
        if (args.vac):
            eps_perp *= 2. / 3.
            eps_perp_self *= 2. / 3.
            eps_perp_coll *= 2. / 3.
            deps_perp *= 2. / 3.

    else:
        eps_perp = (- eps0inv * beta * pref * cov_perp) \
            / (1 + eps0inv * beta * pref / avV * var_perp)
        deps_perp = np.abs((- eps0inv * beta * pref)
                           / (1 + eps0inv * beta * pref / avV * var_perp)) * dcov_perp \
            + np.abs((- eps0inv * beta * pref * cov_perp)
                     / (1 + eps0inv * beta * pref / avV * var_perp)**2) * dvar_perp

        eps_perp_self = (- eps0inv * beta * pref * cov_perp_self) \
            / (1 + eps0inv * beta * pref / avV * var_perp)
        eps_perp_coll = (- eps0inv * beta * pref * cov_perp_coll) \
            / (1 + eps0inv * beta * pref / avV * var_perp)

    if (args.zmax == -1):
        z = np.linspace(0, Lz / args.frame, len(eps_par)) / 10
    else:
        z = np.linspace(args.zmin, args.zmax, len(eps_par)) / 10.

    outdata_perp = np.hstack([z[:, np.newaxis], eps_perp.sum(axis=1)[:, np.newaxis], eps_perp,
                              np.linalg.norm(deps_perp, axis=1)[
        :, np.newaxis], deps_perp,
        eps_perp_self.sum(axis=1)[:, np.newaxis], eps_perp_coll.sum(
            axis=1)[:, np.newaxis],
        eps_perp_self, eps_perp_coll])

    outdata_par = np.hstack([z[:, np.newaxis], eps_par.sum(axis=1)[:, np.newaxis], eps_par,
                             np.linalg.norm(deps_par, axis=1)[
        :, np.newaxis], deps_par,
        eps_par_self.sum(axis=1)[:, np.newaxis], eps_par_coll.sum(
            axis=1)[:, np.newaxis],
        eps_par_self, eps_par_coll])

    if (args.bsym):
        for i in range(len(outdata_par) - 1):
            outdata_par[i + 1] = .5 * \
                (outdata_par[i + 1] + outdata_par[i + 1][-1::-1])
        for i in range(len(outdata_perp) - 1):
            outdata_perp[i + 1] = .5 * \
                (outdata_perp[i + 1] + outdata_perp[i + 1][-1::-1])

    np.savetxt(args.output + '_perp.dat', outdata_perp,
               header="statistics over %d picoseconds" % (
                   (args.endframe - args.beginframe + 1) * args.dt))
    np.savetxt(args.output + '_par.dat', outdata_par,
               header="statistics over %d picoseconds" % (
                   (args.endframe - args.beginframe + 1) * args.dt))

    return


def main(firstarg=2):
    global args

    args = parser.parse_args(args=sys.argv[firstarg:])
    u = initilize_universe(args)

    print("\nCalcualate profile for the following group(s):")
    mysels = []
    for i, gr in enumerate(args.groups):
        mysels.append(u.select_atoms(gr))
        print("{:>15}: {:>10} atoms".format(gr, mysels[i].n_atoms))
        if mysels[i].n_atoms == 0:
            sys.exit(
                "\n Error: {} does not contain any atoms. Please adjust '-gr' selection.".format(gr))

    print("\n")

    sol = u.select_atoms('resname SOL')

    # Assume a threedimensional universe...
    xydims = np.roll(np.arange(3), -args.dim)[1:]
    dz = args.binwidth * 10  # Convert to Angstroms

    if (args.zmax == -1):
        zmax = u.dimensions[args.dim]
    else:
        args.zmax *= 10.
        zmax = args.zmax

    args.zmin = args.zmin * 10.
    args.nbins = int((zmax - args.zmin) / dz)  # CAVE: binwidth varies in NPT !

    if (args.end != -1):
        end = args.end
    else:
        end = u.trajectory.totaltime

    ''' Use resampling for error estimation
        Now even better: MDAnalysis stores offsets,
        thus we can easily do block averaging
        for now hardcode 10 blocks...
    '''
    args.resample = 10
    resample_freq = u.trajectory.n_frames // args.resample


    V = 0
    Lz = 0
    A = np.prod(u.dimensions[xydims])

    m_par = np.zeros((args.nbins, len(args.groups), args.resample))
    mM_par = np.zeros((args.nbins, len(args.groups), args.resample)
                      )  # total fluctuations
    mm_par = np.zeros((args.nbins, len(args.groups)))  # self
    cmM_par = np.zeros((args.nbins, len(args.groups)))  # collective contribution
    cM_par = np.zeros((args.nbins, len(args.groups)))
    M_par = np.zeros((args.resample))

    # Same for perpendicular
    m_perp = np.zeros((args.nbins, len(args.groups), args.resample))
    mM_perp = np.zeros((args.nbins, len(args.groups), args.resample)
                       )  # total fluctuations
    mm_perp = np.zeros((args.nbins, len(args.groups)))  # self
    cmM_perp = np.zeros((args.nbins, len(args.groups)))  # collective contribution
    cM_perp = np.zeros((args.nbins, len(args.groups)))  # collective contribution
    M_perp = np.zeros((args.resample))
    M_perp_2 = np.zeros((args.resample))


    print('Using', args.nbins, 'bins.')

    args.frame = 0
    print("\rEvaluating frame: ", u.trajectory.frame, "\ttime: ", int(u.trajectory.time), end="")
    for ts in u.trajectory[args.beginframe:args.endframe + 1:args.skipframes]:

        if (args.zmax == -1):
            zmax = ts.dimensions[args.dim]

        if args.membrane_shift:
            # shift membrane
            ts.positions[:, args.dim] += ts.dimensions[args.dim] / 2
            ts.positions[:, args.dim] %= ts.dimensions[args.dim]
        if args.com:
            # put water COM into center
            waterCOM = np.sum(
                sol.atoms.positions[:, 2] * sol.atoms.masses) / sol.atoms.masses.sum()
            print("shifting by ", waterCOM)
            ts.positions[:, args.dim] += ts.dimensions[args.dim] / 2 - waterCOM
            ts.positions[:, args.dim] %= ts.dimensions[args.dim]

        if args.bpbc:
            # make broken molecules whole again!
            pbctools.repairMolecules(u.atoms)

        dz_frame = ts.dimensions[args.dim] / args.nbins

        # precalculate total polarization of the box
        this_M_perp, this_M_par = np.split(
            np.roll(np.dot(u.atoms.charges, u.atoms.positions), -args.dim), [1])

        # Use polarization density ( for perpendicular component )
        # ========================================================

        # sum up the averages
        M_perp[args.frame // resample_freq] += this_M_perp
        M_perp_2[args.frame // resample_freq] += this_M_perp**2
        for i, sel in enumerate(mysels):
            bins = ((sel.atoms.positions[:, args.dim] - args.zmin) /
                    ((zmax - args.zmin) / (args.nbins))).astype(int)
            curQ = np.histogram(bins, bins=np.arange(
                args.nbins + 1), weights=sel.atoms.charges)[0]
            this_m_perp = -np.cumsum(curQ / A)
            m_perp[:, i, args.frame // resample_freq] += this_m_perp
            mM_perp[:, i, args.frame // resample_freq] += this_m_perp * this_M_perp
            mm_perp[:, i] += this_m_perp * this_m_perp * \
                (ts.dimensions[args.dim] / args.nbins) * A  # self term
            # collective contribution
            cmM_perp[:, i] += this_m_perp * \
                (this_M_perp - this_m_perp * (A * dz_frame))
            cM_perp[:, i] += this_M_perp - this_m_perp * A * dz_frame

        # Use virtual cutting method ( for parallel component )
        # ========================================================
        nbinsx = 250  # number of virtual cuts ("many")

        for i, sel in enumerate(mysels):
            # Move all z-positions to 'center of charge' such that we avoid monopoles in z-direction
            # (compare Eq. 33 in Bonthuis 2012; we only want to cut in x/y direction)
            chargepos = sel.atoms.positions * \
                np.abs(sel.atoms.charges[:, np.newaxis])
            atomsPerMolecule = sel.n_atoms // sel.n_residues
            centers = np.sum(chargepos[i::atomsPerMolecule] for i in range(atomsPerMolecule)) \
                / np.abs(sel.residues[0].atoms.charges).sum()
            testpos = sel.atoms.positions
            testpos[:, args.dim] = np.repeat(
                centers[:, args.dim], atomsPerMolecule)

            binsz = (((testpos[:, args.dim] - args.zmin) % ts.dimensions[args.dim])
                     / ((zmax - args.zmin) / args.nbins)).astype(int)

            # Average parallel directions
            for j, direction in enumerate(xydims):
                dx = ts.dimensions[direction] / nbinsx
                binsx = (sel.atoms.positions[:, direction]
                         / (ts.dimensions[direction] / nbinsx)).astype(int)
                binsx[np.where(binsx < 0)] = 0
                curQx = np.histogram2d(binsz, binsx, bins=[np.arange(0, args.nbins + 1), np.arange(0, nbinsx + 1)],
                                       weights=sel.atoms.charges)[0]
                curqx = np.cumsum(curQx, axis=1) / (ts.dimensions[xydims[1 - j]] * (
                    ts.dimensions[args.dim] / args.nbins))  # integral over x, so units of area
                this_m_par = -curqx.mean(axis=1)

                m_par[:, i, args.frame // resample_freq] += this_m_par
                mM_par[:, i, args.frame // resample_freq] += this_m_par * \
                    this_M_par[j]
                M_par[args.frame // resample_freq] += this_M_par[j]
                mm_par[:, i] += this_m_par * this_m_par * dz_frame * A
                # collective contribution
                cmM_par[:, i] += this_m_par * \
                    (this_M_par[j] - this_m_par * dz_frame * A)
                cM_par[:, i] += this_M_par[j] - this_m_par * dz_frame * A

        V += ts.volume
        Lz += ts.dimensions[args.dim]

        args.frame += 1
        print_frameinfo(ts, args.frame)
        # call for output
        if (args.frame % args.outfreq == 0 and args.frame >= args.outfreq):
            output(V, Lz, A, m_par, mM_par, mm_par, cmM_par, cM_par, M_par, m_perp, mM_perp, mm_perp, cmM_perp, cM_perp, M_perp, M_perp_2)

    print('\n')
    output(V, Lz, A, m_par, mM_par, mm_par, cmM_par, cM_par, M_par, m_perp, mM_perp, mm_perp, cmM_perp, cM_perp, M_perp, M_perp_2)


if __name__ == "__main__":
    main(firstarg=1)
