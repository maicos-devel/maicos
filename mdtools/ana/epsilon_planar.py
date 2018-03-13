#!/usr/bin/env python3
# coding: utf8

from __future__ import division, print_function

# parse command line options
import argparse
import sys

import MDAnalysis
import numpy as np

import pbctools
from . import add_traj_arguments, print_frameinfo

parser = argparse.ArgumentParser(description="Calculate the dielectric profile.\
        See Bonthuis et. al., Langmuir 28, vol. 20 (2012) for details.",
     prog = "mdtools epsilon_planar", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

add_traj_arguments(parser)

parser.add_argument('-dz', dest='binwidth', type=float,
                    default=0.05, help='specify the binwidth [nm]')
parser.add_argument('-dt', dest='skipframes', type=int,
                    default=1, help='skip every N frames')
parser.add_argument('-d', dest='dim', type=int,
                    default=2, help='direction normal to the surface (x,y,z=0,1,2, default: z)')
parser.add_argument('-zmin', dest='zmin', type=float,
                    default=0, help='minimal z-coordinate for evaluation')
parser.add_argument('-zmax', dest='zmax', type=float,
                    default=-1, help='maximal z-coordinate for evaluation')
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
parser.add_argument('-box', dest='box', type=float, nargs="+",
                    default=None,
                    help='Sets the box dimensions x y z [alpha beta gamma] (in Angstrom!).\
                   If None dimensions from the trajectory will be used.')
parser.add_argument('-nopbcrepair', dest='bpbc', action='store_false',
                    help='do not make broken molecules whole again (only works if molecule is smaller than shortest box vector')


def output():
    avV = V / frame

    cov_perp = mM_perp.sum(axis=2) / frame - \
        m_perp.sum(axis=2) / frame * M_perp.sum() / frame
    dcov_perp = np.sqrt((mM_perp.std(axis=2) / frame * resample)**2
                        + (m_perp.std(axis=2) / frame *
                           resample * M_perp.sum() / frame)**2
                        + (m_perp.sum(axis=2) / frame * M_perp.std() / frame * resample)**2) / np.sqrt(resample - 1)
    cov_perp_self = mm_perp / frame - \
        (m_perp.sum(axis=2) / frame * m_perp.sum(axis=2) /
         frame * A * Lz / nbins / frame)
    cov_perp_coll = cmM_perp / frame - \
        m_perp.sum(axis=2) / frame * cM_perp / frame

    var_perp = M_perp_2.sum() / frame - (M_perp.sum() / frame)**2
    dvar_perp = (M_perp_2 / frame - (M_perp / frame)
                 ** 2).std() / np.sqrt(resample - 1)

    cov_par = mM_par.sum(axis=2) / frame - \
        m_par.sum(axis=2) / frame * M_par.sum() / frame
    cov_par_self = mm_par / frame - \
        m_par.sum(axis=2) / frame * (m_par.sum(axis=2)
                                     * Lz / nbins / frame * A) / frame
    cov_par_coll = cmM_par / frame - m_par.sum(axis=2) / frame * cM_par / frame
    dcov_par = np.sqrt((mM_par.std(axis=2) / frame * resample)**2
                       + (m_par.std(axis=2) / frame *
                          resample * M_par.sum() / frame)**2
                       + (m_par.sum(axis=2) / frame * M_par.std() / frame * resample)**2) / np.sqrt(resample - 1)

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

    if (zmax == -1):
        z = np.linspace(0, Lz / frame, len(eps_par)) / 10
    else:
        z = np.linspace(zmin, zmax, len(eps_par)) / 10.

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
               header="statistics over %d picoseconds" % ts.time)
    np.savetxt(args.output + '_par.dat', outdata_par,
               header="statistics over %d picoseconds" % ts.time)

    return


def main(firstarg=2):
    global args
    
    args = parser.parse_args(args=sys.argv[firstarg:])

    u = MDAnalysis.Universe(args.topology, args.trajectory)
    mysels = [u.select_atoms(i) for i in args.groups]
    sol = u.select_atoms('resname SOL')

    if args.box != None:
        assert (len(args.box) == 6 or len(args.box) == 3),\
            'The boxdimensions must contain 3 entries for the box vectors and possibly 3 more for the angles.'
        u.dimensions = np.array(args.box)

    dim = args.dim
    # Assume a threedimensional universe...
    xydims = np.roll(np.arange(3), -dim)[1:]
    dz = args.binwidth * 10  # Convert to Angstroms

    if (args.zmax == -1):
        zmax = u.dimensions[dim]
    else:
        zmax = args.zmax * 10.
    zmin = args.zmin * 10.
    nbins = int((zmax - zmin) / dz)  # CAVE: binwidth varies in NPT !

    begin = args.begin

    if (args.end != -1):
        end = args.end
    else:
        end = u.trajectory.totaltime

    ''' Use resampling for error estimation
        Now even better: MDAnalysis stores offsets,
        thus we can easily do block averaging
        for now hardcode 10 blocks...
    '''
    resample = 10
    resample_freq = u.trajectory.n_frames // resample

    m_par = np.zeros((nbins, len(args.groups), resample))
    mM_par = np.zeros((nbins, len(args.groups), resample))  # total fluctuations
    mm_par = np.zeros((nbins, len(args.groups)))  # self
    cmM_par = np.zeros((nbins, len(args.groups)))  # collective contribution
    cM_par = np.zeros((nbins, len(args.groups)))
    M_par = np.zeros((resample))

    # Same for perpendicular
    m_perp = np.zeros((nbins, len(args.groups), resample))
    mM_perp = np.zeros((nbins, len(args.groups), resample))  # total fluctuations
    mm_perp = np.zeros((nbins, len(args.groups)))  # self
    cmM_perp = np.zeros((nbins, len(args.groups)))  # collective contribution
    cM_perp = np.zeros((nbins, len(args.groups)))  # collective contribution

    V = 0
    Lz = 0

    M_perp = np.zeros((resample))
    M_perp_2 = np.zeros((resample))

    print('Using', nbins, 'bins.')

    A = np.prod(u.dimensions[xydims])

    frame = 0
    print("\rEvaluating frame: ", u.trajectory.frame, "\ttime: ", int(u.trajectory.time), end="")

    startframe = int(begin // u.trajectory.dt)
    endframe = int(end // u.trajectory.dt)

    for ts in u.trajectory[startframe:endframe:args.skipframes]:

        if (args.zmax == -1):
            zmax = ts.dimensions[dim]

        if args.membrane_shift:
            # shift membrane
            ts.positions[:, dim] += ts.dimensions[dim] / 2
            ts.positions[:, dim] %= ts.dimensions[dim]
        if args.com:
            # put water COM into center
            waterCOM = np.sum(
                sol.atoms.positions[:, 2] * sol.atoms.masses) / sol.atoms.masses.sum()
            print("shifting by ", waterCOM)
            ts.positions[:, dim] += ts.dimensions[dim] / 2 - waterCOM
            ts.positions[:, dim] %= ts.dimensions[dim]

        if args.bpbc:
            # make broken molecules whole again!
            pbctools.repairMolecules(u.atoms)

        dz_frame = ts.dimensions[dim] / nbins

        # precalculate total polarization of the box
        this_M_perp, this_M_par = np.split(
            np.roll(np.dot(u.atoms.charges, u.atoms.positions), -dim), [1])

        # Use polarization density ( for perpendicular component )
        # ========================================================

        # sum up the averages
        M_perp[frame // resample_freq] += this_M_perp
        M_perp_2[frame // resample_freq] += this_M_perp**2
        for i, sel in enumerate(mysels):
            bins = ((sel.atoms.positions[:, dim] - zmin) /
                    ((zmax - zmin) / (nbins))).astype(int)
            curQ = np.histogram(bins, bins=np.arange(
                nbins + 1), weights=sel.atoms.charges)[0]
            this_m_perp = -np.cumsum(curQ / A)
            m_perp[:, i, frame // resample_freq] += this_m_perp
            mM_perp[:, i, frame // resample_freq] += this_m_perp * this_M_perp
            mm_perp[:, i] += this_m_perp * this_m_perp * \
                (ts.dimensions[dim] / nbins) * A  # self term
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
            testpos[:, dim] = np.repeat(centers[:, dim], atomsPerMolecule)

            binsz = (((testpos[:, dim] - zmin) % ts.dimensions[dim])
                     / ((zmax - zmin) / nbins)).astype(int)

            # Average parallel directions
            for j, direction in enumerate(xydims):
                dx = ts.dimensions[direction] / nbinsx
                binsx = (sel.atoms.positions[:, direction]
                         / (ts.dimensions[direction] / nbinsx)).astype(int)
                binsx[np.where(binsx < 0)] = 0
                curQx = np.histogram2d(binsz, binsx, bins=[np.arange(0, nbins + 1), np.arange(0, nbinsx + 1)],
                                       weights=sel.atoms.charges)[0]
                curqx = np.cumsum(curQx, axis=1) / (ts.dimensions[xydims[1 - j]] * (
                    ts.dimensions[dim] / nbins))  # integral over x, so units of area
                this_m_par = -curqx.mean(axis=1)

                m_par[:, i, frame // resample_freq] += this_m_par
                mM_par[:, i, frame // resample_freq] += this_m_par * this_M_par[j]
                M_par[frame // resample_freq] += this_M_par[j]
                mm_par[:, i] += this_m_par * this_m_par * dz_frame * A
                # collective contribution
                cmM_par[:, i] += this_m_par * \
                    (this_M_par[j] - this_m_par * dz_frame * A)
                cM_par[:, i] += this_M_par[j] - this_m_par * dz_frame * A

        V += ts.volume
        Lz += ts.dimensions[dim]

        frame += 1
        print_frameinfo(ts,frame)
        # call for output
        if (frame % args.outfreq == 0 and frame >= args.outfreq):
            output()


    print('\n')
    output()

if __name__ == "__main__":
    main(firstarg=1)
