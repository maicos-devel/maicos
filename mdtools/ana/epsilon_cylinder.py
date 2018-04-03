#!/usr/bin/env python3
# coding: utf8

from __future__ import division, print_function, absolute_import

import argparse
import os
import sys

import MDAnalysis as mda
import numpy as np

import pbctools
from . import initilize_universe, print_frameinfo
from .. import initilize_parser

parser = initilize_parser(add_traj_arguments=True)
parser.description="""Calculation of the dielectric
profile for axial and radial direction at the system's center of geometry."""
parser.add_argument('-g', dest='geometry', type=str,
                    default=None, help="A gro file w/o water")
parser.add_argument('-r', dest='radius', type=float,
                    default=None, help='radius of the cylinder in Angstrom')
parser.add_argument('-dr', dest='binwidth', type=float,
                    default=0.05, help='specify the binwidth [Angtsrom]')
parser.add_argument('-vr', dest='variable_dr', action='store_true',
                    help="Use a variable binwidth, where the volume kept fixed.")
parser.add_argument('-l', dest='length', type=float,
                    default=None, help='length of the cylinder in Angstrom')
parser.add_argument('-o', dest='output', type=str,
                    default='eps_cyl', help='Prefix for output filenames')
parser.add_argument('-dout', dest='outfreq', type=float,
                    default='1000', help='Default time after which output files are refreshed (1000 ps)')
parser.add_argument('-si', dest='single', action='store_true',
                    help='Single water line?')

#======== DEFINITIONS ========
#=============================
epsilon_0 = 5.526350e-3  # ElementaryCharge (Angstroms Volts)^-1
kb = 8.6173324e-5  # electronVolts Kelvins^-1
T = 300  # Kelvins


def output():

    if args.single == True:  # removed average of M if single line water.
        cov_ax = mM_ax.sum(axis=1) / frame
        cov_rad = mM_rad.sum(axis=1) / frame

        dcov_ax = (mM_ax.std(axis=1) / frame * resample) / \
            np.sqrt(resample - 1)
        dcov_rad = (mM_rad.std(axis=1) / frame * resample) / \
            np.sqrt(resample - 1)
    else:
        cov_ax = mM_ax.sum(axis=1) / frame - \
            m_ax.sum(axis=1) / frame * M_ax.sum() / frame
        cov_rad = mM_rad.sum(axis=1) / frame - \
            m_rad.sum(axis=1) / frame * M_rad.sum() / frame

        dcov_ax = np.sqrt(
            (mM_ax.std(axis=1) / frame * resample)**2
            + (m_ax.std(axis=1) / frame * resample * M_ax.sum() / frame)**2
            + (m_ax.sum(axis=1) / frame * M_ax.std() / frame * resample)**2) / np.sqrt(resample - 1)
        dcov_rad = np.sqrt(
            (mM_rad.std(axis=1) / frame * resample)**2
            + (m_rad.std(axis=1) / frame * resample * M_rad.sum() / frame)**2
            + (m_rad.sum(axis=1) / frame * M_rad.std() / frame * resample)**2) / np.sqrt(resample - 1)

    eps_ax = 1 + cov_ax / (epsilon_0 * kb * T)
    eps_rad_inv = 1 - 2 * np.pi * r * length * cov_rad / (epsilon_0 * kb * T)

    deps_ax = dcov_ax / (epsilon_0 * kb * T)
    deps_rad_inv = 2 * np.pi * r * length * dcov_rad / (epsilon_0 * kb * T)

    outdata_ax = np.array(
        [r, eps_ax, deps_ax])
    outdata_rad = np.array(
        [r, eps_rad_inv, deps_rad_inv])

    header = 'Command line was: %s\n' % ' '.join(sys.argv) +\
             'statistics over: %d ps\n' % (ts.time - begin) +\
             'r [Angtsroms]\t eps \t eps_err'
    np.savetxt(args.output + '_ax.dat', outdata_ax.T, header=header)
    np.savetxt(args.output + '_rad.dat', outdata_rad.T, header=header)


#=========== MAIN ===========
#============================

def main(firstarg=2):
    global args

    args = parser.parse_args(args=sys.argv[firstarg:])

    u = mda.Universe(args.topology, args.trajectory)
    if args.geometry != None:
        system = MDAnalysis.Universe(args.geometry)
        cog = system.atoms.center_of_geometry()
    else:
        print("No geometry set. Calculate center of geometry from box dimensions.")
        cog = u.dimensions[:3] / 2

    if args.radius != None:
        radius = args.radius
    else:
        print("No radius set. Take smallest box extension.")
        radius = u.dimensions[:2].min() / 2

    if args.length != None:
        length = args.length
    else:
        length = u.dimensions[2]

    dt = u.trajectory.dt

    begin = int(args.begin // dt)
    if args.end != None:
        end = int(args.end // dt)
    else:
        end = int(u.trajectory.totaltime // u.trajectory.dt)

    if begin > end:
        sys.exit("Start time is larger than end time!")


    nbins = int(np.ceil(radius / args.binwidth))

    if args.variable_dr:
        # variable dr
        sol = np.ones(nbins) * radius**2 / nbins
        mat = np.diag(np.ones(nbins)) + np.diag(np.ones(nbins - 1) * -1, k=-1)

        r_bins = np.sqrt(np.linalg.solve(mat, sol))
        dr = r_bins - np.insert(r_bins, 0, 0)[0:-1]
    else:
        # Constant dr
        dr = np.ones(nbins) * radius / nbins
        r_bins = np.arange(nbins) * dr + dr

    delta_r_sq = r_bins**2 - np.insert(r_bins, 0, 0)[0:-1]**2  # r_o^2-r_i^2
    r = np.copy(r_bins) - dr / 2

    # Use resampling for error estimation.
    # We do block averaging for 10 hardcoded blocks.

    resample = 10
    resample_freq = int(np.ceil((end - begin) / resample))

    m_rad = np.zeros((nbins, resample))

    M_rad = np.zeros((resample))
    mM_rad = np.zeros((nbins, resample))  # total fluctuations

    m_ax = np.zeros((nbins, resample))
    M_ax = np.zeros((resample))
    mM_ax = np.zeros((nbins, resample))  # total fluctuations

    print('Computing dielectric profiles for water in the cylinder.')
    print('Using', nbins, 'bins.')
    frame = 0
    for ts in u.trajectory[begin:end:args.skipframes]:

        # make broken molecules whole again!
        pbctools.repairMolecules(u.atoms)

        # Transform from cartesian coordinates [x,y,z] to cylindrical coordinates [r,z] (skip phi because of symmetry)
        positions_cyl = np.empty([u.atoms.positions.shape[0], 2])
        positions_cyl[:, 0] = np.linalg.norm(
            (u.atoms.positions[:, 0:2] - cog[0:2]), axis=1)
        positions_cyl[:, 1] = u.atoms.positions[:, 2]

        # Use polarization density ( for radial component )
        # ========================================================
        bins_rad = np.digitize(positions_cyl[:, 0], r_bins)

        curQ_rad = np.histogram(bins_rad, bins=np.arange(
            nbins + 1), weights=u.atoms.charges)[0]
        this_m_rad = -np.cumsum((curQ_rad / delta_r_sq) *
                                r * dr) / (r * np.pi * length)

        this_M_rad = np.sum(this_m_rad * dr)
        M_rad[frame // resample_freq] += this_M_rad

        m_rad[:, frame // resample_freq] += this_m_rad
        mM_rad[:, frame // resample_freq] += this_m_rad * this_M_rad

        # Use virtual cutting method ( for axial component )
        # ========================================================
        nbinsz = 250  # number of virtual cuts ("many")

        this_M_ax = np.dot(u.atoms.charges, positions_cyl[:, 1])
        M_ax[frame // resample_freq] += this_M_ax

        # Move all r-positions to 'center of charge' such that we avoid monopoles in r-direction.
        # We only want to cut in z direction.
        chargepos = positions_cyl * np.abs(u.atoms.charges[:, np.newaxis])
        centers = np.sum(chargepos[i::3] for i in range(
            3)) / np.abs(u.residues[0].atoms.charges).sum()
        testpos = np.empty(positions_cyl[:, 0].shape)
        testpos = np.repeat(centers[:, 0], 3)

        binsr = np.digitize(testpos, r_bins)

        dz = np.ones(nbinsz) * length / nbinsz
        z = np.arange(nbinsz) * dz + dz

        binsz = np.digitize(positions_cyl[:, 1], z)
        binsz[np.where(binsz < 0)] = 0
        curQz = np.histogram2d(binsr, binsz, bins=[np.arange(nbins + 1), np.arange(nbinsz + 1)],
                               weights=u.atoms.charges)[0]
        curqz = np.cumsum(curQz, axis=1) / (np.pi * delta_r_sq)[:, np.newaxis]

        this_m_ax = -curqz.mean(axis=1)

        m_ax[:, frame // resample_freq] += this_m_ax
        mM_ax[:, frame // resample_freq] += this_m_ax * this_M_ax


        frame += 1
        print_frameinfo(ts,frame)
        # call for output
        if (int(ts.time) % args.outfreq == 0 and ts.time - args.begin >= args.outfreq):
            output()

    print("\n")
    output()

if __name__ == "__main__":
    main(firstarg=1)
