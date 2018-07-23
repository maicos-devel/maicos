#!/usr/bin/env python
# coding: utf-8

# Mandatory imports
from __future__ import absolute_import, division, print_function

import argparse
import os
import sys

import MDAnalysis as mda
import numpy as np
import scipy.optimize
import scipy.constants
import time

from . import initilize_universe, print_frameinfo
from .. import initilize_parser

from ..utils import repairMolecules, FT, ScalarProdCorr

# ========== PARSER ===========
# =============================

# parser object will already contain options for
# the topology, trajectory, begin, end, skipped frames and the box dimenions

parser = initilize_parser(add_traj_arguments=True)
parser.description = """This script, given molecular dynamics trajectory data, should produce a\
    .txt file containing the complex dielectric function as a function of the (linear, not radial)\
    frequency, along with the associated standard deviations."""
parser.add_argument('-temp',   dest='temperature',      type=float,
                    default=300, help='Reference temperature.')
parser.add_argument("-o", "--output",
                    default="", help="Prefix for the output file.")
parser.add_argument("-truncfac", type=float, default=30.0,
                    help="Truncation factor.\
    By default, the autocorrelation of the polarization is fit with A*exp( -t/Tau ),\
    and the truncation length is then taken as truncfac*Tau. The default is 30.\
    Alternatively, the truncation length in number of steps may be specified with -trunclen.")
parser.add_argument("-trunclen", type=float,
                    help="Truncation length in picoseconds.\
    Specifying a value overrides the fitting procedure otherwise used to find the truncation length.")
parser.add_argument("-Nsegments", type=int, default=100,
                    help="The number of segments the polarization trajectory is broken into in order\
    to find the standard deviation.\
    Specifying a value overrides the fitting procedure otherwise used to find the truncation length.")
parser.add_argument("-np", "--noplots",
                    help="Prevents plots from being generated.", action="store_true")

# ======== DEFINITIONS ========
# =============================

def TimeDerivative5PS(v, dt): # Numerical 5-point stencil time derivative
    # note: v must be evenly-spaced 1-d array

    dvdt = np.empty(len(v))

    dvdt[2:-2] = (8 * (v[2:] - v[:-2])[1:-1] - v[4:] + v[:-4]) / float(12 * dt)

    # 2 pt stencil for the 2nd-to-last datapoints; prevents any truncatoin at the ends
    dvdt[1] = (v[2] - v[0]) / float(2 * dt)
    dvdt[-2] = (v[-1] - v[-3]) / float(2 * dt)

    # slope of last segment for final datapoints
    dvdt[0] = (v[1] - v[0]) / float(dt)
    dvdt[-1] = (v[-1] - v[-2]) / float(dt)

    return dvdt

# Single exponential for fitting:

def single_exp(x, A, D):
    return np.absolute(A) * np.exp(-x / D)

# ========== MAIN ============
# ============================

def main(firstarg=2):

    global args

    # parse the arguments and saves them in an args object
    args = parser.parse_args(args=sys.argv[firstarg:])

    if not args.noplots:
        import matplotlib.pyplot as plt

    # == POLARIZATION/AUTOCORR ===
    # ============================

    # the MDAnalysis universe given by the user for analysis
    u = initilize_universe(args)

    dt = args.dt * args.skipframes

    Nframes = (args.endframe - args.beginframe) // args.skipframes

    if len(args.output) > 0:
        args.output += "_"

    args.frame = 0
    t = (np.arange(args.beginframe, args.endframe) - args.beginframe) * dt
    
    t_0 = time.clock()

    if not os.path.isfile(args.output+'P_tseries.npy'): # check if polarization is present

        P = np.zeros((Nframes , 3))
        print('Polarization file not found: calculating polarization trajectory and average volume')
        V = np.zeros(1)
        print("\rEvaluating frame: {:>12}        time: {:>12} ps".format(
            args.frame, round(u.trajectory.time)), end="")
        for ts in u.trajectory[args.beginframe:args.endframe:args.skipframes]:

            # Calculations done in every frame
            V[0] += ts.volume
            repairMolecules(u.atoms)
            P[args.frame,:] = np.dot(u.atoms.charges, u.atoms.positions)
            args.frame += 1
            print_frameinfo(ts, args.frame)

        P /= 10 # MDA gives units of Angstroms, we use nm
        V[0] *= 1e-3 / float(args.frame) # normalization and unit conversion
        np.save(args.output+'P_tseries.npy', P)
        np.savetxt(args.output+'V.txt', V)

    elif not os.path.isfile(args.output+'V.txt'):

        print('Polarization file found: loading polarization and calculating average volume')
        P = np.load(args.output+'P_tseries.npy')
        V = np.zeros(1)
        print("\rEvaluating frame: {:>12}       time: {:>12} ps".format(
            args.frame, round(u.trajectory.time)), end="")

        for ts in u.trajectory[args.beginframe:args.endframe:args.skipframes]:

            # Calculations done in every frame
            V[0] += ts.volume
            args.frame += 1
            print_frameinfo(ts, args.frame)

        V *= 1e-3 / float(args.frame) # normalization and unit conversion                
        np.savetxt(args.output+'V.txt', V)

    else:

        print('Polarization and volume files found: loading both...', end="")
        P = np.load(args.output+'P_tseries.npy')
        V = np.loadtxt(args.output+'V.txt')

    t_1 = time.clock()

    print("\nTook {:.2f} seconds".format(t_1 - t_0))

    P_P = ScalarProdCorr(P)  # Autocorrelation fn of P for all timesteps

    # ======== TRUNCATION ========
    # ============================

    # Colors for plotting
    col1 = 'royalblue'
    col2 = 'red'
    col3 = 'grey'

    # Define the truncation length:

    if args.trunclen == None:

        p_opt, p_cov = scipy.optimize.curve_fit(
            single_exp, t, P_P, p0=(1, 1))  # fit whole dataset

        # if necessary, cut off data and fit again
        if 2 * args.truncfac * p_opt[1] < len(P_P):
            # cutoff index for 2. fit
            im = np.absolute(t - 2 * args.truncfac * p_opt[1]).argmin()
            p_opt, p_cov = scipy.optimize.curve_fit(
                single_exp, t[:im], P_P[:im], p0=p_opt)

        # if necessary, cut off data and fit again
        if args.truncfac * p_opt[1] < len(P_P):
            # cutoff index for 2. fit
            im = np.absolute(t - args.truncfac * p_opt[1]).argmin()
            p_opt, p_cov = scipy.optimize.curve_fit(
                single_exp, t[:im], P_P[:im], p0=p_opt)

        # step where data is cut off
        args.trunclen = np.absolute(t - args.truncfac * p_opt[1]).argmin()

        # Plot the trunclen fit:

        if not args.noplots:

            plotlen = 2 * args.trunclen

            if plotlen > len(P_P):
                plotlen = int(1.1 * args.trunclen)

            if plotlen > len(P_P):
                plotlen = args.trunclen

            plt.figure(figsize=(8, 5.657))

            plt.title('Exponential Fit to Determine Truncation Length')
            plt.ylabel('$<P(0)$ $P(t)>$')
            plt.xlabel('t [ps]')

            plt.xlim(-0.02 * t[plotlen], t[plotlen])

            plt.axvline(x=t[args.trunclen], linewidth=1, color=col3, linestyle='--',
                        label='truncation length = {0:.4} s'.format(t[args.trunclen]))
            plt.plot(t[:plotlen], P_P[:plotlen], color=col1, marker='.',
                     markersize=4, linestyle='', label='$<P(0)$ $P(t)>$')
            plt.plot(t[:plotlen], single_exp(t[:plotlen], p_opt[0], p_opt[1]),
                     linewidth=1, color=col2, label='fit: ~exp( -t/{0:.4} )'.format(p_opt[1]))

            plt.legend(loc='best')

            plt.savefig(args.output+'P_autocorr_trunc_fit.pdf', format='pdf')
            plt.close()

        print('Truncation length set via exponential fit to {0} steps, i.e. {1:.6} ps'.format(
            args.trunclen, args.trunclen * dt))

    else:
        # convert from picoseconds into the frame index
        args.trunclen = int(args.trunclen / dt)
        print('Truncation length set manually to {0} steps, i.e. {1:.3} ps'.format(
            args.trunclen, args.trunclen * dt))


    # Truncate and pad with zeros:

    t = np.resize(t, 2 * args.trunclen)  # resize
    P_P = np.append(np.resize(P_P, args.trunclen), np.zeros(args.trunclen))  # resize, pad w zeros

    # ====== SUSCEPTIBILITY ======
    # ============================

    # Calculate susceptibility from entire autocorrelation:

    print('Calculating susceptibilty and errors...')

    t0 = time.clock()

    nu, susc = FT(t, TimeDerivative5PS(P_P, dt))


    # Find the variance/std deviation of the susceptibility:

    seglen = int(Nframes / float(args.Nsegments))  # length of segments

    if args.trunclen > seglen:
        args.Nsegments = int(Nframes / float(args.trunclen))
        seglen = int(Nframes / float(args.Nsegments))

    print('Number of segments to be used in error:\t{0}'.format(args.Nsegments))

    # std deviation of susceptibility
    dsusc = np.zeros(2 * args.trunclen, dtype=complex)

    for seg in range(0, args.Nsegments):
        P_P = ScalarProdCorr(P[seg * seglen:(seg + 1) * seglen, :])
        P_P = np.append(np.resize(P_P, args.trunclen), np.zeros(args.trunclen))
        ss = FT(t, TimeDerivative5PS(P_P, dt), False)
        dsusc += (ss - susc).real * (ss - susc).real + \
            1j * (ss - susc).imag * (ss - susc).imag

    dsusc = np.sqrt(dsusc) / args.Nsegments  # convert from variance to std. deviation

    t1 = time.clock()

    print(
        'Susceptibility and associated errors calculated - took {0:.3} s'.format(t1 - t0))


    # Discard negative-frequency data; contains the same information as positive regime:

    nu = nu[args.trunclen:]
    susc = susc[args.trunclen:]
    dsusc = dsusc[args.trunclen:]

    pref = scipy.constants.e * scipy.constants.e * 1e9 / \
        (3 * V * scipy.constants.k * args.temperature * scipy.constants.epsilon_0)

    susc *= -pref
    susc.imag *= -1  # we want -1 * Im susc (sometimes denoted as Chi'')

    dsusc *= pref  # std. deviation of susc is linear in susc
    nu /= 2 * np.pi  # now nu represents f instead of omega

    # Save susceptibility in a user-friendly text file:

    suscfilename = args.output+'susc.txt'

    np.savetxt(suscfilename,
               np.transpose([nu, susc.real, dsusc.real, susc.imag, dsusc.imag]),
               delimiter='\t',
               header='freq\tsusc\'\tstd_dev_susc\'\tsusc\'\'\tstd_dev_susc\'\'')

    print('Susceptibility data saved as ' + suscfilename)

    # ==== OPTIONAL PLOTTING =====
    # ============================

    if args.noplots:
        print('User specified not to generate plots -- finished :)')
    else:

        print('Calculations complete. Generating plots...')

        # Extraction of values useful for plotting:

        nuPeak = nu[np.argmax(susc.imag)]  # frequency at peak
        nuL = nu[1]  # lower x benchmark
        nuBuf = 1.4  # buffer factor for extra room in the x direction

        # max value of data
        suscMax = np.ceil(np.max([np.max(susc.real), np.max(susc.imag)]))
        suscL = np.min([susc.real[-1], susc.imag[1]]) / 2  # lower y benchmark
        suscBuf = 1.2  # buffer factor for extra room in the x direction


        # Plot lin-log:

        plt.figure(figsize=(8, 5.657))

        plt.title('Complex Dielectric Function (lin-log)')
        plt.ylabel('$\chi$')
        plt.xlabel('$\\nu$ [THz]')

        plt.grid()

        plt.xlim(nuL / nuBuf, nu[-1] * nuBuf)

        plt.xscale('log')

        plt.fill_between(nu, susc.real - dsusc.real, susc.real +
                         dsusc.real, color=col2, alpha=0.1)
        plt.fill_between(nu, susc.imag - dsusc.imag, susc.imag +
                         dsusc.imag, color=col1, alpha=0.1)

        plt.plot(nu, susc.real, col2, linewidth=1, label='$\chi^{{\prime}}$')
        plt.plot(nu, susc.imag, col1, linewidth=1, label='$\chi^{{\prime \prime}}$')

        plt.legend(loc='best')

        plt.savefig(args.output + 'susc_linlog.pdf', format='pdf')

        plt.close()

        # Plot log-log

        plt.figure(figsize=(8, 5.657))

        plt.title('Complex Dielectric Function (log-log)')
        plt.ylabel('$\chi$')
        plt.xlabel('$\\nu$ [THz]')

        plt.grid()

        plt.ylim(suscL / suscBuf, suscMax * suscBuf)
        plt.xlim(nuL / nuBuf, nu[-1] * nuBuf)

        plt.yscale('log')
        plt.xscale('log')

        plt.fill_between(nu, susc.real - dsusc.real, susc.real +
                         dsusc.real, color=col2, alpha=0.1)
        plt.fill_between(nu, susc.imag - dsusc.imag, susc.imag +
                         dsusc.imag, color=col1, alpha=0.1)

        plt.plot(nu, susc.real, color=col2, linewidth=1,
                 label='$\chi^{{\prime}}$ : max = {0:.2f}'.format(np.max(susc.real)))
        plt.plot(nu, susc.imag, color=col1, linewidth=1,
                 label='$\chi^{{\prime \prime}}$ : max = {0:.2f}'.format(np.max(susc.imag)))

        plt.legend(loc='best')

        plt.savefig(args.output + 'susc_log.pdf', format='pdf')

        print('Plots generated -- finished :)')

    print('\n\n')

if __name__ == "__main__":
    main(firstarg=1)
