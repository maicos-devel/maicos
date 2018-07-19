#!/usr/bin/env python
# coding: utf-8

# Mandatory imports
from __future__ import absolute_import, division, print_function

import argparse
import os
import sys
import timeit

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
parser.description = """Description for my awesome analysis script."""
parser.add_argument('-temp',   dest='temperature',      type=float,
                    default=300, help='Reference temperature')
parser.add_argument("-o", "--output",
                    default="", help="Prefix for the output file.")
parser.add_argument("-truncfac", type=float,
                    help="Truncation factor.\
    By default, the autocorrelation of the polarization is fit with A*exp( -t/Tau ),\
    and the truncation length is then taken as truncfac*Tau. The default is 30.\
    Alternatively, the truncation length in number of steps may be specified with -trunclen.")
parser.add_argument("-trunclen", type=float,
                    help="Truncation length in picoseconds.\
    Specifying a value overrides the fitting procedure otherwise used to find the truncation length.")
parser.add_argument("-Nsegments", type=int,
                    help="The number of segments the polarization trajectory is broken into in order to find the\
    standard deviation.\
    Specifying a value overrides the fitting procedure otherwise used to find the truncation length.")
parser.add_argument("-np", "--noplots",
                    help="Prevents plots from being generated.", action="store_true")


# Numerical time derivative function using a 5-point stencil:

def TimeDerivative5PS(v, dt):  # Five-point stencil time derivative for evenly spaced array
    # note: v must be 1-d array

    dvdt = np.empty(len(v))

    dvdt[2:-2] = (8 * (v[2:] - v[:-2])[1:-1] - v[4:] + v[:-4]) / float(12 * dt)

    # 2 pt stencil for the 2nd-to-last datapoints
    dvdt[1] = (v[2] - v[0]) / float(2 * dt)
    dvdt[-2] = (v[-1] - v[-3]) / float(2 * dt)

    # slope of last segment for final datapoints
    dvdt[0] = (v[1] - v[0]) / float(dt)
    dvdt[-1] = (v[-1] - v[-2]) / float(dt)

    return dvdt

def single_exp(x, A, D):
    return np.absolute(A) * np.exp(-x / D)

# ========== MAIN ============
# ============================


def main(firstarg=2):
    # Not essential but nice to use args also in custo functions without passing
    # explicitly
    global args

    # parse the arguments and saves them in an args object
    args = parser.parse_args(args=sys.argv[firstarg:])

    # the MDAnalysis universe given by the user for analysis
    u = initilize_universe(args)

    T = args.temperature  # temperature in Kelvin
    truncfac = args.truncfac  # truncation factor to determine truncation length
    trunclen = args.trunclen  # truncation length
    Nseg = args.Nsegments  # number of segments for calculating variance
    dt = args.dt

    if len(args.output) > 0:
        args.output += "_"

    if truncfac == None:
        truncfac = 30.0
    if Nseg == None:
        Nseg = 100

    P = np.zeros( ( ( args.endframe-args.beginframe) // args.skipframes , 3) )
    print(P.shape)
    V = 0
    t = dt * np.arange(args.beginframe,args.endframe,args.skipframes)
    t -= dt * args.beginframe
    print(t.shape)
    # ======== MAIN LOOP =========
    # ============================
    t_0 = time.clock()
    args.frame = 0
    print("\rEvaluating frame: {:>12} time: {:>12} ps".format(
        args.frame, round(u.trajectory.time)), end="")
    for ts in u.trajectory[args.beginframe:args.endframe:args.skipframes]:

        # Calculations done in every frame
        V += ts.volume
        repairMolecules(u.atoms)
        P[args.frame,:] = np.dot(u.atoms.charges, u.atoms.positions)

        args.frame += 1
        print_frameinfo(ts, args.frame)

    t_end = time.clock()
    print("\n")

    # Final calculations i.e. printing informations and call for output
    print("Calculation took {:.2f} seconds.".format(t_end - t_0))

    V /= args.frame
    P_P = ScalarProdCorr(P)  # Autocorrelation fn of P for all timesteps
    print(P_P.shape)
    # Define the truncation length:

    col1 = 'royalblue'
    col2 = 'red'
    col3 = 'grey'

    if trunclen == None:

        p_opt, p_cov = scipy.optimize.curve_fit(
            single_exp, t, P_P, p0=(1, 1))  # fit whole dataset

        # if necessary, cut off data and fit again
        if 2 * truncfac * p_opt[1] < len(P_P):
            # cutoff index for 2. fit
            im = np.absolute(t - 2 * truncfac * p_opt[1]).argmin()
            p_opt, p_cov = scipy.optimize.curve_fit(
                single_exp, t[:im], P_P[:im], p0=p_opt)

        # if necessary, cut off data and fit again
        if truncfac * p_opt[1] < len(P_P):
            # cutoff index for 2. fit
            im = np.absolute(t - truncfac * p_opt[1]).argmin()
            p_opt, p_cov = scipy.optimize.curve_fit(
                single_exp, t[:im], P_P[:im], p0=p_opt)

        # step where data is cut off
        trunclen = np.absolute(t - truncfac * p_opt[1]).argmin()

        # Plot the trunclen fit:

        if not args.noplots:

            import matplotlib.pyplot as plt

            plotlen = 2 * trunclen

            if plotlen > len(P_P):
                plotlen = int(1.1 * trunclen)

            if plotlen > len(P_P):
                plotlen = trunclen

            plt.figure(figsize=(8, 5))

            plt.title('Exponential Fit to Determine Truncation Length')
            plt.ylabel('$<P(0)$ $P(t)>$')
            plt.xlabel('t [ps]')

            plt.xlim(-0.02 * t[plotlen], t[plotlen])

            plt.axvline(x=t[trunclen], linewidth=1, color=col3, linestyle='--',
                        label='truncation length = {0:.4} s'.format(t[trunclen]))
            plt.plot(t[:plotlen], P_P[:plotlen], color=col1, marker='.',
                     markersize=4, linestyle='', label='$<P(0)$ $P(t)>$')
            plt.plot(t[:plotlen], single_exp(t[:plotlen], p_opt[0], p_opt[1]),
                     linewidth=1, color=col2, label='fit: ~exp( -t/{0:.4} )'.format(p_opt[1]))

            plt.legend(loc='best')

            plt.savefig(args.output+'P_autocorr_trunc_fit.pdf', format='pdf')
            plt.close()

        print('Truncation length set via exponential fit to {0} steps, i.e. {1:.6} ps\n'.format(
            trunclen, trunclen * dt))

    else:
        # convert from picoseconds into the frame index
        trunclen = int(trunclen / dt)
        print('Truncation length set manually to {0} steps, i.e. {1:.3} ps\n'.format(
            trunclen, trunclen * dt))


    # Truncate and pad with zeros:

    t = np.resize(t, 2 * trunclen)  # resize
    P_P = np.append(np.resize(P_P, trunclen), np.zeros(trunclen))  # resize, pad w zeros


    # Calculate susceptibility from entire autocorrelation:

    print('Calculating susceptibilty and errors...')

    t0 = timeit.default_timer()

    nu, susc = FT(t, TimeDerivative5PS(P_P, u.trajectory.dt))


    # Find the variance/std deviation of the susceptibility:

    seglen = int(len(u.trajectory) / Nseg)  # length of segments

    if trunclen > seglen:
        Nseg = int(len(u.trajectory) / float(trunclen))
        seglen = int(len(u.trajectory) / float(Nseg))

    print('Number of segments to be used in error:\t{0}'.format(Nseg))

    # std deviation of susceptibility
    dsusc = np.zeros(2 * trunclen, dtype=complex)

    for seg in range(0, Nseg):
        P_P = ScalarProdCorr(P[seg * seglen:(seg + 1) * seglen, :])
        P_P = np.append(np.resize(P_P, trunclen), np.zeros(trunclen))
        ss = FT(t, TimeDerivative5PS(P_P, u.trajectory.dt), False)
        dsusc += (ss - susc).real * (ss - susc).real + \
            1j * (ss - susc).imag * (ss - susc).imag

    dsusc = np.sqrt(dsusc) / Nseg  # convert from variance to std. deviation

    t1 = timeit.default_timer()

    print(
        'Susceptibility and associated errors calculated - took {0:.3} s\n'.format(t1 - t0))


    # Discard negative-frequency data; contains the same information as positive regime:

    nu = nu[trunclen:]
    susc = susc[trunclen:]
    dsusc = dsusc[trunclen:]

    pref = scipy.constants.e * scipy.constants.e * 1e9 / \
        (3 * V * scipy.constants.k * T * scipy.constants.epsilon_0)

    susc *= -pref
    susc.imag *= -1  # we want -1 * Im susc (sometimes denoted as Chi'')

    dsusc *= pref  # std. deviation of susc is linear in susc
    nu /= 2 * np.pi  # now nu represents f instead of omega

    # Save susceptibility in a user-friendly text file:

    suscfilename = args.output+'susc.txt'

    np.savetxt(suscfilename,
               np.transpose([nu, susc.real, dsusc.real, susc.imag, dsusc.imag]),
               delimiter='\t',
               header='nu\tsusc\'\tstd_dev_susc\'\tsusc\'\'\tstd_dev_susc\'\'')

    print('Susceptibility data saved as ' + suscfilename + '\n')

    if args.noplots:
        print('User specified not to generate plots -- finished')
    else:
        # -------------------- Plotting ------------------------ #

        print('Calculations complete. Generating plots...\n')

        # Extraction of values useful for plotting:

        nuPeak = nu[np.argmax(susc.imag)]  # frequency at peak
        nuL = nu[1]  # lower x benchmark
        nuBuf = 1.4  # buffer factor for extra room in the x direction

        # max value of data
        suscMax = np.ceil(np.max([np.max(susc.real), np.max(susc.imag)]))
        suscL = np.min([susc.real[-1], susc.imag[1]]) / 2  # lower y benchmark
        suscBuf = 1.2  # buffer factor for extra room in the x direction


        # Plot lin-log:

        plt.figure(figsize=(8, 5))

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

        plt.figure(figsize=(8, 5))

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

        print('Plots generated -- finished\n')

    print('- - - - - - - - - - - - - - - - - - - - \n')

if __name__ == "__main__":
    main(firstarg=1)
