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

from ..utils import repairMolecules, FT, iFT, ScalarProdCorr


# ========== PARSER ===========
# =============================

# parser object will already contain options for
# the topology, trajectory, begin, end, skipped frames and the box dimenions

parser = initilize_parser(add_traj_arguments=True)
parser.description = """This script, given molecular dynamics trajectory data, should produce a
    .txt file containing the complex dielectric function as a function of the (linear, not radial -
    i.e. nu or f, rather than omega) frequency, along with the associated standard deviations.
    The two algorithms are based on linear-response theory, specifically the equation\
    chi(f) = -1/(3 V k_B T epsilon_0) FT{theta(t) <P(0) dP(t)/dt>}.\
    By default, the polarization trajectory and the average system volume are saved in the
    working directory, and the data are reloaded from these files if they are present.
    Lin-log and log-log plots of the susceptibility are also produced by default,\
    along with a plot of the truncation-length fit if method 1 is used."""
parser.add_argument("-method", type=int, default=1, choices=[1,2],
                    help="Method 1 follows the longer, more intuitive procedure involving 3 FFTs\
    and a numerical time derivative. Method 2 uses 1 FFT and multiplys by the frequency,\
    and uses 2 more FFT's in Kramers Kronig to obtain the real part of the frequency.")
parser.add_argument('-temp',   dest='temperature',      type=float,
                    default=300, help='Reference temperature.')
parser.add_argument("-o", dest="output",
                    default="", help="Prefix for the output files.")
parser.add_argument("-u", dest="use",
                    help="Looks for polarization and volume files with this prefix.\
    By default, the program looks for files with the prefix -o.")
parser.add_argument("-truncfac", type=float, default=30.0,
                    help="Truncation factor.\
    By default, the autocorrelation of the polarization is fit with A*exp( -t/Tau ),\
    and the truncation length is then taken as truncfac*Tau.\
    Alternatively, the truncation length in number of steps may be specified with -trunclen.")
parser.add_argument("-trunclen", type=float,
                    help="Truncation length in picoseconds.\
    Specifying a value overrides the fitting procedure otherwise used to find the truncation length.")
parser.add_argument("-segs", type=int, default=20,
                    help="Sets the number of segments the trajectory is broken into.\
    This overrides the -df option.")
parser.add_argument("-df", type=float,
                    help="The desired frequency spacing in THz. This determines the minimum\
    frequency about which there is data. Overrides -segs option.")
parser.add_argument("-noplots",
                    help="Prevents plots from being generated.", action="store_true")
parser.add_argument("-plotformat", default="pdf", choices=["png", "pdf", "ps", "eps", "svg"],
                    help="Allows the user to choose the format of generated plots.")
parser.add_argument("-ymin", type=float,
                    help="Manually sets the minimum lower bound for the log-log plot.")
parser.add_argument("-nobin",
                    help="Prevents the data from being binned for graphing.\
    The data are by default binned logarithmically. This helps to reduce noise, particularly in\
    the high-frequency domain, and also prevents plots from being massive files."
    , action="store_true")

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


def Bin(a, bins):   # Averages array values in bins for easier plotting
    # note: "bins" array should contain the INDEX (integer) where that bin begins

    if np.iscomplex(a).any():
        avg = np.zeros(len(bins), dtype=complex) # average of data
    else:
        avg = np.zeros(len(bins))

    count = np.zeros(len(bins), dtype=int)
    ic = -1

    for i in range(0, len(a)):
        if i in bins:
            ic += 1 # index for new average
        avg[ic] += a[i]
        count[ic] += 1

    return avg / count


def single_exp(x, A, D): # Single exponential for fitting:

    return np.absolute(A) * np.exp(-x / D)


# ========== MAIN ============
# ============================

def main(firstarg=2, DEBUG=False):

    print('\n====== DIELECTRIC SPECTRUM CALCULATOR ======')

    global args

    # parse the arguments and saves them in an args object
    args = parser.parse_args(args=sys.argv[firstarg:])

    if not args.noplots: # if plots are to be created

        import matplotlib.pyplot as plt

        # Colors/alpha values for plotting
        col1 = 'royalblue'
        col2 = 'crimson'
        col3 = 'grey'
        curve = 0.9
        shade = 0.1

        # Parameters for when data needs to be thinned for plotting

        Npp = 100 # Max number of points for susc plots
        Lpp = 20 # Num points of susc plotted with lin spacing: Lpp<Npp


    # ====== INITIALIZATION ======
    # ============================

    # the MDAnalysis universe given by the user for analysis
    u = initilize_universe(args)

    dt = args.dt*args.skipframes
    Nframes = (args.endframe - args.beginframe) // args.skipframes

    # Find a suitable number of segments if it's not specified:
    if not args.df == None:
        args.segs = np.max([int(Nframes*dt*args.df), 2])

    seglen = int(Nframes / args.segs)

    if len(args.output) > 0:
        args.output += "_"

    if args.use == None:
        args.use = args.output

    args.frame = 0
    t = (np.arange(args.beginframe, args.endframe) - args.beginframe)*dt

    # ======= POLARIZATION =======
    # ============================

    t_0 = time.clock()

    if not os.path.isfile(args.use+'P_tseries.npy'): # check if polarization is present

        print('Polarization file not found: calculating polarization trajectory and average volume')

        P = np.zeros((Nframes , 3))
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

    elif not os.path.isfile(args.use+'V.txt'):

        print('Polarization file found: loading polarization and calculating average volume')
        P = np.load(args.use+'P_tseries.npy')
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

        print('Polarization and volume files found: loading both...', end='')
        P = np.load(args.use+'P_tseries.npy')
        V = np.loadtxt(args.use+'V.txt')

    t_1 = time.clock()
    print("\nTook {:.2f} s".format(t_1 - t_0))
    t_0 = time.clock()

    # Prefactor for susceptibility:
    pref = scipy.constants.e * scipy.constants.e * 1e9 / \
        (3 * V * scipy.constants.k * args.temperature * scipy.constants.epsilon_0)

    # ========= METHOD 1 =========
    # ============================

    if args.method == 1:

        # Autocorrelation

        print('Calculating the autocorrelation... ', end='')
        P_P = ScalarProdCorr(P)  # Autocorrelation fn of P for all timesteps
        print('Done!')

        # Truncation:

        print('Finding the truncation length for the autocorrelation...')
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

                sk = 1 # how many data points to skip when plotting

                if plotlen > 2*Npp:
                    sk = plotlen // (2*Npp) + 1 # ~2x as many points as for susc figs

                plt.figure(figsize=(8, 5.657))
                plt.title('Exponential Fit to Determine Truncation Length')
                plt.ylabel('$<P(0)$ $P(t)>$')
                plt.xlabel('t [ps]')
                plt.xlim(-0.02 * t[plotlen], t[plotlen])
                plt.axvline(x=t[args.trunclen], linewidth=1, color=col3, alpha=curve, linestyle='--',
                            label='truncation length = {0:.4} ps'.format(t[args.trunclen]))
                plt.plot(t[:plotlen:sk], P_P[:plotlen:sk], color=col1, alpha=curve, marker='.',
                         markersize=4, linestyle='', label='$<P(0)$ $P(t)>$')
                plt.plot(t[:plotlen:sk], single_exp(t[:plotlen:sk], p_opt[0], p_opt[1]),
                         linewidth=1, color=col2, alpha=curve, label='fit: ~exp( -t/{0:.4} )'.format(p_opt[1]))
                plt.legend(loc='best')
                plt.savefig(args.output+'P_autocorr_trunc_fit.'+args.plotformat, format=args.plotformat)
                plt.close()

            print('Truncation length set via exponential fit to {0} steps, i.e. {1:.6} ps'.format(
                args.trunclen, t[args.trunclen]))

        else: # args.trunclen specified by user
            # convert from picoseconds into the frame index
            args.trunclen = int(args.trunclen / dt)
            print('Truncation length set manually to {0} steps, i.e. {1:.3} ps'.format(
                args.trunclen, args.trunclen * dt))

        # Truncate and pad with zeros:

        if len(t) < 2 * args.trunclen: # if t too short to simply truncate
            t = np.append(t, t+t[-1]+dt)
        t = np.resize(t, 2 * args.trunclen)  # truncate   
        P_P = np.append(np.resize(P_P, args.trunclen), np.zeros(args.trunclen))  # truncate, pad w zeros

        # Susceptibility and errors:

        print('Calculating susceptibilty and errors via Method 1...')

        if args.trunclen > seglen: # if segments too short

            args.segs = int(Nframes / args.trunclen)
            seglen = int(Nframes / args.segs) # lengthen seglen to >trunclen

        nu, susc = FT(t, TimeDerivative5PS(P_P, dt)) # total susc
        dsusc = np.zeros(2 * args.trunclen, dtype=complex)

        for s in range(0, args.segs):

            P_P = ScalarProdCorr(P[s*seglen:(s+1)*seglen, :])
            P_P = np.append(np.resize(P_P, args.trunclen), np.zeros(args.trunclen))
            ss = FT(t, TimeDerivative5PS(P_P, dt), False)
            dsusc += (ss - susc).real*(ss - susc).real + \
                1j * (ss - susc).imag*(ss - susc).imag

        dsusc.real = np.sqrt(dsusc.real)*pref / args.segs  # convert from variance to std. deviation
        dsusc.imag = np.sqrt(dsusc.imag)*pref / args.segs  # convert from variance to std. deviation

        susc.real *= -pref
        susc.imag *= pref  # we want -1 * Im susc (sometimes denoted as Chi'')

    # ========= METHOD 2 =========
    # ============================

    if args.method == 2:

        # Susceptibility and errors:

        print('Calculating susceptibilty and errors\
            \nUsing method 2 (real part via Kramers Kronig)...')

        t = t[:2*seglen] # truncate t array (it's automatically longer than 2*seglen)
        ss = np.zeros((2*seglen, args.segs), dtype=complex) # ss[t:segment]

        nu = FT(t, np.append(P[:seglen, 0], np.zeros(seglen)))[0] # get freqs

        for s in range(0, args.segs):
            for i in range(0, len(P[0,:])):
 
                FP = FT(t, np.append(P[s*seglen:(s+1)*seglen, i], np.zeros(seglen)), False)
                ss[:,s] += FP.real*FP.real + FP.imag*FP.imag

            ss[:,s] *= nu*1j*pref / (2*seglen*dt)
            # (1/2 because it's the full FT, not only the pos domain)

            # Get the real part by Kramers Kronig:
            ss[:,s].real = iFT(t, 1j*np.sign(nu)*FT(nu, ss[:,s], False), False).imag

        susc = np.mean(ss, axis=1) # susc[t]
        dsusc = np.zeros(2*seglen, dtype=complex)

        for s in range(0, args.segs):
            dsusc += (ss[:,s] - susc).real*(ss[:,s] - susc).real + \
                1j * (ss[:,s] - susc).imag*(ss[:,s] - susc).imag

        dsusc.real = np.sqrt(dsusc.real) / args.segs
        dsusc.imag = np.sqrt(dsusc.imag) / args.segs
        args.trunclen = seglen

    t_1 = time.clock()

    print('Susceptibility and errors calculated - took {0:.3} s'.format(t_1 - t_0))
    print('Number of segments:\t{0}'.format(args.segs))
    print('Length of segments:\t{0} frames, {1:.0f} ps'.format(seglen, seglen*dt))
    print('Frequency spacing: \t~ {0:.5f} THz'.format(args.segs/(Nframes*dt)))

    # ========= SAVE DATA ========
    # ============================

    # Discard negative-frequency data; contains the same information as positive regime:

    nu = nu[args.trunclen:]
    susc = susc[args.trunclen:]
    dsusc = dsusc[args.trunclen:]

    nu /= 2 * np.pi  # now nu represents f instead of omega

    # Save susceptibility in a user-friendly text file:

    suscfilename = args.output+'susc.txt'

    np.savetxt(suscfilename,
           np.transpose([nu, susc.real, dsusc.real, susc.imag, dsusc.imag]),
           delimiter='\t',
           header='freq\tsusc\'\tstd_dev_susc\'\t-susc\'\'\tstd_dev_susc\'\'')

    print('Susceptibility data saved as ' + suscfilename)

    # ==== OPTIONAL PLOTTING =====
    # ============================

    if args.noplots:
        print('User specified not to generate plots -- finished :)')

    else:

        print('Calculations complete. Generating plots...')

        # Bin data if there are too many points:
        # NOTE: matplotlib.savefig() will plot 50,000 points, but not 60,000

        if not (args.nobin or args.trunclen <= Npp): # all data is used

            bins = np.logspace(np.log(Lpp) / np.log(10), np.log(len(susc)) / np.log(10), Npp-Lpp).astype(int)
            bins = np.unique(np.append(np.arange(Lpp), bins))[:-1]

            susc = Bin(susc, bins)
            dsusc = Bin(dsusc, bins)
            nu = Bin(nu, bins)

            print('Averaging data above datapoint {0} in log-spaced bins'.format(Lpp))
            print('Plotting {0} datapoints'.format(len(susc)))

        else: # data is binned
            print('Plotting all {0} datapoints'.format(len(susc)))

        nuBuf = 1.4  # buffer factor for extra room in the x direction

        # Plot lin-log:

        plt.figure(figsize=(8, 5.657))
        plt.title('Complex Dielectric Function')
        plt.ylabel('$\chi$')
        plt.xlabel('$\\nu$ [THz]')
        plt.grid()
        plt.xlim(nu[1] / nuBuf, nu[-1] * nuBuf)
        plt.xscale('log')
        plt.fill_between(nu, susc.real - dsusc.real, susc.real + dsusc.real,
            color=col2, alpha=shade)
        plt.fill_between(nu, susc.imag - dsusc.imag, susc.imag + dsusc.imag,
            color=col1, alpha=shade)
        plt.plot(nu, susc.real, color=col2, alpha=curve, linewidth=1, label='$\chi^{{\prime}}$')
        plt.plot(nu, susc.imag, color=col1, alpha=curve, linewidth=1, label='$\chi^{{\prime \prime}}$')
        plt.legend(loc='best')
        plt.savefig(args.output + 'susc_linlog.'+args.plotformat, format=args.plotformat)
        plt.close()

        # Plot log-log:

        plt.figure(figsize=(8, 5.657))
        plt.title('Complex Dielectric Function')
        plt.ylabel('$\chi$')
        plt.xlabel('$\\nu$ [THz]')
        plt.grid()
        plt.xlim(nu[1] / nuBuf, nu[-1] * nuBuf)
        plt.yscale('log')
        plt.xscale('log')
        plt.fill_between(nu, susc.real - dsusc.real, susc.real + dsusc.real,
            color=col2, alpha=shade)
        plt.fill_between(nu, susc.imag - dsusc.imag, susc.imag + dsusc.imag, 
            color=col1, alpha=shade)
        plt.plot(nu, susc.real, color=col2, alpha=curve, linewidth=1,
             label='$\chi^{{\prime}}$ : max = {0:.2f}'.format(np.max(susc.real)))
        plt.plot(nu, susc.imag, color=col1, alpha=curve, linewidth=1,
             label='$\chi^{{\prime \prime}}$ : max = {0:.2f}'.format(np.max(susc.imag)))
        if not args.ymin == None:
            plt.ylim(ymin=args.ymin)     
        plt.legend(loc='best')
        plt.savefig(args.output + 'susc_log.'+args.plotformat, format=args.plotformat)

        print('Susceptibility plots generated -- finished :)')

    print('\n============================================\n\n')

    if DEBUG:
        # Inject local variables into global namespace for debugging.
        for key, value in locals().items():
            globals()[key] = value

if __name__ == "__main__":
    main(firstarg=1)
