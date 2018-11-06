#!/usr/bin/env python
# coding: utf-8

# Mandatory imports
from __future__ import absolute_import, division, print_function

import os
import sys
import time

import MDAnalysis as mda
import numpy as np
import scipy.constants

from . import initilize_universe, print_frameinfo
from .. import initilize_parser
from ..utils import FT, ScalarProdCorr, iFT, repairMolecules

# TODO set up script to calc spectrum at intervals while calculating polarization
# for very big-data trajectories

# TODO merge with molecular version?

# ========== PARSER ===========
# =============================

# parser object will already contain options for
# the topology, trajectory, begin, end, skipped frames and the box dimenions

parser = initilize_parser(add_traj_arguments=True)
parser.description = """This script, given molecular dynamics trajectory data, should produce a
    .txt file containing the complex dielectric function as a function of the (linear, not radial -
    i.e. nu or f, rather than omega) frequency, along with the associated standard deviations.
    The algorithm is based on the Fluctuation Dissipation Relation (FDR):\
    chi(f) = -1/(3 V k_B T epsilon_0) FT{theta(t) <P(0) dP(t)/dt>}.\
    By default, the polarization trajectory, time series array and the average system volume are
    saved in the working directory, and the data are reloaded from these files if they are present.
    Lin-log and log-log plots of the susceptibility are also produced by default."""
parser.add_argument("-recalc",
                    help="Forces to recalculate the polarization, regardless if it is already present.",
    action="store_true")
parser.add_argument('-temp',   dest='temperature',      type=float,
                    default=300, help='Reference temperature.')
parser.add_argument("-o", dest="output",
                    default="", help="Prefix for the output files.")
parser.add_argument("-u", dest="use",
                    help="Looks for polarization and volume files with this prefix.\
    By default, the program looks for files with the prefix -o.")
parser.add_argument("-segs", type=int, default=20,
                    help="Sets the number of segments the trajectory is broken into.")
parser.add_argument("-df", type=float,
                    help="The desired frequency spacing in THz. This determines the minimum\
    frequency about which there is data. Overrides -segs option.")
parser.add_argument("-noplots",
                    help="Prevents plots from being generated.", action="store_true")
parser.add_argument("-plotformat", default="pdf", choices=["png", "pdf", "ps", "eps", "svg"],
                    help="Allows the user to choose the format of generated plots.")
parser.add_argument("-ymin", type=float,
                    help="Manually sets the minimum lower bound for the log-log plot.")
parser.add_argument("-bins", type=int, default=200,
                    help="Determines the number of bins used for data averaging;\
    (this parameter sets the upper limit).\
    The data are by default binned logarithmically. This helps to reduce noise, particularly in\
    the high-frequency domain, and also prevents plot files from being too large.")
parser.add_argument("-binafter", type=int, default=20,
                    help="The number of low-frequency data points that are left unbinned.")
parser.add_argument("-nobin",
                    help="Prevents the data from being binned altogether.\
    This can result in very large plot files and errors.",
    action="store_true")

# ======== DEFINITIONS ========
# =============================


def Bin(a, bins):
    """Averages array values in bins for easier plotting. 
    Note: "bins" array should contain the INDEX (integer) where that bin begins"""

    if np.iscomplex(a).any():
        avg = np.zeros(len(bins), dtype=complex)  # average of data
    else:
        avg = np.zeros(len(bins))

    count = np.zeros(len(bins), dtype=int)
    ic = -1

    for i in range(0, len(a)):
        if i in bins:
            ic += 1  # index for new average
        avg[ic] += a[i]
        count[ic] += 1

    return avg / count


# ========== MAIN ============
# ============================

def main(firstarg=2, DEBUG=False):

    print('\n====== DIELECTRIC SPECTRUM CALCULATOR ======\n')


    # ====== INITIALIZATION ======
    # ============================

    global args

    # parse the arguments and saves them in an args object
    args = parser.parse_args(args=sys.argv[firstarg:])

    if len(args.output) > 0:
        args.output += "_"

    if args.use == None:
        args.use = args.output
    else:        
        args.use += "_"

    # Check file existence
    t_exists = os.path.isfile(args.use + 'tseries.npy')
    if t_exists and not args.recalc:
        t = np.load(args.use + 'tseries.npy')

    V_exists = os.path.isfile(args.use + 'V.txt')
    if V_exists and not args.recalc:
        with open(args.output + 'V.txt', "r") as Vfile:
            V = float(Vfile.readline())

    P_exists = os.path.isfile(args.use + 'P_tseries.npy')
    if P_exists and not args.recalc:  # check if polarization is present
        P = np.load(args.use + 'P_tseries.npy')

    if args.recalc or not t_exists or not V_exists or not P_exists:
        print('Loading universe and generating polarization trajectory,\nvolume and time series array:')
        # the MDAnalysis universe given by the user for analysis
        u = initilize_universe(args)

        dt = args.dt * args.skipframes
        Nframes = (args.endframe - args.beginframe) // args.skipframes
        args.frame = 0

        t_0 = time.clock()
        P = np.zeros((Nframes, 3))
        V = 0.0
        t = (np.arange(args.beginframe, args.endframe) - args.beginframe) * dt

        print("\rEvaluating frame: {:>12}        time: {:>12} ps".format(
            args.frame, round(u.trajectory.time)), end="")

        for ts in u.trajectory[args.beginframe:args.endframe:args.skipframes]:
            if not V_exists or args.recalc:
                V += ts.volume
            if not P_exists or args.recalc:
                repairMolecules(u.atoms)
                P[args.frame, :] = np.dot(u.atoms.charges, u.atoms.positions)

            args.frame += 1
            print_frameinfo(ts, args.frame)

        t_1 = time.clock()
        print("\nTook {:.2f} s".format(t_1 - t_0))

        if not t_exists or args.recalc:
            np.save(args.output + 'tseries.npy', t)

        if not V_exists or args.recalc:
            V *= 1e-3 / float(args.frame)  # normalization and unit conversion
            with open(args.output + 'V.txt', "w") as Vfile:
                Vfile.write(str(V))

        if not P_exists or args.recalc:
            P /= 10  # MDA gives units of Angstroms, we use nm
            np.save(args.output + 'P_tseries.npy', P)

    else:
        print('All data files found: loading files directly')

        Nframes = len(t)
        dt = (t[-1] - t[0]) / (Nframes - 1)

    t_0 = time.clock()

    # Find a suitable number of segments if it's not specified:
    if not args.df == None:
        args.segs = np.max([int(Nframes * dt * args.df), 2])

    seglen = int(Nframes / args.segs)

    # Prefactor for susceptibility:
    pref = scipy.constants.e * scipy.constants.e * 1e9 / \
        (3 * V * scipy.constants.k * args.temperature * scipy.constants.epsilon_0)

    # Susceptibility and errors:

    print('Calculating susceptibilty and errors...')

    if len(t) < 2 * seglen:  # if t too short to simply truncate
        t = np.append(t, t + t[-1] + dt)
    # truncate t array (it's automatically longer than 2*seglen)
    t = t[:2 * seglen]

    nu = FT(t, np.append(P[:seglen, 0], np.zeros(seglen)))[0]  # get freqs

    susc = np.zeros(seglen, dtype=complex) # susceptibility
    dsusc = np.zeros(seglen, dtype=complex) # std deviation of susceptibility
    ss = np.zeros((2 * seglen), dtype=complex)  # susceptibility for current seg

    for s in range(0, args.segs): # loop over segs

        print('\rSegment {0} of {1}'.format(s + 1, args.segs), end='')
        ss = 0 + 0j

        for i in range(0, len(P[0, :])): # loop over x, y, z

            FP = FT(t, np.append(
                P[s * seglen:(s + 1) * seglen, i], np.zeros(seglen)), False)
            ss += FP.real * FP.real + FP.imag * FP.imag

        ss *= nu * 1j

        # Get the real part by Kramers Kronig:
        ss.real = iFT(t, 1j * np.sign(nu) *
                            FT(nu, ss, False), False).imag

        if s == 0:

            susc += ss[seglen:]
        
        else:

            ds = ss[seglen:] - (susc / s)
            susc += ss[seglen:]
            dif = ss[seglen:] - (susc / (s + 1))
            ds.real *= dif.real
            ds.imag *= dif.imag
            dsusc += ds # variance by Welford's Method

    dsusc.real = np.sqrt(dsusc.real)
    dsusc.imag = np.sqrt(dsusc.imag)

    susc *= pref / (2 * seglen * args.segs * dt) # 1/2 b/c it's the full FT, not only half-domain
    dsusc *= pref / (2 * seglen * args.segs * dt)

    # Discard negative-frequency data; contains the same information as positive regime:

    nu = nu[seglen:] / (2 * np.pi) # now nu represents positive f instead of omega

    t_1 = time.clock()

    print('\nSusceptibility and errors calculated - took {0:.3} s'.format(t_1 - t_0))
    print('Length of segments:    {0} frames, {1:.0f} ps'.format(
        seglen, seglen * dt))
    print('Frequency spacing:    ~ {0:.5f} THz'.format(
        args.segs / (Nframes * dt)))


    # ========= SAVE DATA ========
    # ============================

    # Save susceptibility in a user-friendly text file:

    suscfilename = args.output + 'susc.txt'

    np.savetxt(suscfilename,
               np.transpose(
                   [nu, susc.real, dsusc.real, susc.imag, dsusc.imag]),
               delimiter='\t',
               header='freq\tsusc\'\tstd_dev_susc\'\t-susc\'\'\tstd_dev_susc\'\'')

    print('Susceptibility data saved as ' + suscfilename)

    # Bin data if there are too many points:

    if not (args.nobin or seglen <= args.bins):

        bins = np.logspace(np.log(args.binafter) / np.log(10), np.log(len(susc)) / 
            np.log(10), args.bins - args.binafter + 1).astype(int)
        bins = np.unique(np.append(np.arange(args.binafter), bins))[:-1]

        susc = Bin(susc, bins)
        dsusc = Bin(dsusc, bins)
        nu = Bin(nu, bins)

        print('Binning data above datapoint {0} in log-spaced bins'.format(args.binafter))
        print('Binned data consists of {0} datapoints'.format(len(susc)))

        suscfilename = args.output + 'susc_binned.txt'

        np.savetxt(suscfilename,
                   np.transpose(
                       [nu, susc.real, dsusc.real, susc.imag, dsusc.imag]),
                   delimiter='\t',
                   header='freq\tsusc\'\tstd_dev_susc\'\t-susc\'\'\tstd_dev_susc\'\'')

        print('Binned susceptibility data saved as ' + suscfilename)


    else:  # data is binned
        print('Not binning data: there are {0} datapoints'.format(len(susc)))


    # ==== OPTIONAL PLOTTING =====
    # ============================

    if args.noplots:
        print('User specified not to generate plots -- finished :)')

    else:
        print('Generating plots...')

        import matplotlib.pyplot as plt

        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')

        # Colors/alpha values/labels/params for plotting
        col1 = 'royalblue'
        col2 = 'crimson'
        curve = 0.9
        shade = 0.15
        lw = 1.0
        nuBuf = 1.4  # buffer factor for extra room in the x direction
        cp = '$\chi^{{\prime}}$'
        cpp = '$\chi^{{\prime \prime}}$'
        width = 3.5 # width in inches

        # Plots:

        def my_plot():

            fig, ax = plt.subplots(1, figsize=[width,width/np.sqrt(2)])
            ax.set_ylabel('$\chi$')
            ax.set_xlabel('$\\nu$ [THz]')
            ax.set_xlim(nu[1] / nuBuf, nu[-1] * nuBuf)
            ax.set_xscale('log')
            ax.set_yscale(yscale)
            ax.fill_between(nu[1:], susc.real[1:] - dsusc.real[1:], susc.real[1:] +
                dsusc.real[1:], color=col2, alpha=shade)
            ax.fill_between(nu[1:], susc.imag[1:] - dsusc.imag[1:], susc.imag[1:] +
                dsusc.imag[1:], color=col1, alpha=shade)
            ax.plot(nu[:2], susc.real[:2], color=col2, alpha=curve, linestyle=':', linewidth=lw)
            ax.plot(nu[:2], susc.imag[:2], color=col1, alpha=curve, linestyle=':', linewidth=lw)
            ax.plot(nu[1:], susc.real[1:], color=col2, alpha=curve, label=cp, linewidth=lw)
            ax.plot(nu[1:], susc.imag[1:], color=col1, alpha=curve, label=cpp, linewidth=lw)

            if i == 0 and (not args.ymin == None):
                plt.set_ylim(ymin=args.ymin)
            ax.legend(loc='best', frameon=False)
            fig.tight_layout(pad=0.1)
            fig.savefig(plotname, format=args.plotformat)

        yscale = 'log'
        plotname = args.output + 'susc_log.' + args.plotformat
        my_plot() # log-log

        yscale = 'linear'
        plotname = args.output + 'susc_linlog.' + args.plotformat
        my_plot() # lin-log

        plt.close('all')

        print('Susceptibility plots generated -- finished :)')

    print('\n============================================\n\n')

    if DEBUG:
        # Inject local variables into global namespace for debugging.
        for key, value in locals().items():
            globals()[key] = value


if __name__ == "__main__":
    main(firstarg=1)
