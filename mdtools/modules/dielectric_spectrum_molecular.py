#!/usr/bin/env python3
# coding: utf-8

import os
import sys

import MDAnalysis as mda
import numpy as np
import scipy.optimize
import scipy.constants
import time

from . import initilize_universe, print_frameinfo
from .. import initilize_parser

from ..utils import repairMolecules, FT, iFT, ScalarProdCorr, savetxt

# ========== PARSER ===========
# =============================

# parser object will already contain options for
# the topology, trajectory, begin, end, skipped frames and the box dimenions

parser = initilize_parser(add_traj_arguments=True)
parser.description = """This script, given molecular dynamics trajectory data, should produce a
    .txt file containing the single-molecule
    complex dielectric function as a function of the (linear, not radial -
    i.e. nu or f, rather than omega) frequency, along with the associated standard deviations.
    The algorithm is based on linear-response theory, specifically the equation
    chi(f) = -1/(3 V k_B T epsilon_0) FT{theta(t) <P(0) dP(t)/dt>}.
    By default, the polarization trajectory, timeseries array, and average system volume
    are saved in the
    working directory, and the data are reloaded from these files if they are present.
    Lin-log and log-log plots of the susceptibility are also produced by default."""
parser.add_argument(
    "-init",
    help="Causes initialization of the MDAnalysis universe. Alternatively,\
    polarization data is loaded from saved files, if they are present.",
    action="store_true")
parser.add_argument(
    '-temp',
    dest='temperature',
    type=float,
    default=300,
    help='Reference temperature.')
parser.add_argument(
    "-o", dest="output", default="", help="Prefix for the output files.")
parser.add_argument(
    "-u",
    dest="use",
    help="Looks for polarization and volume files with this prefix.\
    By default, the program looks for files with the prefix given in the output option -o."
)
parser.add_argument(
    "-segs",
    type=int,
    default=1,
    help="Sets the number of segments the trajectory is broken into.")
parser.add_argument(
    "-df",
    type=float,
    help="The desired frequency spacing in THz. This determines the minimum\
    frequency about which there is data. Overrides -segs option.")
parser.add_argument(
    "-noplots",
    help="Prevents plots from being generated.",
    action="store_true")
parser.add_argument(
    "-plotformat",
    default="pdf",
    choices=["png", "pdf", "ps", "eps", "svg"],
    help="Allows the user to choose the format of generated plots.")
parser.add_argument(
    "-ymin",
    type=float,
    help="Manually sets the minimum lower bound for the log-log plot.")
parser.add_argument(
    "-nobin",
    help="Prevents the data from being binned for graphing.\
    The data are by default binned logarithmically. This helps to reduce noise, particularly in\
    the high-frequency domain, and also prevents plot files from being too large.",
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

    global args

    # parse the arguments and saves them in an args object
    args = parser.parse_args(args=sys.argv[firstarg:])

    if not args.noplots:  # if plots are to be created

        import matplotlib.pyplot as plt

        # Colors/alpha values for plotting
        col1 = 'royalblue'
        col2 = 'crimson'
        col3 = 'grey'
        curve = 0.9
        shade = 0.1

        # Parameters for when data needs to be thinned for plotting

        Npp = 100  # Max number of points for susc plots
        Lpp = 20  # Num points of susc plotted with lin spacing: Lpp<Npp

    # ====== INITIALIZATION ======
    # ============================

    if len(args.output) > 0:
        args.output += "_"

    if args.use == None:
        args.use = args.output
    else:
        args.use += "_"

    # Either get polarization, volume, and time from a trajectory...

    if args.init\
    or not os.path.isdir(args.use+'PM_tseries')\
    or not os.path.isfile(args.use+'tseries.npy')\
    or not os.path.isfile(args.use+'V.txt'):

        # the MDAnalysis universe given by the user for analysis
        u = initilize_universe(args)

        NM = len(u.residues)
        dt = args.dt * args.skipframes
        Nframes = (args.endframe - args.beginframe) // args.skipframes

        args.frame = 0
        t = (np.arange(args.beginframe, args.endframe) - args.beginframe) * dt
        np.save(args.output + 'tseries.npy', t)

        # ======= POLARIZATION =======
        # ============================

        t_0 = time.clock()

        if not os.path.isdir(args.use +
                             'PM_tseries'):  # check if polarization is present

            print(
                'Polarization files not found: calculating polarization trajectories and average volume'
            )

            P = np.zeros((Nframes, NM, 3))
            V = 0.0

            print(
                "\rEvaluating frame: {:>12}        time: {:>12} ps".format(
                    args.frame, round(u.trajectory.time)),
                end="")

            for ts in u.trajectory[args.beginframe:args.endframe:args.
                                   skipframes]:

                # Calculations done in every frame
                V += ts.volume
                repairMolecules(u.atoms)
                for m in u.residues:
                    P[args.frame, m.resid, :] = np.dot(m.atoms.charges,
                                                       m.atoms.positions)
                args.frame += 1
                print_frameinfo(ts, args.frame)

            P /= 10  # MDA gives units of Angstroms, we use nm
            V *= 1e-3 / float(args.frame)  # normalization and unit conversion

            os.mkdir(args.output + 'PM_tseries')
            for m in range(0, NM):
                np.save(
                    args.output + 'PM_tseries/PM_tseries_' + str(m) + '.npy',
                    P[:, m, :])

            with open(args.output + 'V.txt', "w") as Vfile:
                Vfile.write(str(V))

        elif not os.path.isfile(args.use + 'V.txt'):

            print('Polarization file found: calculating average volume')
            V = 0.0
            print(
                "\rEvaluating frame: {:>12}       time: {:>12} ps".format(
                    args.frame, round(u.trajectory.time)),
                end="")

            for ts in u.trajectory[args.beginframe:args.endframe:args.
                                   skipframes]:

                # Calculations done in every frame
                V += ts.volume
                args.frame += 1
                print_frameinfo(ts, args.frame)

            V *= 1e-3 / float(args.frame)  # normalization and unit conversion
            with open(args.output + 'V.txt', "w") as Vfile:
                Vfile.write(str(V))

        else:

            print(
                'Polarization and volume files found: loading volume...',
                end='')
            V = np.loadtxt(args.use + 'V.txt')

        t_1 = time.clock()

        del P  # delete P as it's a very large array

    else:  # no universe initialization needed, data is loaded from files

        print('All data files found - not loading universe...', end='')
        t_0 = time.clock()
        t = np.load(args.use + 'tseries.npy')
        V = np.loadtxt(args.use + 'V.txt')
        NM = len(next(os.walk(args.use + 'PM_tseries'))[2])
        t_1 = time.clock()

    print("\nTook {:.2f} s".format(t_1 - t_0))
    print('Number of molecules: {0}'.format(NM))

    Nframes = len(t)
    print('Number of frames in trajectory: {0}'.format(Nframes))

    dt = (t[-1] - t[0]) / (Nframes - 1)

    # Set number of segments based on df if given:
    if not args.df == None:
        args.segs = np.max([int(Nframes * dt * args.df), 1])

    print('Number of segments in trajectory: {0}'.format(args.segs))

    seglen = int(Nframes / args.segs)

    print('Number of frames per segment: {0}'.format(seglen))

    if len(t) < 2 * seglen:  # if t too short to simply truncate
        t = np.append(t, t + t[-1] + dt)
    t = t[:2 * seglen]  # truncate t array

    # Prefactor for susceptibility:
    pref = scipy.constants.e*scipy.constants.e*1e9 / \
        (3*V*scipy.constants.k*args.temperature*scipy.constants.epsilon_0)

    # Susceptibility and errors:

    print('Calculating susceptibilty and errors:')

    P = np.load(args.use + 'PM_tseries/PM_tseries_0.npy')

    nu = FT(t, np.zeros(2 * seglen))[0]  # get freqs

    sm = np.zeros(2 * seglen, dtype=complex)
    susc = np.zeros(seglen, dtype=complex)
    dsusc = np.zeros(seglen, dtype=complex)

    for m in range(0, NM):  # loop over molecules

        print('\rMolecule {0} of {1}'.format(m + 1, NM), end='')
        P = np.load(args.use + 'PM_tseries/PM_tseries_' + str(m) + '.npy')
        sm = 0 + 0j

        for s in range(0, args.segs):  # loop over segments
            for i in range(0, len(P[0, :])):  # loop over x y z

                FP = FT(
                    t,
                    np.append(P[s * seglen:(s + 1) * seglen, i],
                              np.zeros(seglen)), False)
                sm += FP.real * FP.real + FP.imag * FP.imag

        sm *= nu * 1j

        # Get the real part by Kramers Kronig:
        sm.real = iFT(t, 1j * np.sign(nu) * FT(nu, sm, False), False).imag

        if m == 0:

            susc += sm[seglen:]

        else:

            dm = sm[seglen:] - (susc / m)
            susc += sm[seglen:]
            dif = sm[seglen:] - (susc / (m + 1))
            dm.real *= dif.real
            dm.imag *= dif.imag
            dsusc += dm  # variance by Welford's Method

    dsusc.real = np.sqrt(dsusc.real / args.segs)
    dsusc.imag = np.sqrt(dsusc.imag / args.segs)

    susc *= pref / (2 * seglen * args.segs * dt
                   )  # 1/2 b/c it's the full FT, not only half-domain
    dsusc *= pref / (2 * seglen * dt)

    nu = nu[seglen:] / (2 * np.pi
                       )  # now nu represents positive f instead of omega

    t_1 = time.clock()

    print('\nSusceptibility and errors calculated - took {0:.3} s'.format(t_1 -
                                                                          t_0))
    print('Frequency spacing: ~ {0:.5f} THz'.format(1 / (seglen * dt)))

    # ========= SAVE DATA ========
    # ============================

    # Save susceptibility in a user-friendly text file:

    suscfilename = args.output + 'suscM.txt'

    savetxt(
        suscfilename,
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

        if not (args.nobin or seglen <= Npp):  # data is binned

            bins = np.logspace(
                np.log(Lpp) / np.log(10),
                np.log(len(susc)) / np.log(10), Npp - Lpp + 1).astype(int)
            bins = np.unique(np.append(np.arange(Lpp), bins))[:-1]

            print(
                'Averaging data above datapoint {0} in log-spaced bins'.format(
                    Lpp))
            print('Binning data... ', end='')
            susc = Bin(susc, bins)
            dsusc = Bin(dsusc, bins)
            nu = Bin(nu, bins)

            print('finished - plotting {0} datapoints'.format(len(susc)))

        else:  # all data used
            print('Plotting all {0} datapoints'.format(len(susc)))

        nuBuf = 1.4  # buffer factor for extra room in the x direction

        # Plot lin-log:

        plt.figure()
        plt.title('Complex Dielectric Function')
        plt.ylabel('$\chi$')
        plt.xlabel('$\\nu$ [THz]')
        plt.grid()
        plt.xlim(nu[1] / nuBuf, nu[-1] * nuBuf)
        plt.xscale('log')
        plt.fill_between(
            nu,
            susc.real - dsusc.real,
            susc.real + dsusc.real,
            color=col2,
            alpha=shade)
        plt.fill_between(
            nu,
            susc.imag - dsusc.imag,
            susc.imag + dsusc.imag,
            color=col1,
            alpha=shade)
        plt.plot(
            nu,
            susc.real,
            color=col2,
            alpha=curve,
            linewidth=1,
            label='$\chi^{{\prime}}$')
        plt.plot(
            nu,
            susc.imag,
            color=col1,
            alpha=curve,
            linewidth=1,
            label='$\chi^{{\prime \prime}}$')
        plt.legend(loc='best')
        plt.savefig(
            args.output + 'suscM_linlog.' + args.plotformat,
            format=args.plotformat)
        plt.close()

        # Plot log-log:

        plt.figure()
        plt.title('Complex Dielectric Function')
        plt.ylabel('$\chi$')
        plt.xlabel('$\\nu$ [THz]')
        plt.grid()
        plt.xlim(nu[1] / nuBuf, nu[-1] * nuBuf)
        plt.yscale('log')
        plt.xscale('log')
        plt.fill_between(
            nu,
            susc.real - dsusc.real,
            susc.real + dsusc.real,
            color=col2,
            alpha=shade)
        plt.fill_between(
            nu,
            susc.imag - dsusc.imag,
            susc.imag + dsusc.imag,
            color=col1,
            alpha=shade)
        plt.plot(
            nu,
            susc.real,
            color=col2,
            alpha=curve,
            linewidth=1,
            label='$\chi^{{\prime}}$ : max = {0:.2f}'.format(np.max(susc.real)))
        plt.plot(
            nu,
            susc.imag,
            color=col1,
            alpha=curve,
            linewidth=1,
            label='$\chi^{{\prime \prime}}$ : max = {0:.2f}'.format(
                np.max(susc.imag)))
        if not args.ymin == None:
            plt.ylim(ymin=args.ymin)
        plt.legend(loc='best')
        plt.savefig(
            args.output + 'suscM_log.' + args.plotformat,
            format=args.plotformat)

        print('Susceptibility plots generated -- finished :)')

    print('\n============================================\n\n')

    if DEBUG:
        # Inject local variables into global namespace for debugging.
        for key, value in locals().items():
            globals()[key] = value


if __name__ == "__main__":
    main(firstarg=1)
