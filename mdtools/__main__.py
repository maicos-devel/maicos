#!/usr/bin/env python3
# coding: utf-8

import argparse
import os
import sys
import traceback
import warnings

import MDAnalysis as mda
import mdtools

from . import __version__
from . import __all__ as available_modules
from .utils import get_cli_input

# Try to use IPython shell for debug
try:
    import IPython
    use_IPython = True
except ImportError:
    import code
    use_IPython = False


class bcolors:
    warning = '\033[93m'
    fail = '\033[91m'
    endc = '\033[0m'


def _warning(message,
             category=UserWarning,
             filename='',
             lineno=-1,
             file=None,
             line=None):
    print("{}Warning: {}{}".format(bcolors.warning, message, bcolors.endc))


warnings.showwarning = _warning


def main():
    """The mdtools main function including the argument parser and universe
       initialization."""

    if '--bash_completion' in sys.argv:
        print(
            os.path.join(
                os.path.dirname(__file__), "share/mdtools-completion.bash"))
        sys.exit(0)

    parser = argparse.ArgumentParser(
        description="""
        A collection of scripts to analyse molecular dynamics simulations.""",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "program", type=str, help="Program to start", choices=available_modules)
    parser.add_argument(
        '--debug',
        action='store_true',
        help=
        "Run with debug options. Will start an interactive Python interpreter at the end of the program."
    )
    parser.add_argument(
        '--version', action='version', version="mdtools {}".format(__version__))

    try:
        sys.argv.remove("--debug")
        debug = True
    except ValueError:
        debug = False
        warnings.filterwarnings("ignore")

    try:
        if sys.argv[1] in available_modules:
            selected_module = getattr(mdtools, sys.argv[1])
        else:
            parser.parse_args()
    except IndexError:
        parser.parse_args()

    print('\n{}\n'.format(get_cli_input()))
    parser = argparse.ArgumentParser(
        prog="mdtools " + sys.argv[1],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        "-s",
        dest="topology",
        type=str,
        default="topol.tpr",
        help="The topolgy file. The FORMATs " + "           {}".format(
            ", ".join(mda._PARSERS.keys())) +
        "           are implemented in MDAnalysis.")
    parser.add_argument(
        "-top",
        dest="topology_format",
        type=str,
        default=None,
        help="Override automatic topology type detection." +
        "See topology for implemented formats")
    parser.add_argument(
        "-f",
        dest="trajectory",
        type=str,
        default=None,
        nargs="+",
        help="A single or multiple trajectory files. The FORMATs " +
        "           {}".format(", ".join(
            mda._READERS.keys())) + "           are implemented in MDAnalysis.")
    parser.add_argument(
        "-traj",
        dest="trajectory_format",
        type=str,
        default=None,
        help="Override automatic trajectory type detection." +
        "See trajectory for implemented formats")
    parser.add_argument(
        "-atom_style",
        dest="atom_style",
        type=str,
        default='None',
        help=
        "Manually set the atom_style information (currently only LAMMPS parser)."
        + "E.g. atom_style='id type x y z'.")
    parser.add_argument(
        "-b",
        dest="begin",
        type=float,
        default=0,
        help="start time (ps) for evaluation.")
    parser.add_argument(
        "-e",
        dest="end",
        type=float,
        default=None,
        help="end time (ps) for evaluation.")
    parser.add_argument(
        "-dt",
        dest="dt",
        type=float,
        default=0,
        help="time step (ps) to read analysis frame. If `0` take all frames")
    parser.add_argument(
        "-box",
        dest="box",
        type=float,
        default=None,
        nargs="+",
        help="Sets the box dimensions x y z [alpha beta gamma] (Å)." +
        "If 'None' dimensions from the trajectory will be used.")
    parser.add_argument(
        "-nt",
        dest="num_threads",
        type=int,
        default=0,
        help="Total number of threads to start (0 is guess)")

    _configure_parser = getattr(selected_module, "_configure_parser")
    _configure_parser(selected_module, parser)
    args = parser.parse_args(sys.argv[2:])

    if args.num_threads > 0:
        os.environ["OMP_NUM_THREADS"] = str(args.num_threads)

    print("Loading trajectory... ", end="")
    sys.stdout.flush()

    # prepare kwargs dictionary for other optional arguments
    ukwargs = {}
    if args.atom_style is not None:
        ukwargs['atom_style'] = args.atom_style

    u = mda.Universe(
        args.topology, topology_format=args.topology_format, **ukwargs)
    if args.trajectory is not None:
        u.load_new(args.trajectory, format=args.trajectory_format)
    print("Done!\n")

    if args.box is not None:
        if len(args.box) == 6:
            u.dimensions = args.box
        if len(args.box) == 3:
            u.dimensions[:3] = args.box
        else:
            sys.exit("{}Error: The boxdimensions must contain 3 entries for "
                     "the box vectors and possibly 3 more for the angles.{}"
                     "".format(bcolors.fail, bcolors.endc))

    try:
        ana_obj = selected_module(u.atoms, verbose=True, save=True)
        # Insert parser arguments into ana_obj
        for var in vars(args):
            if var not in [
                    "topology", "trajectory", "topology_format", "begin", "end",
                    "skipframes", "box"
            ]:
                vars(ana_obj)[var] = vars(args)[var]

        ana_obj.run(begin=args.begin, end=args.end, dt=args.dt)

    except Exception as e:
        if debug:
            traceback.print_exc()
        else:
            print("{}Error: {}{}".format(bcolors.fail, e, bcolors.endc))

    if args.num_threads > 0:
        del os.environ["OMP_NUM_THREADS"]

    if debug:
        # Inject local variables into global namespace for debugging.
        for key, value in locals().items():
            globals()[key] = value

        banner = "\nStarting interactive Python interpreter for debug.⁠.."
        if use_IPython:
            IPython.embed(banner1=banner)
        else:
            code.interact(banner=banner, local=dict(globals(), **locals()))


if __name__ == "__main__":
    main()
