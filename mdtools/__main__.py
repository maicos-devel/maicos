#!/usr/bin/env python
# coding: utf-8

from __future__ import absolute_import, print_function

import argparse
import importlib
import inspect
import math
import os
import sys
import traceback
import warnings

from MDAnalysis import _PARSERS, _READERS

from . import version
from .ana import __all__ as anamodules
from .build import __all__ as buildmodules

# Try to use IPython shell for debug
try:
    import IPython
    use_IPython = True
except ImportError:
    import code
    use_IPython = False


def main():
    """The mdtools main function including the argument parser and universe
       initialization."""
    # Dictionary containing the app name and the directory
    apps = {}
    for module in anamodules:
        module = module.split(".")
        apps[module[-1]] = "ana." + ".".join(module[:-1])
    for module in buildmodules:
        module = module.split(".")
        apps[module[-1]] = "build." + ".".join(module[:-1])

    applist = list(apps.keys())
    applist.sort()

    parser = argparse.ArgumentParser(
        description="""
        A collection of scripts to analyse and build systems for molecular dynamics simulations.""",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "program", type=str, help="Program to start", choices=applist)
    parser.add_argument(
        '--debug',
        action='store_true',
        help=
        "Run with debug options. Will start an interactive Python interpreter at the end of the program."
    )
    parser.add_argument(
        '--version',
        action='version',
        version="mdtools {}".format(version.__version__))

    try:
        sys.argv.remove("--debug")
        debug = True
    except ValueError:
        debug = False
        warnings.filterwarnings("ignore")

    try:
        if sys.argv[1] in applist:
            app = importlib.import_module("mdtools.{}".format(
                apps[sys.argv[1]]))
            met = getattr(app, sys.argv[1])
        else:
            parser.parse_args()
    except IndexError:
        parser.parse_args()

    print('\nCommand line was: mdtools {}\n'.format(' '.join(sys.argv[1:])))
    parser = argparse.ArgumentParser(
        prog="mdtools " + sys.argv[1],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    if "ana" in apps[sys.argv[1]]:
        parser.add_argument(
            "-s",
            dest="topology",
            type=str,
            default="topol.tpr",
            help="The topolgy file. The FORMATs " + "           {}".format(
                ", ".join(_PARSERS.keys())) +
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
                _READERS.keys())) + "           are implemented in MDAnalysis.")
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
            help="start time (ps) for evaluation")
        parser.add_argument(
            "-e",
            dest="end",
            type=float,
            default=None,
            help="end time (ps) for evaluation")
        parser.add_argument(
            "-skip",
            dest="skipframes",
            type=int,
            default=1,
            help="skip every N frames")
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

    _configure_parser = [
        i for i in inspect.getmembers(met) if '_configure_parser' in i[0]
    ][0][1]
    _configure_parser(met, parser)
    args = parser.parse_args(sys.argv[2:])

    if "ana" in apps[sys.argv[1]]:

        if args.num_threads > 0:
            os.environ["OMP_NUM_THREADS"] = str(args.num_threads)

        import MDAnalysis

        print("Loading trajectory... ", end="")
        sys.stdout.flush()

        # prepare kwargs dictionary for other optional arguments
        ukwargs = {}
        if args.atom_style is not None:
            ukwargs['atom_style'] = args.atom_style

        u = MDAnalysis.Universe(
            args.topology, topology_format=args.topology_format, **ukwargs)
        if args.trajectory is not None:
            u.load_new(args.trajectory, format=args.trajectory_format)
        print("Done!")

        args.begin = int(math.ceil(args.begin // u.trajectory.dt))

        if args.end != None:
            args.end = int(math.ceil(args.end // u.trajectory.dt))
        else:
            args.end = int(math.ceil(u.trajectory.totaltime // u.trajectory.dt))

        args.end += 1  # catch also last frame in loops

        if args.begin > args.end:
            sys.exit("Start time is larger than end time!")

        if args.box != None:
            assert (len(args.box) == 6 or len(args.box) == 3),\
                'The boxdimensions must contain 3 entries for the box vectors and possibly 3 more for the angles.'
            if len(args.box) == 6:
                u.dimensions = np.array(args.box)
            else:
                u.dimensions[:2] = np.array(args.box)

        try:
            ana_obj = met(u.atoms, verbose=True, save=True)
            print("")
            # Insert parser arguments into ana_obj
            for var in vars(args):
                if var not in [
                        "topology", "trajectory", "topology_format", "begin",
                        "end", "skipframes", "box"
                ]:
                    vars(ana_obj)[var] = vars(args)[var]

            ana_obj.run(start=args.begin, stop=args.end, step=args.skipframes)

        except Exception as e:
            if debug:
                traceback.print_exc()
            else:
                raise e

        if args.num_threads > 0:
            del os.environ["OMP_NUM_THREADS"]

    elif apps[sys.argv[1]] == "build":
        pass

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
