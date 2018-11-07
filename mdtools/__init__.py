#!/usr/bin/env python
# coding: utf-8

import argparse
import os
import sys

from MDAnalysis import _READERS

from .version import __version__

sharePath = os.path.join(os.path.dirname(os.path.realpath(__file__)), "share")
__all__ = ["utils"]

def initilize_parser(add_traj_arguments=False):
    """Initilize an argparsing instance.

    *name* name of the module.
    *add_traj_arguments* adds basic trajectory analysis options.

    :returns: an argparse object"""
    print('\nCommand line was: mdtools {}\n'.format(' '.join(sys.argv[1:])))
    parser = argparse.ArgumentParser(
        prog="mdtools " + sys.argv[1],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    if add_traj_arguments:
        parser.add_argument("-s",   dest="topology",    type=str,
                            default="topol.tpr",            help="the topolgy file")
        parser.add_argument("-f",   dest="trajectory",  type=str,   default=None,
                            nargs="+", help="A single or multiple trajectory files.")
        parser.add_argument("-top",   dest="topology_format",   type=str, default=None,
                            help="Override automatic topology type detection. The FORMATs " +
                            "           {}".format(", ".join(_READERS.keys())) +
                            "           are implemented in MDAnalysis.")
        parser.add_argument("-b",   dest="begin",       type=float, default=0,
                            help="start time (ps) for evaluation")
        parser.add_argument("-e",   dest="end",         type=float,
                            default=None,                   help="end time (ps) for evaluation")
        parser.add_argument("-skip",  dest="skipframes",  type=int,
                            default=1,                      help="skip every N frames")
        parser.add_argument("-box", dest="box", type=float, nargs="+",
                            default=None,
                            help="Sets the box dimensions x y z [alpha beta gamma] (in Angstrom!).\
                           If 'None' dimensions from the trajectory will be used.")
    return parser
