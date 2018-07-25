#!/usr/bin/env python
# coding: utf-8

from __future__ import absolute_import

import argparse
import code
import importlib
import sys
import warnings

from . import version

# Dictionary containing the app name and the directory
apps = {"carbonstructure": "build", "insert": "build", "debyer": "ana",
        "density": "ana", "diporder": "ana", "epsilon_bulk": "ana",
        "epsilon_cylinder": "ana", "epsilon_planar": "ana", "pertop": "build",
        "saxs": "ana", "velocity": "ana", "dielectric_spectrum": "ana",
        "density_cylinder": "ana"}

applist = list(apps.keys())
applist.sort()

parser = argparse.ArgumentParser(description="""
    A collection of scripts to analyse and build systems for molecular dynamics simulations.""",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("program", type=str, help="Program to start",
                    choices=applist)
parser.add_argument('--debug', action='store_true',
                    help="Run with debug options. Will start an interactive Python interpreter at the end of the program.")
parser.add_argument('--version', action='version',
                    version="mdtools {}".format(version.__version__))


def main():

    try:
        sys.argv.remove("--debug")
        DEBUG = True
    except ValueError:
        DEBUG = False
        warnings.filterwarnings("ignore")

    try:
        if sys.argv[1] in applist:
            app = importlib.import_module(
                ".{}.{}".format(apps[sys.argv[1]], sys.argv[1]), package="mdtools")
        else:
            parser.parse_args()
    except IndexError:
        parser.parse_args()

    app.main(DEBUG=DEBUG)

    if DEBUG:
        code.interact(
            banner="Start interactive Python interpreter.⁠ Acces program namespace by using app.<variable>.",
            local=dict(globals(), **locals())
            )

if __name__ == "__main__":
    main()
