#!/usr/bin/env python
# coding: utf-8

from __future__ import absolute_import

import argparse
import importlib
import sys

from . import version

# Dictionary containing the app name and the directory
apps = {"carbonstructure": "build", "insert": "build", "debyer": "ana",
        "density": "ana", "diporder": "ana", "epsilon_bulk": "ana",
        "epsilon_cylinder": "ana", "epsilon_planar": "ana", "pertop": "build"}

applist = list(apps.keys())
applist.sort()

parser = argparse.ArgumentParser(description="""
    A collection of scripts to analyse and build systems for molecular dynamics simulations.""",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("program", type=str, help="Program to start",
                    choices=applist)
parser.add_argument('--version', action='version',
                    version="mdtools {}".format(version.__version__))


def main():

    try:
        if sys.argv[1] in applist:
            app = importlib.import_module(
                ".{}.{}".format(apps[sys.argv[1]], sys.argv[1]), package="mdtools")
        else:
            parser.parse_args()
    except IndexError:
        parser.parse_args()

    app.main()


if __name__ == "__main__":
    main()
