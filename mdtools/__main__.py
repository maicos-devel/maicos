#!/usr/bin/env python
# coding: utf-8

import argparse
import sys

# Dictionary containing the app name and the directory
apps = {"carbonstructure": "build", "insert": "build", "debyer": "ana",
        "density": "ana", "diporder": "ana", "epsilon_bulk": "ana",
        "epsilon_cylinder": "ana", "epsilon_planar": "ana", "saxs": "ana",
        "pertop":"build"}

applist = list(apps.keys())
applist.sort()

parser = argparse.ArgumentParser(description="""
    A collection of scripts to analyse and build systems for molecular dynamics simulations.""",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("program", type=str, help="Program to start",
                    choices=applist)


def main():

    try:
        if sys.argv[1] in applist:
            exec("import {}.{} as app".format(apps[sys.argv[1]], sys.argv[1]))
        else:
            parser.parse_args()
    except IndexError:
        parser.parse_args()

    app.main()


if __name__ == "__main__":
    main()
