#!/usr/bin/env python
# coding: utf-8

import argparse
import sys

build_apps = ["carbonstructure", "insert"]

ana_apps = ["debyer", "density", "diporder", "epsilon_bulk", "saxs"]

apps = build_apps + ana_apps
apps.sort()

parser = argparse.ArgumentParser(description="""
    A collection of scripts to analyse and build systems for molecular dynamics simulations.""",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("program", type=str, help="Program to start",
                    choices=apps)


def main():

    try:
        if sys.argv[1] in build_apps:
            exec("import build.{}".format(sys.argv[1]))
            eval("build.{}.main()".format(sys.argv[1]))
        elif sys.argv[1] in ana_apps:
            exec("import ana.{}".format(sys.argv[1]))
            eval("ana.{}.main()".format(sys.argv[1]))
        else:
            parser.parse_args()
    except IndexError:
        parser.parse_args()


if __name__ == "__main__":
    main()
