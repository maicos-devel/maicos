#!/usr/bin/env python
# coding: utf-8

from __future__ import division, print_function, absolute_import

import argparse
import copy
import getopt
import os
import sys

import MDAnalysis
import numpy as np

import moleculeinsertion

from .. import sharePath, initilize_parser

parser = initilize_parser()
parser.description="""Inserts Nw watermolecules in a given
z distance."""
parser.add_argument('-cp', '--refstructure', type=str, required=True,
                    default="protein.gro", help='Structure file: gro, pdb, ...')
parser.add_argument('-cs', '--addstructure', type=str,
                    default=os.path.join(sharePath,'watermolecule.gro'), help='Structure file for added molecules: gro, pdb, ...')
parser.add_argument('-o', '--output', type=str,
                    default='out.gro', help='Output file')
parser.add_argument('-zmin', '--zmin', type=float,
                    default=0, help='Minimal z coordinate for insertion.')
parser.add_argument('-zmax', '--zmax', type=float,
                    default=None, help='Maximal z coordinate for insertion. If None box dimensions are taken.')
parser.add_argument('-Nw', '--Nw', type=int,
                    default=1, help='Number of molecules to insert.')
parser.add_argument('-d', '--dist', type=float,
                    default=1.25, help='Minimal distance [Å] between two molecules.')


def main(firstarg=2):

    args = parser.parse_args(args=sys.argv[firstarg:])

    u = MDAnalysis.Universe(args.refstructure)

    if args.zmax == None:
        args.zmax = u.dimensions[2]

    print("Inserting", args.Nw, "water molecules bewteen z=",
          args.zmin, "and z=", args.zmax)

    water = MDAnalysis.Universe(args.addstructure)
    for i in range(args.Nw):
        u = moleculeinsertion.box(
            u, water, zmax=args.zmax, zmin=args.zmin, distance=args.dist)
        print("\r{} of {} watermolecules placed.".format(i, args.Nw), end=" ")
        sys.stdout.flush()
        if (i + 1) % 250 == 0:
            u.atoms.write(args.output)

    u.atoms.write(args.output)

    print(" ")


if __name__ == "__main__":
    main(firstarg=1)
