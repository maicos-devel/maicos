#!/usr/bin/env python
# coding: utf-8

# Uses algorithms of build Cstrust1_1 from Andrea Minoia (http://chembytes.wikidot.com)

# ===================================================================================================
# IMPORTS
# ===================================================================================================

from __future__ import absolute_import, division, print_function

import argparse
import os
import shutil
import subprocess
import sys

import MDAnalysis
import numpy as np

from .. import initilize_parser, sharePath
from ..utils import cyzone

# ===================================================================================================
# INPUT OPTIONS
# ===================================================================================================
parser = initilize_parser()
parser.description = """Builds an armchair CNT with
or withput two HOPG at the end. Or a single HOPG sheet."""
parser.add_argument('-s', '--structure', type=str, required=True,
                    default=None, help='Structure to build type.', choices=["armcnt", "hopg", "hopgcnt"])
parser.add_argument('-i', '--index', type=int, required=True,
                    default=None, help='number of unitcells for the CNT/HOPG.')
parser.add_argument('-l', '--length', type=float, required=True,
                    default=None, help='length of the structure [Ansgtroem]')
parser.add_argument('-w', '--fillwater', action='store_true',
                    help='If applied the structure will be filled with water')
parser.add_argument('-d', '--density', type=float,
                    default=None,   help='The water number density per nm^3. If no input density will be estimated.')
parser.add_argument('-x', '--gromacs', action='store_false',
                    help='Use "gmx solvate" for solvation. Gromacs installation needed.')
parser.add_argument('-o', dest='output', type=str,
                    default='.', help='Output directory.')

ccbond = 1.42  # C-C bond length in Angstrom
pbcx = False
pbcy = False
FNULL = open(os.devnull, 'w')


# ===================================================================================================
# FUNCTIONS
# ===================================================================================================

def armcnt(n, l):
    '''build armchair carbon nanotube'''
    atc = []
    circ1 = []
    circ2 = []
    dx = ccbond * np.cos(120 / 2 * (np.pi / 180))
    dy = ccbond * np.sin(120 / 2 * (np.pi / 180))
    radius = (n * (2 * dx + ccbond) + n * ccbond) / (2 * np.pi)
    ycoord = +dy
    natoms = 2 * n
    # create circumferences
    for i in range(n):
        circ1.append(2 * dx + ccbond)
        circ1.append(ccbond)
        circ2.append(ccbond)
        circ2.append(2 * dx + ccbond)
    # adjust the circumferences
    circ1.insert(0, 0.0)
    circ1.pop()
    circ2.insert(0, dx)
    circ2.pop()
    # Build CNT
    while ycoord > -l:
        ycoord -= dy
        arc = 0.0
        for i in range(natoms):
            tmpcoords = ['C']
            arc += circ1[i]
            theta = arc / radius
            tmpcoords.append(radius * np.cos(theta))
            tmpcoords.append(radius * np.sin(theta))
            tmpcoords.append(ycoord)
            atc.append(tmpcoords)
        ycoord -= dy
        arc = 0.0
        for i in range(natoms):
            tmpcoords = ['C']
            arc += circ2[i]
            theta = arc / radius
            tmpcoords.append(radius * np.cos(theta))
            tmpcoords.append(radius * np.sin(theta))
            tmpcoords.append(ycoord)
            atc.append(tmpcoords)

    return atc, len(atc), radius, abs(l + dy)


def graphite(x, y, zshift=0):
    ''' generate single square sheet of graphite HOPG'''
    atc = []
    dx = ccbond * np.cos(120 / 2 * (np.pi / 180))
    dy = ccbond * np.sin(120 / 2 * (np.pi / 180))
    ycoord = +dy
    xcoords1 = []
    xcoord = 0.00
    xcoords1.append(xcoord)
    # build 1st row for X
    while xcoord <= x:
        xcoord += ccbond + 2 * dx
        xcoords1.append(xcoord)
        xcoord += ccbond
        xcoords1.append(xcoord)
    xcoords1.pop()  # remove last element, i.e. the bond exceeding the size
    # build 2nd row for X
    xcoord = dx
    xcoords2 = []
    xcoords2.append(xcoord)
    while xcoord <= x + dx:
        xcoord += ccbond
        xcoords2.append(xcoord)
        xcoord += ccbond + 2 * dx
        xcoords2.append(xcoord)
    xcoords2.pop()  # remove last element, i.e. the bond exceeding the size
    while ycoord > -y:
        ycoord -= dy
        for coord in xcoords1:
            tmpcoords = ['C']
            tmpcoords.append(coord)
            tmpcoords.append(ycoord + y)
            tmpcoords.append(-zshift)
            atc.append(tmpcoords)
        ycoord -= dy
        for coord in xcoords2:
            tmpcoords = ['C']
            tmpcoords.append(coord)
            tmpcoords.append(ycoord + y)
            tmpcoords.append(-zshift)
            atc.append(tmpcoords)

    a_pbc = atc[len(xcoords1) - 1][1] + ccbond
    b_pbc = abs(ycoord) + dy

    return atc, len(atc), a_pbc, b_pbc


def write_gro(file, data, xwidth, ywidth, zwidth):
    '''
    Write a gromacs gro file.
        Input Variables:
            file: output file (type: file)
            data: list of lists. Each list contains:
                    1) atom name
                2,3,4) X-, Y- and Z-coordinates
            pbc1/pbc2: periodic lengths
        Variables:
            line: store each list of data (type: list)
            outline: string containing a np.single line to be written in file (type: string)
    '''
    file.write("Generated by buildNanotube.py\n " + str(len(data)) + "\n")
    for index, line in enumerate(data):
        outline = "%5i%-5s%5s%5s%8.3f%8.3f%8.3f" % (1, "CNT", line[0], str(index + 1)[-5:], float(line[1]) / 10.0, float(line[2] / 10.0)                                                    # outline="%5i%-5s%5s%5i%8.3f%8.3f%8.3f" % (1,"CNT",line[0],index+1,float(line[1])/10.0,float(line[2]/10.0)\
                                                    , float(line[3] / 10.0))
        file.write(outline + "\n")
    # dividing by 10 to go from A to nm
    outline = "  %.5f  %5f %5f" % (xwidth / 10, ywidth / 10, zwidth / 10)
    file.write(outline + "\n")


def write_carbon_itp(file, nCNTatoms):
    file.write('[ moleculetype ]\n')
    file.write('; molname nrexcl\n')
    file.write('CNT   1\n\n')

    file.write('[ atoms ]\n')
    file.write(';   nr   type  resnr residue  atom   cgnr     charge       mass\n')
    index = 1
    while index <= nCNTatoms:
        sindex = str(index)
        if index <= 9:
            outline = '     ' + sindex + '    C     1     CNT      C         ' + \
                sindex + '     0.0    12.0110\n'
        elif index <= 99:
            outline = '    ' + sindex + '    C     1     CNT      C        ' + \
                sindex + '     0.0    12.0110\n'
        elif index <= 999:
            outline = '   ' + sindex + '    C     1     CNT      C       ' + \
                sindex + '     0.0    12.0110\n'
        else:
            outline = '  ' + sindex + '    C     1     CNT      C      ' + \
                sindex + '     0.0    12.0110\n'

        file.write(outline)
        index += 1

def gmxsolvate(radius, length, reservoir, nCatoms, xshift, yshift, scale=0.57):

    subprocess.call("gmx solvate -cp out.gro -cs spc216.gro -o out.gro -scale " +
                    str(scale), shell=True, stdout=FNULL, stderr=subprocess.STDOUT)

    u = MDAnalysis.Universe('out.gro')

    CNT = u.atoms[:nCatoms]
    oxygens = u.select_atoms("name OW")

    if reservoir != 0:
        O_selection = oxygens[np.where(
            np.logical_or(
                (((oxygens.positions[:, 0] - xshift)**2 +
                  (oxygens.positions[:, 1] - yshift)**2) < radius**2),
                (np.absolute(oxygens.positions[:, 2] - (reservoir + length / 2)) >= length / 2)))[0]]
    else:
        O_selection = oxygens[np.where(
            (((oxygens.positions[:, 0] - xshift)**2 +
              (oxygens.positions[:, 1] - yshift)**2) < radius**2),
        )[0]]

    selectionarray = np.sort(np.hstack(
        (O_selection.atoms.indices, O_selection.atoms.indices + 1, O_selection.atoms.indices + 2)))

    System = CNT + u.atoms[selectionarray]
    System.atoms.write('out.gro')

    return O_selection.n_atoms


def deletesolvate(nCatoms, nSOLresidues):
    u = MDAnalysis.Universe('out.gro')

    CNT = u.atoms[:nCatoms]
    oxygens = u.select_atoms("name OW")

    randomoxygens = np.random.choice(
        range(oxygens.n_atoms - 1), nSOLresidues, replace=False) * 3
    selectionarray = np.sort(
        np.hstack((randomoxygens, randomoxygens + 1, randomoxygens + 2))) + nCatoms

    System = CNT + u.atoms[selectionarray]
    System.atoms.write('out.gro')

    return System


def density_fit(radius):
    # calculate density from radius, Parameters determined from fit. See thesis
    r0 = 6.05972
    eta0 = 69.3043
    eta1 = 0.942101
    return eta1 * (32.9 - eta0 / (radius / r0 + 1)**2)


# ===================================================================================================
# MAIN
# ===================================================================================================

def main(firstarg=2, DEBUG=False):

    args = parser.parse_args(args=sys.argv[firstarg:])

    print("Bulding C structure...", end=" ")

    if args.structure == "hopg":
        dx = ccbond * np.cos(120 / 2 * (np.pi / 180))
        width = (args.index * (2 * dx + ccbond) + args.index * ccbond)
        length = args.length
        graphitecoords, nHOPGatoms, xwidth, ywidth = graphite(width, width)
    elif args.structure == "armcnt":
        armcntcoords, nCNTatoms, radius, length = armcnt(
            args.index, args.length)
    elif args.structure == "hopgcnt":
        armcntcoords, nCNTatoms, radius, length = armcnt(
            args.index, args.length)

    if args.structure == "hopg":
        reservoir = 0
    else:
        # if structure is to small peridic images will see each other therefore
        # shift whole cnt at least by 0.75 nm
        if 3 * radius <= 20:
            centercoord = 20
        elif radius >= 20:
            centercoord = radius + 10
        else:
            centercoord = 2 * radius

    if args.structure == "armcnt":
        xwidth = ywidth = 2 * centercoord
        reservoir = 0

    elif args.structure == "hopgcnt":
        graphitecoords1, nHOPGatoms1, xwidth, ywidth = graphite(
            2 * centercoord, 2 * centercoord, length)
        graphitecoords2, nHOPGatoms2 = graphite(
            2 * centercoord, 2 * centercoord)[0:2]

        graphitecoords = graphitecoords1 + graphitecoords2
        nHOPGatoms = nHOPGatoms1 + nHOPGatoms2
        reservoir = 20

    if args.structure != "hopg":
        # shift cnt in the center of the box
        for i in range(len(armcntcoords)):
            armcntcoords[i][1] += xwidth / 2
            armcntcoords[i][2] += ywidth / 2

    if args.structure == "armcnt":
        nCatoms = nCNTatoms
        coords = armcntcoords
    elif args.structure == "hopgcnt":
        # removing atoms which are blocking the cnt.
        removedatoms = 0
        i = 0
        while i < len(graphitecoords):
            if (graphitecoords[i][1] - xwidth / 2)**2 + (graphitecoords[i][2] - ywidth / 2)**2 \
                    <= radius**2:
                del graphitecoords[i]
                removedatoms += 1
            else:
                i += 1

        nCatoms = nCNTatoms + nHOPGatoms - removedatoms
        coords = armcntcoords + graphitecoords

    if args.structure == "hopg":
        nCatoms = nHOPGatoms
        coords = graphitecoords
    else:
        # shifting whole structure
        for i in range(len(coords)):
            coords[i][3] += length + reservoir

    print("Done!")
    sys.stdout.flush()

    os.chdir(args.output)

    OUT = open("out.gro", 'w')
    write_gro(OUT, coords, xwidth, ywidth, (length + 2 * reservoir))
    OUT.close()

    if args.fillwater == True:
        if args.structure != "hopg":
            if args.density == None:
                args.density = density_fit(radius)

            if args.gromacs == True:
                print("Filling structure with water using gmx solvate...", end=" ")
                sys.stdout.flush()

                if args.structure == "armcnt":
                    structure = MDAnalysis.Universe('out.gro')
                    cog = structure.atoms.center_of_geometry()
                    InsertionShift = np.array((cog[0], cog[1], 0))
                    nSOLresidues = int(
                        round(args.density * length * np.pi * radius**2 / 1000))

                    GMXnSOLresidues = gmxsolvate(
                        radius, length, reservoir, nCatoms, xwidth / 2, ywidth / 2, scale=0.37)
                    print("Done!")
                    if GMXnSOLresidues >= nSOLresidues:
                        print("Delete superfluous atoms...", end=" ")
                        sys.stdout.flush()
                        u = deletesolvate(nCatoms, nSOLresidues)
                        print("Done!")
                    else:
                        print(
                            "Filling structure with water up to the given density and writing a backup file every 500 steps...")

                        water = MDAnalysis.Universe(
                            os.path.join(sharePath, "watermolecule.gro"))
                        u = MDAnalysis.Universe('out.gro')

                        for i in range(nSOLresidues - GMXnSOLresidues):
                            u = cyzone(
                                u, water, radius, InsertionShift)
                            print("%i of %i watermolecules placed." % (
                                i + 1 + GMXnSOLresidues, nSOLresidues), end="\r")
                            sys.stdout.flush()
                            if (i + 1) % 500 == 0:
                                u.atoms.write('out.gro')
                        print("\nDone!")
                elif args.structure == "hopgcnt":
                    nSOLresidues = gmxsolvate(
                        radius, length, reservoir, nCatoms, xwidth / 2, ywidth / 2)
                    u = MDAnalysis.Universe('out.gro')
                    print("Done!")

            else:
                print(
                    "Filling structure with water and writing a backup file every 500 steps...")
                water = MDAnalysis.Universe(
                    os.path.join(sharePath, "watermolecule.gro"))
                u = MDAnalysis.Universe('out.gro')

                if args.structure == "armcnt":
                    cog = u.atoms.center_of_geometry()
                    InsertionShift = np.array((cog[0], cog[1], 0))
                    nSOLresidues = int(
                        round(args.density * length * np.pi * radius**2 / 1000))
                    for i in range(nSOLresidues):
                        u = cyzone(
                            u, water, radius, InsertionShift)
                        print("{} of {} watermolecules placed.".format(
                            i + 1, nSOLresidues), end="\r")
                        sys.stdout.flush()
                        if (i + 1) % 500 == 0:
                            u.atoms.write('out.gro')

                elif args.structure == "hopgcnt":

                    nSOLresiduesCNT = int(
                        round(args.density * length * np.pi * radius**2 / 1000))
                    nSOLresiduesbulk = int(
                        round(32.9 * reservoir * u.dimensions[0] * u.dimensions[1] / 1000))
                    nSOLresidues = nSOLresiduesCNT + 2 * nSOLresiduesbulk
                    InsertionShift = np.array(
                        (u.dimensions[0] / 2, u.dimensions[1] / 2, 0))

                    for i in range(nSOLresiduesCNT):
                        u = cyzone(
                            u, water, radius, InsertionShift, zmin=reservoir, zmax=length + reservoir)
                        print("{} of {} watermolecules placed.".format(
                            i + 1, nSOLresidues), end="\r")
                        sys.stdout.flush()
                        if (i + 1) % 500 == 0:
                            u.atoms.write('out.gro')

                    for i in range(nSOLresiduesbulk):
                        u = moleculeinsertion.box(u, water, zmax=reservoir)
                        print("{} of {} watermolecules placed.".format(
                            2 * (i + 1) - 1 + nSOLresiduesCNT, nSOLresidues), end="\r")
                        sys.stdout.flush()
                        u = moleculeinsertion.box(
                            u, water, zmin=reservoir + length)
                        print("{} of {} watermolecules placed.".format(
                            2 * (i + 1) + nSOLresiduesCNT, nSOLresidues), end="\r")
                        sys.stdout.flush()
                        if (i + 1) % 250 == 0:
                            u.atoms.write('out.gro')

            u.atoms.write('out.gro')
        else:
            scale = 0.57
            if args.density != None:
                scale *= (32.9/args.density)**0.33

            subprocess.call(
                "gmx solvate -cp out.gro -cs spc216.gro -scale {:.3f} -o out.gro".format(scale),
                shell=True, stdout=FNULL, stderr=subprocess.STDOUT)
            u = MDAnalysis.Universe('out.gro')
            oxygens = u.select_atoms("name OW")
            nSOLresidues = oxygens.n_atoms
    else:
        nSOLresidues = 0

    print("Creating gromacs files...", end=" ")
    sys.stdout.flush()

    OUT = open('carbon.itp', 'w')
    write_carbon_itp(OUT, nCatoms)
    OUT.close()

    with open('parameters.txt', 'w') as OUT:
        OUT.write(str(args.index) + '\t\t#n\n')
        if args.structure != "hopg":
            OUT.write(str(radius) + '\t#cnt radius (Angstrom)\n')
        OUT.write(str(length) + '\t#cnt length (Angstrom)\n')
        if args.structure != "hopg":
            OUT.write(str(centercoord) +
                      '\t#center coordinate (x,y) (Angstrom)\n')
        OUT.write(str(xwidth) + '\t#box x-width (Angstrom)\n')
        OUT.write(str(ywidth) + '\t#box y-width (Angstrom)\n')
        if args.structure != "hopg":
            OUT.write(str(length + 2 * reservoir) +
                      '\t#box z-width (Angstrom)\n')
        if args.fillwater == True:
            OUT.write(str(args.density) +
                      '\t\t#waterdensity (Molecules/nm^-3)\n')

    print("All parameters are written to parameters.txt")

    print("Finished!")


if __name__ == "__main__":
    main(firstarg=1)
