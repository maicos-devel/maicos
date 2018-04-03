#!/usr/bin/env python
# coding: utf-8

from __future__ import print_function, absolute_import

import os
import sys

import numpy as np

from .. import initilize_parser

parser = initilize_parser()
parser.description = """Script for modifying topology files for linear periodic peptides.
For the final peptide sequence of '-A-B-C-D-A-B . . . B-C-D-'
the input topolgy must contain THREE extra
residues on each end of the peptide: '(b-c-d-)A-B-C-D-A-B ... B-C-D-(a-b-c)'.
The peptide in the numbering in the topolgy must start with 1!"""

parser.add_argument('-p', '--topology', type=str, required=True,
                    default="topol.top", help='topolgy file: top, itp')
parser.add_argument('-l', '--length', type=int, required=True,
                    help='Length of the final peptide.')
parser.add_argument('-o', '--output', type=str,
                    default='topol_per', help='Output topolgy')
parser.add_argument('-v', dest='verbose',
                    action='store_true', help='Be loud and noisy.')

# currently not used


def remove_comments(datastr):
    # TODO check for spaces, etc
    newdatastr = []
    for str in datastr:
        if str[0] != ";":
            newdatastr.append(str)

    return newdatastr


def main(firstarg=2):
    args = parser.parse_args(args=sys.argv[firstarg:])

    with open(args.topology, 'r') as f:
        lines = f.readlines()

    # what to keep, what to replace? ---
    keepres_orig = np.arange(4, args.length + 4)
    replaceres_orig = {3: args.length + 3, 2: args.length +
                       2, args.length + 4: 4, args.length + 5: 5}

    keepres_orig = np.arange(4, 24)
    replaceres_orig = {3: 23, 2: 22, 24: 4, 25: 5}

    substractres = min(keepres_orig) - 1

    nvalues = {'moleculetype': [],
               'atoms': [0, 5],
               'bonds': range(0, 2),
               'pairs': range(0, 2),
               'angles': range(0, 3),
               'dihedrals': range(0, 4),
               'cmap': range(0, 5),
               'position_restraints': [],
               'system': [],
               'molecules': []}

    rjusts = {'moleculetype': [1, 8],
              'atoms': [6, 11, 7, 7, 7, 7, 11, 11],
              'bonds': [5, 6, 6, 11, 11, 11, 11],
              'pairs': [5, 6, 6, 11, 11, 11, 11],
              'angles': [5, 6, 6, 6, 11, 11, 11, 11],
              'dihedrals': [5, 6, 6, 6, 6, 11, 11, 11, 11, 11, 11, 11],
              'cmap': [5, 6, 6, 6, 6, 6, 11, 11, 11, 11, 11, 11]}
           #, 'position_restraints':[4,5,11,11,11]}

    # atomnumberpos, resnumberpos in [ atoms ]
    arnvals = [0, 2]

    header = "; --- This file was generated with mdtools pertop --- \n"
    header += ";\n; The result comes without any warranty!\n;\n"
    header += ";" + 62 * "-" + "\n"

    # some checks

    for kr in keepres_orig:
        if kr in replaceres_orig:
            raise RuntimeError("Error: kept atoms are also replaced.")

    # the skript internally reduces all residue numbers by 1
    # maybe better use additional namelist

    # reduce all numbers by 1
    keepres = []
    for i in keepres_orig:
        keepres.append(i - 1)

    replaceres = {}
    for i in replaceres_orig:
        replaceres.update({(i - 1): (replaceres_orig[i] - 1)})

    # look for keywords
    keys = []
    keypos = []
    keycommentlines = []

    # Add header
    for j in range(0, len(lines)):
        if lines[j][0] == "[":
            break
        else:
            header += lines[j]

    for i in range(j - 1, len(lines)):

        if lines[i][0] == "[":
            sline = lines[i].strip().split()
            if len(sline) < 2 or sline[2] != "]":
                raise RuntimeError("Invalid input file format.")
            keys.append(sline[1])
            keypos.append(i)
            if (lines[i + 1][0] == ";" or lines[i + 1][0] == "#"):
                keycommentlines.append(lines[i + 1])
            else:
                keycommentlines.append("")

    keypos.append(len(lines) + 1)

    # group the residues and find atoms
    if keys[1] != "atoms":
        raise RuntimeError("Key atoms is at wrong position.")

    # ---------------------------------------
    # find residues and collect atom numbers
    datastr = lines[(keypos[1] + 1):keypos[2]]

    nresidue = []
    residues = []

    ncurrentres = -1
    currentres = []

    for line in datastr:
        if not (line[0] == ';' or line[0] == '#'):
            sline = line.strip().split()
            if sline != []:
                dummy = []
                for val in arnvals:
                    dummy.append(sline[val])

                    dummy1 = np.array(dummy)
                    nline = dummy1.astype(np.int)

                    if nline != []:
                        if nline[1] != ncurrentres:
                            # append last one
                            if currentres != []:
                                residues.append(currentres)
                            currentres = []

                            # set up new one
                            ncurrentres = nline[1]
                            if ncurrentres in nresidue:
                                raise RuntimeError(
                                    "Currently residues atoms must be next to each other")
    # WARNING: we start with zero in this skript!!
                            nresidue.append(ncurrentres - 1)
                        currentres.append(nline[0])

    residues.append(currentres)

    # ---------------------------
    # ok now nresidue knows the residue-numbers and residues the corresponding atoms
    # len(nresidues) residues
    if nresidue != range(0, len(nresidue)):
        raise RuntimeError("Currently we need the correct residue numbering!")
    # --- HERE

    # prepare atoms which are kept

    keep = []
    for ires in keepres:
        keep.extend(residues[ires])

    # now prepare exchange list

    # first update replaceres
    for ires in nresidue:
        if (ires in keepres) and (ires not in replaceres):
            replaceres.update({ires: ires})
    # done

    replace = {}
    for ires in replaceres:
        # first check length
        if len(residues[ires]) != len(residues[replaceres[ires]]):
            raise RuntimeError("Residues to replace do not match.")

        for jatom in range(0, len(residues[ires])):
            replace.update(
                {residues[ires][jatom]: residues[replaceres[ires]][jatom]})

    # now add renumbering
    firstatom = min(keep)
    # print(firstatom)

    for iatom in replace:
        replace[iatom] = replace[iatom] - firstatom + 1

    # in theory this should work now ;-)

    # get the new data
    newlines = [header]
    for i in range(0, len(keys)):
        if args.verbose:
            print("processing [ ", keys[i], " ] ...")
        datastr = lines[(keypos[i] + 1):keypos[i + 1]]

        # now process datastr
        newdatastr = []
        for line in datastr:
            if nvalues[keys[i]] == [] or line[0] == ';' or line[0] == '#':
                newdatastr.append(line)
            else:
                sline = line.strip().split()
                if sline == []:
                    newdatastr.append(line)
                else:
                    # ok this is the place where numbers will actually be changed!
                    # residue number will be changed here (improve this maybe ..)
                    if keys[i] == "atoms":
                        sline[arnvals[1]] = str(
                            int(sline[arnvals[1]]) - substractres)

                    dummy = []
                    for val in nvalues[keys[i]]:
                        dummy.append(sline[val])
                    dummy1 = np.array(dummy)
                    nline = dummy1.astype(np.int)
                    if nline != [] and nline[0] in keep:
                        tmpline = []
                        for j in range(0, len(nline)):
                            if nline[j] not in replace:
                                raise RuntimeError(
                                    "There is an atom which is neither to be kept nor to be replaced. {}".format(j))
                            tmpline.append(replace[nline[j]])
                            # check if the new number
                        #
                        newline = []
                        for val in range(0, len(nline)):
                            vpos = nvalues[keys[i]][val]
                            sline[vpos] = str(tmpline[val])
                        #

                        # now print approrpiate string
                        newline = ""
                        for j in range(0, min(len(rjusts[keys[i]]), len(sline))):
                            newline = newline + \
                                sline[j].rjust(rjusts[keys[i]][j])
                        newline = newline + "\n"
                        newdatastr.append(newline)

        newlines.append("[ " + keys[i] + " ]\n")
        newlines.extend(newdatastr)

    # finally write a file containing the information about atoms to replace
    topend = os.path.splitext(args.topology)[1]
    with open(args.output + topend, 'w') as f:
        for line in newlines:
            f.write(line)


if __name__ == "__main__":
    main(firstarg=1)
