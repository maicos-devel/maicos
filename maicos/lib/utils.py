#!/usr/bin/env python3
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
#
# Copyright (c) 2020 Authors and contributors
# (see the file AUTHORS for the full list of names)
#
# Released under the GNU Public Licence, v2 or any higher version
# SPDX-License-Identifier: GPL-2.0-or-later

import os
import sys
import re
import warnings

import numpy as np


def repairMolecules(selection):
    """Repairs molecules that are broken due to peridodic boundaries.
    To this end the center of mass is reset into the central box.
    CAVE: Only works with small (< half box) molecules."""

    warnings.warn(
        "repairMolecules is deprecated, use AtomGroup.unwrap() from MDAnalysis instead.",
        category=DeprecationWarning)

    # we repair each moleculetype individually for performance reasons
    for seg in selection.segments:
        atomsPerMolecule = seg.atoms.n_atoms // seg.atoms.n_residues

        # Make molecules whole, use first atom as reference
        distToFirst = np.empty((seg.atoms.positions.shape))
        for i in range(atomsPerMolecule):
            distToFirst[i::atomsPerMolecule] = seg.atoms.positions[i::atomsPerMolecule] - \
                seg.atoms.positions[0::atomsPerMolecule]
        seg.atoms.positions -= (
            np.abs(distToFirst) > selection.dimensions[:3] /
            2.) * selection.dimensions[:3] * np.sign(distToFirst)

        # Calculate the centers of the objects ( i.e. Molecules )
        masspos = (seg.atoms.positions *
                   seg.atoms.masses[:, np.newaxis]).reshape(
                       (seg.atoms.n_atoms // atomsPerMolecule,
                        atomsPerMolecule, 3))
        # all molecules should have same mass
        centers = np.sum(masspos.T, axis=1).T / \
            seg.atoms.masses[:atomsPerMolecule].sum()

        # now shift them back into the primary simulation cell
        seg.atoms.positions += np.repeat(
            (centers % selection.dimensions[:3]) - centers,
            atomsPerMolecule,
            axis=0)


def check_compound(AtomGroup):
    """Checks if compound 'molecules' exists. If not it will
    fallback to 'fragments' or 'residues'.
    """
    if hasattr(AtomGroup, "molnums"):
        return "molecules"
    elif hasattr(AtomGroup, "fragments"):
        warnings.warn("Cannot use 'molecules'. Falling back to 'fragments'")
        return "fragments"
    else:
        warnings.warn("Cannot use 'molecules'. Falling back to 'residues'")
        return "residues"


def get_cli_input():
    """Returns a proper fomatted string of the command line input"""
    program_name = os.path.basename(sys.argv[0])
    return "Command line was: {} {}".format(program_name,
                                            " ".join(sys.argv[1:]))


def savetxt(fname, X, header='', fsuffix=".dat", **kwargs):
    """An extension of the numpy savetxt function.
    Adds the command line input to the header and checks for a doubled defined
    filesuffix."""
    header = "{}\n{}".format(get_cli_input(), header)
    fname = "{}{}".format(fname, (not fname.endswith(fsuffix)) * fsuffix)
    np.savetxt(fname, X, header=header, **kwargs)


def atomgroup_header(AtomGroup):
    """Returns a string containing infos about the AtmGroup containing
    the total number of atoms, the including residues and the number of residues.
    Useful for writing output file headers."""

    unq_res, n_unq_res = np.unique(AtomGroup.residues.resnames,
                                   return_counts=True)
    return "{} atom(s): {}".format(
        AtomGroup.n_atoms,
        ", ".join("{} {}".format(*i)
                  for i in np.vstack([n_unq_res, unq_res]).T))


def fill_template(template, vars, s="<", e=">"):
    """
    Search and replace tool for filling template files.
    Replaces text bounded by the delimiters `s` and `e`
    with values found in the lookup dictionary `vars`.
    """
    exp = s + "\w*" + e
    matches = re.findall(exp, template)
    for m in matches:
        key = m[1:-1]
        template = template.replace(m, str(vars.get(key, m)))
    return template

def get_cli_input():
    """Returns a proper fomatted string of the command line input"""
    program_name = os.path.basename(sys.argv[0])
    # Add additional quotes for connected arguments.
    arguments = ['"{}"'.format(arg) if " " in arg else arg for arg in sys.argv[1:]]
    return "Command line was: {} {}".format(program_name, " ".join(arguments))


def savetxt(fname, X, header='', fsuffix=".dat", **kwargs):
    """An extension of the numpy savetxt function.
    Adds the command line input to the header and checks for a doubled defined
    filesuffix."""
    header = "{}\n{}".format(get_cli_input(), header)
    fname = "{}{}".format(fname, (not fname.endswith(fsuffix)) * fsuffix)
    np.savetxt(fname, X, header=header, **kwargs)


def atomgroup_header(AtomGroup):
    """Returns a string containing infos about the AtmGroup containing
    the total number of atoms, the including residues and the number of residues.
    Useful for writing output file headers."""

    unq_res, n_unq_res = np.unique(AtomGroup.residues.resnames,
                                   return_counts=True)
    return "{} atom(s): {}".format(
        AtomGroup.n_atoms, ", ".join(
            "{} {}".format(*i) for i in np.vstack([n_unq_res, unq_res]).T))

def save_path(prefix=""):
    """Returns a formatted output location for a given file prefix."""
    if prefix != "" and prefix[-1] != "/":
        prefix += "_"
    output = prefix if os.path.dirname(prefix) else os.path.join(
        os.getcwd(), prefix)
    if not os.path.exists(os.path.dirname(output)):
        os.makedirs(os.path.dirname(prefix))

    return output


def zero_pad(arr, length=None, add=None):
    """
    Pads an array to a given length with zeros,
    or optionals adds a fixed number of zero to the array end.
    """
    if length is not None:
        return np.append(arr, np.zeros(np.maximum(0, length - len(arr))))
    elif add is not None:
        return np.append(arr, np.zeros(add))
    else:
        raise ValueError(
            "Invalid arguments specified for length of zero padding.")


def bin_data(arr, nbins, after=1):
    """
    Averages array values in bins for easier plotting.
    Note: "bins" array should contain the INDEX (integer) where that bin begins
    """
    # Determine indices to bin at
    bins = np.logspace(
        np.log(after) / np.log(10),
        np.log(len(arr)) / np.log(10), nbins - after + 1).astype(int)
    bins = np.unique(np.append(np.arange(after), bins))

    # Do the averaging
    avg = np.zeros(len(bins), dtype=arr.dtype)  # Average of data
    count = np.zeros(len(bins), dtype=int)

    ic = -1
    for i in range(0, len(arr)):
        if i in bins:
            ic += 1  # index for new average
        avg[ic] += arr[i]
        count[ic] += 1

    return avg / count
