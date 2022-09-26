#!/usr/bin/env python3
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
#
# Copyright (c) 2022 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the GNU Public Licence, v3 or any higher version
# SPDX-License-Identifier: GPL-3.0-or-later
"""Small helper and utilities functions that don't fit anywhere else."""

import functools
import os
import sys
import warnings
from typing import Callable

import numpy as np


_share_path = os.path.realpath(os.path.join(os.path.dirname(__file__),
                                            "..", "share"))


def get_compound(atomgroup, return_index=False):
    """Returns the highest order topology attribute.

    The order is "molecules", "fragments", "residues". If the topology contains
    none of those attributes, an AttributeError is raised. Optionally, the
    indices of the attribute as given by `molnums`, `fragindices` or
    `resindices` respectivly are also returned.

    Parameters
    ----------
    atomgroup : MDAnalysis.core.groups.AtomGroup
        atomgroup taken for weight calculation
    return_index : bool, optional
        If True, also return the indices the corresponding topology attribute.

    Returns
    -------
    compound: string
        Name of the topology attribute.
    index: ndarray, optional
        The indices of the topology attribute.

    Raises
    ------
    AttributeError
        `atomgroup` is missing any connection information"
    """
    if hasattr(atomgroup, "molnums"):
        compound = "molecules"
        indices = atomgroup.atoms.molnums
    elif hasattr(atomgroup, "fragments"):
        warnings.warn("Cannot use 'molecules'. Falling back to 'fragments'")
        compound = "fragments"
        indices = atomgroup.atoms.fragindices
    elif hasattr(atomgroup, "residues"):
        warnings.warn("Cannot use 'fragments'. Falling back to 'residues'")
        compound = "residues"
        indices = atomgroup.atoms.resindices
    else:
        raise AttributeError(
            "Missing any connection information in `atomgroup`.")
    if return_index:
        return compound, indices
    else:
        return compound


def get_cli_input():
    """Return a proper formatted string of the command line input."""
    program_name = os.path.basename(sys.argv[0])
    # Add additional quotes for connected arguments.
    arguments = ['"{}"'.format(arg)
                 if " " in arg else arg for arg in sys.argv[1:]]
    return "{} {}".format(program_name, " ".join(arguments))


def atomgroup_header(AtomGroup):
    """Return a string containing infos about the AtomGroup.

    Infos include the total number of atoms, the including
    residues and the number of residues. Useful for writing
    output file headers.
    """
    if not hasattr(AtomGroup, 'types'):
        warnings.warn("AtomGroup does not contain atom types. "
                      "Not writing AtomGroup information to output.")
        return f"{len(AtomGroup.atoms)} unkown particles"
    unique, unique_counts = np.unique(AtomGroup.types,
                                      return_counts=True)
    return " & ".join(
        "{} {}".format(*i) for i in np.vstack([unique, unique_counts]).T)


def bin(a, bins):
    """Average array values in bins for easier plotting.

    Note: "bins" array should contain the INDEX (integer)
    where that bin begins
    """
    if np.iscomplex(a).any():
        avg = np.zeros(len(bins), dtype=complex)  # average of data
    else:
        avg = np.zeros(len(bins))

    count = np.zeros(len(bins), dtype=int)
    ic = -1

    for i in range(0, len(a)):
        if i in bins:
            ic += 1  # index for new average
        avg[ic] += a[i]
        count[ic] += 1

    return avg / count


doc_dict = dict(
    ATOMGROUP_PARAMETER="""atomgroup : AtomGroup
        A :class:`~MDAnalysis.core.groups.AtomGroup` for which
        the calculations are performed.""",
    ATOMGROUPS_PARAMETER="""atomgroups : list[AtomGroup]
        a list of :class:`~MDAnalysis.core.groups.AtomGroup` for whiches
        the calculations are performed.""",
    BASE_CLASS_PARAMETERS="""refgroup : AtomGroup
        Reference :class:`~MDAnalysis.core.groups.AtomGroup` used for the
        calculation.

        If refgroup is provided, the calculation is
        performed relative to the center of mass of the AtomGroup.

        If refgroup is `None` the calculations
        are performed relative to the center of the box. If the box
        size is fluctuating with time, the instantaneous center
        of the box is used.
    unwrap : bool
        When `unwrap = True`, molecules that are broken due to the
        periodic boundary conditions are made whole.

        If the input contains molecules that are already whole,
        speed up the calculation by disabling unwrap. To do so,
        use the flag `-no-unwrap` when using MAICoS from the
        command line, or use `unwrap = False` when using MAICoS from
        the Python interpreter.

        Note: Molecules containing virtual sites (e.g. TIP4P water
        models) are not currently supported. In this case, provide
        unwrapped trajectory files directly, and disable unwrap.
        Trajectory can be unwrapped for example using the
        trjconv function of GROMACS.
    concfreq : int,
        When concfreq (for conclude frequency) is larger than 0,
        the conclude function is called and the output files are
        written every number=concfreq frames""",
    PLANAR_CLASS_PARAMETERS="""dim : int,
        Dimension for binning (x=0, y=1, z=2).
    zmin : float,
        Minimal coordinate for evaluation (in Å) with respect to the
        center of mass of the refgroup.

        If zmin=None, all coordinates down to the lower cell boundary
        are taken into account.
    zmax : float,
        Maximal coordinate for evaluation (in Å) with respect to the
        center of mass of the refgroup.

        If `zmax = None`, all coordinates up to the upper cell boundary
        are taken into account.""",
    BIN_WIDTH_PARAMETER="""bin_width : float
        Width of the bins (in Å).""",
    RADIAL_CLASS_PARAMETERS="""rmin : float,
        Minimal r-coordinate relative to the center of mass of the
        refgroup for evaluation (in Å).
    rmax : float,
        Maximal r-coordinate relative to the center of mass of the
        refgroup for evaluation (in Å).

        If rmax=None, the box extension is taken.""",
    SYM_PARAMETER="""sym : bool,
        Symmetrize the profile. Only works in combinations with `refgroup`.""",
    PROFILE_CLASS_PARAMETERS="""grouping : str, {'atoms', 'residues', 'segments', 'molecules', 'fragments'}"""  # noqa
    """
          Atom grouping for the calculations of profiles.

          The possible grouping options are the atom positions (in
          the case where grouping='atoms') or the center of mass of
          the specified grouping unit (in the case where
          grouping='residues', 'segments', 'molecules' or 'fragments').
    binmethod : str, {'cog', 'com', 'coc'}"""
    """
        Method for the position binning.

        The possible options are center of geometry (`'cog'`),
        center of mass (`'com'`), and center of charge (`'coc'`).
    output : str
        Output filename.""",
    PLANAR_CLASS_ATTRIBUTES="""results.bin_pos : numpy.ndarray
        Bin positions (in Å) ranging from `zmin` to `zmax`.""",
    RADIAL_CLASS_ATTRIBUTES="""results.bin_pos : numpy.ndarray
        Bin positions (in Å) ranging from `rmin` to `rmax`.""",
    PROFILE_CLASS_ATTRIBUTES="""results.profile_mean : numpy.ndarray
        Calculated profile's averaged value.
    results.profile_err : numpy.ndarray
        Calculated profile's error."""
    )

# Inherit docstrings
doc_dict["PLANAR_CLASS_PARAMETERS"] = \
    doc_dict["BASE_CLASS_PARAMETERS"] + "\n    " + \
    doc_dict["PLANAR_CLASS_PARAMETERS"] + "\n    " + \
    doc_dict["BIN_WIDTH_PARAMETER"]

doc_dict["CYLINDER_CLASS_PARAMETERS"] = \
    doc_dict["PLANAR_CLASS_PARAMETERS"] + "\n    " + \
    doc_dict["RADIAL_CLASS_PARAMETERS"]

doc_dict["SPHERE_CLASS_PARAMETERS"] = \
    doc_dict["RADIAL_CLASS_PARAMETERS"] + "\n    " + \
    doc_dict["BIN_WIDTH_PARAMETER"]

doc_dict["PROFILE_PLANAR_CLASS_PARAMETERS"] = \
    doc_dict["ATOMGROUPS_PARAMETER"] + "\n    " + \
    doc_dict["PLANAR_CLASS_PARAMETERS"] + "\n    " + \
    doc_dict["SYM_PARAMETER"] + "\n    " + \
    doc_dict["PROFILE_CLASS_PARAMETERS"]

doc_dict["PROFILE_CYLINDER_CLASS_PARAMETERS"] = \
    doc_dict["ATOMGROUPS_PARAMETER"] + "\n    " + \
    doc_dict["CYLINDER_CLASS_PARAMETERS"] + "\n    " + \
    doc_dict["PROFILE_CLASS_PARAMETERS"]

doc_dict["PROFILE_SPHERE_CLASS_PARAMETERS"] = \
    doc_dict["ATOMGROUPS_PARAMETER"] + "\n    " + \
    doc_dict["RADIAL_CLASS_PARAMETERS"] + "\n    " + \
    doc_dict["PROFILE_CLASS_PARAMETERS"]

doc_dict["CYLINDER_CLASS_ATTRIBUTES"] = doc_dict["RADIAL_CLASS_ATTRIBUTES"]
doc_dict["SPHERE_CLASS_ATTRIBUTES"] = doc_dict["RADIAL_CLASS_ATTRIBUTES"]

doc_dict["PROFILE_PLANAR_CLASS_ATTRIBUTES"] = \
    doc_dict["PLANAR_CLASS_ATTRIBUTES"] + "\n    " + \
    doc_dict["PROFILE_CLASS_ATTRIBUTES"]

doc_dict["PROFILE_CYLINDER_CLASS_ATTRIBUTES"] = \
    doc_dict["RADIAL_CLASS_ATTRIBUTES"] + "\n    " + \
    doc_dict["PROFILE_CLASS_ATTRIBUTES"]

doc_dict["PROFILE_SPHERE_CLASS_ATTRIBUTES"] = \
    doc_dict["RADIAL_CLASS_ATTRIBUTES"] + "\n    " + \
    doc_dict["PROFILE_CLASS_ATTRIBUTES"]


def render_docs(func: Callable, doc_dict: dict = doc_dict) -> Callable:
    """Replace all template phrases in the functions docstring.

    Parameters
    ----------
    func : callable
        The callable (function, class) where the phrase old should be replaced.
    doc_dict : str
        The dictionary containing phrase which will be replaced

    Returns
    -------
    Callable
        callable with replaced phrase
    """
    if func.__doc__ is not None:
        for pattern in doc_dict.keys():
            func.__doc__ = func.__doc__.replace(f"${{{pattern}}}",
                                                doc_dict[pattern])
    return func


def charge_neutral(filter):
    """Raise a Warning when AtomGroup is not charge neutral.

    Class Decorator to raise an Error/Warning when AtomGroup in an AnalysisBase
    class is not charge neutral. The behaviour of the warning can be controlled
    with the filter attribute. If the AtomGroup's corresponding universe is
    non-neutral an ValueError is raised.

    Parameters
    ----------
    filter : str
        Filter type to control warning filter Common values are: "error"
        or "default" See `warnings.simplefilter` for more options.
    """
    def inner(original_class):
        def charge_check(function):
            @functools.wraps(function)
            def wrapped(self):
                if hasattr(self, 'atomgroup'):
                    groups = [self.atomgroup]
                else:
                    groups = self.atomgroups
                for group in groups:
                    if not np.allclose(
                            group.total_charge(compound=get_compound(group)),
                            0, atol=1E-4):
                        with warnings.catch_warnings():
                            warnings.simplefilter(filter)
                            warnings.warn("At least one AtomGroup has free "
                                          "charges. Analysis for systems "
                                          "with free charges could lead to "
                                          "severe artifacts!")

                    if not np.allclose(group.universe.atoms.total_charge(), 0,
                                       atol=1E-4):
                        raise ValueError(
                            "Analysis for non-neutral systems is not supported."
                            )
                return function(self)

            return wrapped

        original_class._prepare = charge_check(original_class._prepare)

        return original_class

    return inner
