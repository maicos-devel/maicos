#!/usr/bin/env python3
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
#
# Copyright (c) 2022 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the GNU Public Licence, v3 or any higher version
# SPDX-License-Identifier: GPL-3.0-or-later
"""Decorators adding functionalities to MAICoS classes."""
import functools
import warnings
from typing import Callable

import numpy as np

from .utils import check_compound


doc_dict = dict(
    ATOMGROUP_PARAMETER="""atomgroup : AtomGroup
        A :class:`~MDAnalysis.core.groups.AtomGroup` for which
        the calculations are performed.""",
    ATOMGROUPS_PARAMETER="""atomgroups : list[AtomGroup]
        a list of :class:`~MDAnalysis.core.groups.AtomGroup` for which
        the calculations are calculated.""",
    BASE_CLASS_PARAMETERS="""refgroup : AtomGroup
        Perform the calculation relative to the center of mass of the
        provided AtomGroup. If `None` the calculations are performed relative
        to the center of the fluctuating box.
    unwrap : bool
        Make molecules that are broken due to the periodic boundary conditions
        whole again. If the input already contains whole molecules this can
        be disabled to gain speedup.

        Note: Currently molecules containing virtual sites (e.g. TIP4P water
        model) are not supported. In this case, provide unwrapped trajectory
        file directly, and use the command line flag -no-unwrap.
    concfreq : int
        Call the conclcude function and write the output files every n frames
    verbose : bool
        Turn on more logging and debugging""",
    PLANAR_CLASS_PARAMETERS="""dim : int, default: 2
        Dimension for binning (x=0, y=1, z=2)
    zmin : float, default: None
        Minimal coordinate for evaluation (Å) with respect to the refgroup.
        If `None` all coordinates
        up to the cell boundary are taken into account.
    zmax : float, default: None
        Maximal coordinate for evaluation (Å) with respect to the refgroup.
        If `None` all coordinates
        up to the cell boundary are taken into account.
    binwidth : float
        binwidth""",
    CYLINDER_CLASS_PARAMETERS="""rmin : float
        Minimal r-coordinate relative to the comgroup center of mass for
        evaluation (Å).
    rmax : float
        Maximal z-coordinate relative to the comgroup center of mass for
        evaluation (Å). If None the box extension is taken.""",
    SYM_PARAMETER="""sym : bool
        symmetrize the profile. Only works in combinations with `refgroup`.""",
    PROFILE_CLASS_PARAMETERS="""grouping : str {'atoms', 'residues', 'segments', 'molecules', 'fragments'}"""    # noqa
    """
          Profile will be computed either on the atom positions (in
          the case of 'atoms') or on the center of mass of the specified
          grouping unit ('residues', 'segments', or 'fragments').
    binmethod : str
        Method for position binning; possible options are
        center of geometry (cog), center of mass (com) or
        center of charge (coc).
    output : str
        Output filename""",
    PLANAR_CLASS_ATTRIBUTES="""results.z : list
        Bin position ranging from `zmin` to `zmax`.""",
    CYLINDER_CLASS_ATTRIBUTES="""results.r : list
        Bin position ranging from `rmin` to `rmax`.""",
    PROFILE_CLASS_ATTRIBUTES="""results.profile_mean : np.ndarray
        calculated profile
    results.profile_err : np.ndarray
        profile's error"""
    )

# Inherit docstrings
doc_dict["PLANAR_CLASS_PARAMETERS"] = \
    doc_dict["BASE_CLASS_PARAMETERS"] + "\n    " + \
    doc_dict["PLANAR_CLASS_PARAMETERS"]

doc_dict["CYLINDER_CLASS_PARAMETERS"] = \
    doc_dict["PLANAR_CLASS_PARAMETERS"] + "\n    " + \
    doc_dict["CYLINDER_CLASS_PARAMETERS"]

doc_dict["PROFILE_PLANAR_CLASS_PARAMETERS"] = \
    doc_dict["ATOMGROUPS_PARAMETER"] + "\n    " + \
    doc_dict["PLANAR_CLASS_PARAMETERS"] + "\n    " + \
    doc_dict["SYM_PARAMETER"] + "\n    " + \
    doc_dict["PROFILE_CLASS_PARAMETERS"]

doc_dict["PROFILE_CYLINDER_CLASS_PARAMETERS"] = \
    doc_dict["ATOMGROUPS_PARAMETER"] + "\n    " + \
    doc_dict["CYLINDER_CLASS_PARAMETERS"] + "\n    " + \
    doc_dict["PROFILE_CLASS_PARAMETERS"]

doc_dict["PROFILE_PLANAR_CLASS_ATTRIBUTES"] = \
    doc_dict["PLANAR_CLASS_ATTRIBUTES"] + "\n    " + \
    doc_dict["PROFILE_CLASS_ATTRIBUTES"]

doc_dict["PROFILE_CYLINDER_CLASS_ATTRIBUTES"] = \
    doc_dict["CYLINDER_CLASS_ATTRIBUTES"] + "\n    " + \
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
                            group.total_charge(compound=check_compound(group)),
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
