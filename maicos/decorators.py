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


verbose_parameter_doc = (
    """verbose : bool
        Turn on more logging and debugging"""
    )

make_whole_parameter_doc = (
    """make_whole : bool
        Make molecules that are broken due to the periodic boundary conditions
        whole again. If the input already contains whole molecules this can
        be disabled to gain speedup.

        Note: Currently molecules containing virtual sites (e.g. TIP4P water
        model) are not supported. In this case, provide unwrapped trajectory
        file directly, and use the command line flag -no-make_whole.
        """
    )

planar_class_parameters_doc = (
    """dim : int
        Dimension for binning (x=0, y=1, z=2)
    zmin : float
        Minimal coordinate for evaluation (Å).
    zmax : float
        Maximal coordinate for evaluation (Å). If None, the entire
        (possibly fluctuating) length of the box is taken into account.
    binwidth : float
        binwidth (nanometer)
    comgroup : AtomGroup
        Perform the binning relative to the center of mass of the
        provided AtomGroup."""
    )

profile_planar_class_parameters_doc = (
    """atomgroups : list[AtomGroup]
        a list of :class:`~MDAnalysis.core.groups.AtomGroup` for which
        the densities are calculated."""
    + planar_class_parameters_doc
    + """sym : bool
        symmetrize the profile. Only works in combinations with `comgroup`.
    grouping : str {'atoms', 'residues', 'segments', 'molecules', 'fragments'}
          Profile will be computed either on the atom positions (in
          the case of 'atoms') or on the center of mass of the specified
          grouping unit ('residues', 'segments', or 'fragments')."""
    + make_whole_parameter_doc
    + """binmethod : str
        Method for position binning; possible options are
        center of geometry (cog), center of mass (com) or
        center of charge (coc).
    output : str
        Output filename
    concfreq : int
        Default number of frames after which results are calculated and
        files refreshed. If `0` results are only calculated at the end
        of the analysis and not saved by default."""
    )

planar_class_attributes_doc = (
    """results.z : list
        bins"""
    )

profile_planar_class_attributes_doc = (
    planar_class_attributes_doc
    + """results.profile_mean : np.ndarray
        calculated profile
    results.profile_std : np.ndarray
        profile's standard deviation
    results.profile_err : np.ndarray
        profile's error"""
    )


def set_doc(func: Callable, old: str, new: str) -> Callable:
    """Replace template phrase in a function with an actual docstring.

    Parameters
    ----------
    func : callable
        The callable (function, class) where the phrase old should be replaced.
    old : str
        The template phrase which will be replaced
    new : str
        The actual phrase which will appear in the docstring
        Returns
        -------
        Callable
            callable with replaced phrase
    """
    if func.__doc__ is not None:
        func.__doc__ = func.__doc__.replace(old, new)
    return func


def set_verbose_doc(public_api):
    """Set doc for planar class."""
    public_api = set_doc(public_api, "${VERBOSE_PARAMETER}",
                         verbose_parameter_doc)
    return public_api


def set_planar_class_doc(public_api: Callable) -> None:
    """Set doc for planar class."""
    public_api = set_doc(public_api, "${PLANAR_CLASS_PARAMETERS}",
                         planar_class_parameters_doc)
    public_api = set_doc(public_api, "${PLANAR_CLASS_ATTRIBUTES}",
                         planar_class_attributes_doc)
    return public_api


def set_profile_planar_class_doc(public_api: Callable) -> None:
    """Set doc for profile planar class."""
    public_api = set_doc(public_api, "${PLANAR_PROFILE_CLASS_PARAMETERS}",
                         profile_planar_class_parameters_doc)
    public_api = set_doc(public_api, "${PLANAR_PROFILE_CLASS_ATTRIBUTES}",
                         profile_planar_class_attributes_doc)
    return public_api


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


def make_whole():
    """Class Decorator to make molecules whole in each analysis step."""
    def inner(original_class):
        def make_whole(function):
            @functools.wraps(function)
            def wrapped(self):
                if hasattr(self, 'make_whole') and self.make_whole:
                    if hasattr(self, "atomgroup"):
                        groups = [self.atomgroup]
                    else:
                        groups = self.atomgroups
                    for group in groups:
                        group.unwrap(compound=check_compound(group))
                return function(self)

            return wrapped

        original_class = set_doc(original_class, "${MAKE_WHOLE_PARAMETER}",
                                 make_whole_parameter_doc)
        original_class._single_frame = make_whole(original_class._single_frame)

        return original_class

    return inner
