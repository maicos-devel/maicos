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

import numpy as np

from .utils import check_compound


verbose_parameter_doc = (
    """verbose : bool
        Turn on more logging and debugging"""
    )

planar_class_parameters_doc = (
    """dim : int
        Dimension for binning (x=0, y=1, z=2)
    zmin : float
        Minimal coordinate for evaluation (nm).
    zmax : float
        Maximal coordinate for evaluation (nm). If None, the entire
        (possibly fluctuating) length of the box is taken into account.
    binwidth : float
        binwidth (nanometer)
    comgroup : AtomGroup
        Perform the binning relative to the center of mass of the
        selected group.
    center : bool
        Perform the binning relative to the center of the (changing) box."""
    )

planar_class_attributes_doc = (
    """results.z : list
        bins"""
    )

make_whole_parameter_doc = (
    """make_whole : bool
        Make molecules whole; If the input already contains whole molecules
        this can be disabled to gain speedup"""
    )


def set_verbose_doc(public_api):
    """Set verbose for planar class."""
    if public_api.__doc__ is not None:
        public_api.__doc__ = public_api.__doc__.replace(
            "${VERBOSE_PARAMETER}",
            verbose_parameter_doc)
    return public_api


def set_planar_class_doc(public_api):
    """Set doc for planar class."""
    if public_api.__doc__ is not None:
        public_api.__doc__ = public_api.__doc__.replace(
            "${PLANAR_CLASS_PARAMETERS}",
            planar_class_parameters_doc)
        public_api.__doc__ = public_api.__doc__.replace(
            "${PLANAR_CLASS_ATTRIBUTES}",
            planar_class_attributes_doc)
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
        Filter type to control warning filter
        Common values are: "error" or "default"
        See `warnings.simplefilter` for more options.
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
                            group.total_charge(compound='fragments'), 0,
                            atol=1E-5):
                        with warnings.catch_warnings():
                            warnings.simplefilter(filter)
                            warnings.warn(
                                "At least one AtomGroup has free charges. "
                                "Analysis for systems with free charges"
                                "could lead to severe artifacts!")

                    if not np.allclose(group.universe.atoms.total_charge(), 0,
                                       atol=1E-5):
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

        if original_class.__doc__ is not None:
            original_class.__doc__ = original_class.__doc__.replace(
                "${MAKE_WHOLE_PARAMETER}",
                make_whole_parameter_doc)
        original_class._single_frame = make_whole(original_class._single_frame)

        return original_class

    return inner
