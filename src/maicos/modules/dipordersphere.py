#!/usr/bin/env python3
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
#
# Copyright (c) 2023 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the GNU Public Licence, v3 or any higher version
# SPDX-License-Identifier: GPL-3.0-or-later
r"""Module for computing spherical dipolar order parameters."""

import logging
from typing import List, Optional, Union

import MDAnalysis as mda

from ..core import ProfileSphereBase
from ..lib.util import render_docs, unit_vectors_sphere
from ..lib.weights import diporder_weights


logger = logging.getLogger(__name__)


@render_docs
class DiporderSphere(ProfileSphereBase):
    r"""Spherical dipolar order parameters.

    ${DIPORDER_DESCRIPTION}

    ${CORRELATION_INFO_CYLINDER}

    Parameters
    ----------
    ${PROFILE_CYLINDER_CLASS_PARAMETERS}
    ${ORDER_PARAMETER_PARAMETER}

    Attributes
    ----------
    ${PROFILE_CYLINDER_CLASS_ATTRIBUTES}
    """

    def __init__(
        self,
        atomgroups: Union[mda.AtomGroup, List[mda.AtomGroup]],
        bin_width: float = 1,
        rmin: float = 0,
        rmax: Optional[float] = None,
        refgroup: Optional[mda.AtomGroup] = None,
        grouping: str = "residues",
        unwrap: bool = True,
        bin_method: str = "com",
        output: str = "diporder_sphere.dat",
        concfreq: int = 0,
        order_parameter: str = "P0",
        jitter: float = 0.0,
    ):
        if order_parameter == "P0":
            normalization = "volume"
        else:
            normalization = "number"

        def get_unit_vectors(atomgroup: mda.AtomGroup, grouping: str):
            return unit_vectors_sphere(
                atomgroup=atomgroup, grouping=grouping, bin_method=bin_method
            )

        super().__init__(
            weighting_function=diporder_weights,
            f_kwargs={
                "order_parameter": order_parameter,
                "get_unit_vectors": get_unit_vectors,
            },
            normalization=normalization,
            atomgroups=atomgroups,
            bin_width=bin_width,
            rmin=rmin,
            rmax=rmax,
            refgroup=refgroup,
            grouping=grouping,
            unwrap=unwrap,
            bin_method=bin_method,
            output=output,
            concfreq=concfreq,
            jitter=jitter,
        )
