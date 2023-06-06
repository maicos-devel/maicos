#!/usr/bin/env python
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
#
# Copyright (c) 2023 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the GNU Public Licence, v3 or any higher version
# SPDX-License-Identifier: GPL-3.0-or-later
"""Module for computing planar temperature profiles."""

import logging
from typing import List, Optional, Union

import MDAnalysis as mda

from ..core import ProfilePlanarBase
from ..lib.util import render_docs
from ..lib.weights import temperature_weights


logger = logging.getLogger(__name__)


@render_docs
class TemperaturePlanar(ProfilePlanarBase):
    """Temperature profiles in a cartesian geometry.

    Currently only atomistic temperature profiles are supported. Therefore grouping per
    molecule, segment, residue, or fragment is not possible.

    ${CORRELATION_INFO_PLANAR}

    Parameters
    ----------
    ${PROFILE_PLANAR_CLASS_PARAMETERS}

    Attributes
    ----------
    ${PROFILE_PLANAR_CLASS_ATTRIBUTES}
    """

    def __init__(
        self,
        atomgroups: Union[mda.AtomGroup, List[mda.AtomGroup]],
        dim: int = 2,
        zmin: Optional[float] = None,
        zmax: Optional[float] = None,
        bin_width: float = 1,
        refgroup: Optional[mda.AtomGroup] = None,
        sym: bool = False,
        grouping: str = "atoms",
        unwrap: bool = True,
        bin_method: str = "com",
        output: str = "temperature.dat",
        concfreq: float = 0,
        jitter: float = 0.0,
    ):
        if grouping != "atoms":
            raise ValueError("Invalid choice of grouping, must use atoms")

        super().__init__(
            weighting_function=temperature_weights,
            normalization="number",
            atomgroups=atomgroups,
            dim=dim,
            zmin=zmin,
            zmax=zmax,
            bin_width=bin_width,
            refgroup=refgroup,
            sym=sym,
            grouping=grouping,
            unwrap=unwrap,
            bin_method=bin_method,
            output=output,
            concfreq=concfreq,
            jitter=jitter,
        )
