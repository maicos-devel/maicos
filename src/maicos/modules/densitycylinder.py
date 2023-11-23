#!/usr/bin/env python
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
#
# Copyright (c) 2023 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the GNU Public Licence, v3 or any higher version
# SPDX-License-Identifier: GPL-3.0-or-later
"""Module for computing cylindrical density profiles."""

import logging
from typing import List, Optional, Union

import MDAnalysis as mda

from ..core import ProfileCylinderBase
from ..lib.util import render_docs
from ..lib.weights import density_weights


logger = logging.getLogger(__name__)


@render_docs
class DensityCylinder(ProfileCylinderBase):
    r"""Cylindrical partial density profiles.

    ${DENSITY_DESCRIPTION}

    ${CORRELATION_INFO_RADIAL}

    Parameters
    ----------
    ${PROFILE_CYLINDER_CLASS_PARAMETERS}
    ${DENS_PARAMETER}

    Attributes
    ----------
    ${PROFILE_CYLINDER_CLASS_ATTRIBUTES}
    """

    def __init__(
        self,
        atomgroups: Union[mda.AtomGroup, List[mda.AtomGroup]],
        dens: str = "mass",
        dim: int = 2,
        zmin: Optional[float] = None,
        zmax: Optional[float] = None,
        bin_width: float = 1,
        rmin: float = 0,
        rmax: Optional[float] = None,
        refgroup: Optional[mda.AtomGroup] = None,
        grouping: str = "atoms",
        unwrap: bool = True,
        bin_method: str = "com",
        output: str = "density.dat",
        concfreq: int = 0,
        jitter: float = 0.0,
    ):
        self._locals = locals()
        super().__init__(
            weighting_function=density_weights,
            f_kwargs={"dens": dens},
            normalization="volume",
            atomgroups=atomgroups,
            dim=dim,
            zmin=zmin,
            zmax=zmax,
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
