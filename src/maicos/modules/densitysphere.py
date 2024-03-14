#!/usr/bin/env python
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
#
# Copyright (c) 2024 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the GNU Public Licence, v3 or any higher version
# SPDX-License-Identifier: GPL-3.0-or-later
"""Module for computing spherical density profiles."""
import logging
from typing import Optional

import MDAnalysis as mda

from ..core import ProfileSphereBase
from ..lib.util import render_docs
from ..lib.weights import density_weights


logger = logging.getLogger(__name__)


@render_docs
class DensitySphere(ProfileSphereBase):
    r"""Spherical partial density profiles.

    ${DENSITY_DESCRIPTION}

    ${CORRELATION_INFO_RADIAL}

    Parameters
    ----------
    ${PROFILE_SPHERE_CLASS_PARAMETERS}
    ${DENS_PARAMETER}

    Attributes
    ----------
    ${PROFILE_SPHERE_CLASS_ATTRIBUTES}
    """

    def __init__(
        self,
        atomgroup: mda.AtomGroup,
        dens: str = "mass",
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
    ) -> None:
        self._locals = locals()
        super().__init__(
            atomgroup=atomgroup,
            unwrap=unwrap,
            refgroup=refgroup,
            jitter=jitter,
            concfreq=concfreq,
            rmin=rmin,
            rmax=rmax,
            bin_width=bin_width,
            grouping=grouping,
            bin_method=bin_method,
            output=output,
            weighting_function=density_weights,
            weighting_function_kwargs={"dens": dens},
            normalization="volume",
        )
