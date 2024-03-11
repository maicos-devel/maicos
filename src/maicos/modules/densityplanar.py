#!/usr/bin/env python
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
#
# Copyright (c) 2024 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the GNU Public Licence, v3 or any higher version
# SPDX-License-Identifier: GPL-3.0-or-later
"""Module for computing planar density profiles."""
import logging
from typing import List, Optional, Union

import MDAnalysis as mda

from ..core import ProfilePlanarBase
from ..lib.util import render_docs
from ..lib.weights import density_weights


logger = logging.getLogger(__name__)


@render_docs
class DensityPlanar(ProfilePlanarBase):
    r"""Cartesian partial density profiles.

    ${DENSITY_DESCRIPTION}

    ${CORRELATION_INFO_PLANAR}

    Parameters
    ----------
    ${PROFILE_PLANAR_CLASS_PARAMETERS}
    ${DENS_PARAMETER}

    Attributes
    ----------
    ${PROFILE_PLANAR_CLASS_ATTRIBUTES}

    Notes
    -----
    Partial mass density profiles can be used to calculate the ideal component of the
    chemical potential. For details, take a look at the corresponding :ref:`How-to
    guide<howto-chemical-potential>`.
    """

    def __init__(
        self,
        atomgroups: Union[mda.AtomGroup, List[mda.AtomGroup]],
        dens: str = "mass",
        dim: int = 2,
        zmin: Optional[float] = None,
        zmax: Optional[float] = None,
        bin_width: float = 1,
        refgroup: Optional[mda.AtomGroup] = None,
        sym: bool = False,
        grouping: str = "atoms",
        unwrap: bool = True,
        bin_method: str = "com",
        output: str = "density.dat",
        concfreq: int = 0,
        jitter: float = 0.0,
    ) -> None:
        self._locals = locals()
        super().__init__(
            atomgroups=atomgroups,
            unwrap=unwrap,
            jitter=jitter,
            concfreq=concfreq,
            dim=dim,
            zmin=zmin,
            zmax=zmax,
            bin_width=bin_width,
            refgroup=refgroup,
            sym=sym,
            grouping=grouping,
            bin_method=bin_method,
            output=output,
            weighting_function=density_weights,
            weighting_function_kwargs={"dens": dens},
            normalization="volume",
        )
