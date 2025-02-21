#!/usr/bin/env python
#
# Copyright (c) 2025 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the GNU Public Licence, v3 or any higher version
# SPDX-License-Identifier: GPL-3.0-or-later
"""Module for computing planar temperature profiles."""

import logging

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
        atomgroup: mda.AtomGroup,
        dim: int = 2,
        zmin: float | None = None,
        zmax: float | None = None,
        bin_width: float = 1,
        refgroup: mda.AtomGroup | None = None,
        sym: bool = False,
        grouping: str = "atoms",
        unwrap: bool = True,
        pack: bool = True,
        bin_method: str = "com",
        output: str = "temperature.dat",
        concfreq: int = 0,
        jitter: float = 0.0,
    ) -> None:
        self._locals = locals()
        if grouping != "atoms":
            raise ValueError("Invalid choice of grouping, must use atoms")

        super().__init__(
            atomgroup=atomgroup,
            unwrap=unwrap,
            pack=pack,
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
            weighting_function=temperature_weights,
            weighting_function_kwargs=None,
            normalization="number",
        )
