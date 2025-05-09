#!/usr/bin/env python
#
# Copyright (c) 2025 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the GNU Public Licence, v3 or any higher version
# SPDX-License-Identifier: GPL-3.0-or-later
"""Module for computing cylindrical density profiles."""

import logging

import MDAnalysis as mda

from ..core import ProfileCylinderBase
from ..lib.util import render_docs
from ..lib.weights import density_weights


@render_docs
class DensityCylinder(ProfileCylinderBase):
    r"""Cylindrical partial density profiles.

    ${DENSITY_CYLINDER_DESCRIPTION}

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
        atomgroup: mda.AtomGroup,
        dens: str = "mass",
        dim: int = 2,
        zmin: float | None = None,
        zmax: float | None = None,
        bin_width: float = 1,
        rmin: float = 0,
        rmax: float | None = None,
        refgroup: mda.AtomGroup | None = None,
        grouping: str = "atoms",
        unwrap: bool = True,
        pack: bool = True,
        bin_method: str = "com",
        output: str = "density.dat",
        concfreq: int = 0,
        jitter: float = 0.0,
    ) -> None:
        self._locals = locals()
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
            rmin=rmin,
            rmax=rmax,
            refgroup=refgroup,
            grouping=grouping,
            bin_method=bin_method,
            output=output,
            weighting_function=density_weights,
            weighting_function_kwargs={"dens": dens},
            normalization="volume",
        )

    def _prepare(self):
        logging.info(f"Analysis of the {self._locals['dens']} density profile.")
        super()._prepare()
