#!/usr/bin/env python
#
# Copyright (c) 2025 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the GNU Public Licence, v3 or any higher version
# SPDX-License-Identifier: GPL-3.0-or-later
"""Module for computing spherical density profiles."""

import MDAnalysis as mda

from ..core import ProfileSphereBase
from ..lib.util import render_docs
from ..lib.weights import density_weights


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
