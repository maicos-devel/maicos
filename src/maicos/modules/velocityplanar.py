#!/usr/bin/env python
#
# Copyright (c) 2025 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the GNU Public Licence, v3 or any higher version
# SPDX-License-Identifier: GPL-3.0-or-later
"""Module for computing planar velocity profiles."""

import MDAnalysis as mda

from ..core import ProfilePlanarBase
from ..lib.util import render_docs
from ..lib.weights import velocity_weights


@render_docs
class VelocityPlanar(ProfilePlanarBase):
    r"""Velocity profiles in a cartesian geometry.

    Reads in coordinates and velocities from a trajectory and calculates a velocity
    :math:`[\mathrm{Å/ps}]` or a flux per unit area :math:`[\mathrm{Å^{-2}\,ps^{-1}}]`
    profile along a given axis.

    The ``grouping`` keyword gives you fine control over the velocity profile, e.g. you
    can choose atomar or molecular velocities. Note that if the first one is employed
    for complex compounds, usually a contribution corresponding to the vorticity appears
    in the profile.

    ${CORRELATION_INFO_PLANAR}

    Parameters
    ----------
    ${PROFILE_PLANAR_CLASS_PARAMETERS}
    ${VDIM_PARAMETER}
    $FLUX_PARAMETER}

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
        bin_width: float = 1.0,
        refgroup: mda.AtomGroup | None = None,
        sym: bool = False,
        grouping: str = "atoms",
        unwrap: bool = True,
        pack: bool = True,
        bin_method: str = "com",
        output: str = "velocity.dat",
        concfreq: int = 0,
        vdim: int = 0,
        flux: bool = False,
        jitter: float = 0.0,
    ) -> None:
        self._locals = locals()
        if vdim not in [0, 1, 2]:
            raise ValueError("Velocity dimension can only be x=0, y=1 or z=2.")
        normalization = "volume" if flux else "number"

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
            weighting_function=velocity_weights,
            weighting_function_kwargs={"vdim": vdim},
            normalization=normalization,
        )
