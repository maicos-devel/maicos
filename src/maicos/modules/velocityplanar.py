#!/usr/bin/env python3
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
#
# Copyright (c) 2022 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the GNU Public Licence, v3 or any higher version
# SPDX-License-Identifier: GPL-3.0-or-later
"""Module for computing planar velocity profiles."""

from ..core import ProfilePlanarBase
from ..lib.util import render_docs
from ..lib.weights import velocity_weights


@render_docs
class VelocityPlanar(ProfilePlanarBase):
    r"""Velocity profiles in a cartesian geometry.

    Reads in coordinates and velocities from a trajectory and calculates a
    velocity :math:`[\mathrm{Å/ps}]` or a flux per unit area
    :math:`[\mathrm{Å^{-2}\,ps^{-1}}]` profile along a given axis.

    The ``grouping`` keyword gives you fine control over the velocity profile,
    e.g. you can choose atomar or molecular velocities.
    Note that if the first one is employed for complex compounds, usually a
    contribution corresponding to the vorticity appears in the profile.

    ${CORRELATION_INFO_PLANAR}

    Parameters
    ----------
    ${PROFILE_PLANAR_CLASS_PARAMETERS}
    vdim : int, {0, 1, 2}
        Dimension for velocity binning (x=0, y=1, z=2).
    flux : bool
        Calculate the flux instead of the velocity.

    Attributes
    ----------
    ${PROFILE_PLANAR_CLASS_ATTRIBUTES}
    """

    def __init__(self,
                 atomgroups,
                 dim=2,
                 zmin=None,
                 zmax=None,
                 bin_width=1,
                 refgroup=None,
                 sym=False,
                 grouping="atoms",
                 unwrap=True,
                 bin_method="com",
                 output="velocity.dat",
                 concfreq=0,
                 vdim=0,
                 flux=False,
                 jitter=0.0):

        if vdim not in [0, 1, 2]:
            raise ValueError("Velocity dimension can only be x=0, y=1 or z=2.")
        if flux:
            normalization = 'volume'
        else:
            normalization = 'number'

        super(VelocityPlanar, self).__init__(
            weighting_function=velocity_weights,
            f_kwargs={"vdim": vdim, "flux": flux},
            normalization=normalization,
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
            jitter=jitter)
