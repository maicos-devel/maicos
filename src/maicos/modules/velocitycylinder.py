#!/usr/bin/env python3
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
#
# Copyright (c) 2022 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the GNU Public Licence, v3 or any higher version
# SPDX-License-Identifier: GPL-3.0-or-later
"""Module for computing cylindrical velocity profiles."""

from ..core import ProfileCylinderBase
from ..lib.util import render_docs
from ..lib.weights import velocity_weights


@render_docs
class VelocityCylinder(ProfileCylinderBase):
    r"""Compute the cartesian velocity profile across a cylinder.

    Parameters
    ----------
    ${PROFILE_CYLINDER_CLASS_PARAMETERS}
    vdim : int {0, 1, 2},
        Dimension for velocity binning (x=0, y=1, z=2).
    flux : bool,
        Calculate the flux instead of the velocity.

        Flux is calculated by multiplying the velocity by the
        number of compounds.

    Attributes
    ----------
    ${PROFILE_CYLINDER_CLASS_ATTRIBUTES}
    """

    def __init__(self,
                 atomgroups,
                 dim=2,
                 zmin=None,
                 zmax=None,
                 bin_width=1,
                 rmin=0,
                 rmax=None,
                 refgroup=None,
                 grouping="atoms",
                 unwrap=True,
                 bin_method="com",
                 output="velocity.dat",
                 concfreq=0,
                 vdim=2,
                 flux=False):

        if vdim not in [0, 1, 2]:
            raise ValueError("Velocity dimension can only be x=0, y=1 or z=2.")

        super(VelocityCylinder, self).__init__(
            weighting_function=velocity_weights,
            f_kwargs={"vdim": vdim, "flux": flux},
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
            concfreq=concfreq)
