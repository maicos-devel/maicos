#!/usr/bin/env python3
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
#
# Copyright (c) 2022 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the GNU Public Licence, v3 or any higher version
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tool for computing transport properties.

The transport module of MAICoS allows for calculating mean velocity
profiles from molecular simulation trajectory files.
"""

from ..core import ProfilePlanarBase
from ..lib.util import render_docs
from ..lib.weights import velocity_weights


@render_docs
class Velocity(ProfilePlanarBase):
    """Analyse mean velocity.

    Reads in coordinates and velocities from a trajectory and calculates a
    velocity profile along a given axis.

    Parameters
    ----------
    ${PROFILE_PLANAR_CLASS_PARAMETERS}
    vdim : int {0, 1, 2}
        Dimension for velocity binning (x=0, y=1, z=2)
    flux : bool
        Do not normalise the velocity to get the flux,
        i.e. the velocity multiplied by the number of compounds

    Attributes
    ----------
    ${PROFILE_PLANAR_CLASS_ATTRIBUTES}
    """

    def __init__(self,
                 atomgroups,
                 dim=2,
                 zmin=None,
                 zmax=None,
                 binwidth=1,
                 refgroup=None,
                 sym=False,
                 grouping="atoms",
                 unwrap=True,
                 binmethod="com",
                 output="velocity.da",
                 concfreq=0,
                 vdim=2,
                 flux=False,
                 **kwargs):

        if vdim not in [0, 1, 2]:
            raise ValueError("Velocity dimension can only be x=0, y=1 or z=2.")

        super(Velocity, self).__init__(
            function=velocity_weights,
            f_kwargs={"vdim": vdim, "flux": flux},
            normalization="volume",
            atomgroups=atomgroups,
            dim=dim,
            zmin=zmin,
            zmax=zmax,
            binwidth=binwidth,
            refgroup=refgroup,
            sym=sym,
            grouping=grouping,
            unwrap=unwrap,
            binmethod=binmethod,
            output=output,
            concfreq=concfreq,
            **kwargs)
