#!/usr/bin/env python3
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
#
# Copyright (c) 2022 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the GNU Public Licence, v3 or any higher version
# SPDX-License-Identifier: GPL-3.0-or-later
r"""Module for computing dipolar order parameter."""

import logging

from ..core import ProfilePlanarBase
from ..lib.util import render_docs
from ..lib.weights import diporder_planar_weights


logger = logging.getLogger(__name__)


@render_docs
class DiporderPlanar(ProfilePlanarBase):
    r"""Calculate dipolar order parameters.

    Calculations include the projected dipole density
    :math:`P_0⋅ρ(z)⋅\cos(θ[z])`, the dipole orientation
    :math:`\cos(θ[z])`, the squared dipole
    orientation :math:`\cos²(Θ[z])` and the number density :math:`ρ(z)`.

    Parameters
    ----------
    ${PROFILE_PLANAR_CLASS_PARAMETERS}
    order_parameter : str
        ``P0``, ``cos_theta`` or ``cos_2_theta``.

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
                 grouping="residues",
                 unwrap=True,
                 bin_method="com",
                 output="diporder_planar.dat",
                 concfreq=0,
                 order_parameter="P0",
                 jitter=None):

        if order_parameter == "P0":
            normalization = "volume"
        else:
            normalization = "number"

        super(DiporderPlanar, self).__init__(
            weighting_function=diporder_planar_weights,
            f_kwargs={"dim": dim, "order_parameter": order_parameter},
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
