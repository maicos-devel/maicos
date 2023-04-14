#!/usr/bin/env python3
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
#
# Copyright (c) 2022 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the GNU Public Licence, v3 or any higher version
# SPDX-License-Identifier: GPL-3.0-or-later
"""Module for computing spherical density profiles."""

import logging

from ..core import ProfileSphereBase
from ..lib.util import render_docs
from ..lib.weights import density_weights


logger = logging.getLogger(__name__)


@render_docs
class DensitySphere(ProfileSphereBase):
    r"""Spherical partial density profiles.

    ${DENSITY_DESCRIPTION}

    ${CORRELATION_INFO_SPHERE}

    Parameters
    ----------
    ${PROFILE_SPHERE_CLASS_PARAMETERS}
    dens : str {'mass', 'number', 'charge'}
        density type to be calculated.

    Attributes
    ----------
    ${PROFILE_SPHERE_CLASS_ATTRIBUTES}
    """

    def __init__(self,
                 atomgroups,
                 dens="mass",
                 bin_width=1,
                 rmin=0,
                 rmax=None,
                 refgroup=None,
                 grouping="atoms",
                 unwrap=True,
                 bin_method="com",
                 output="density.dat",
                 concfreq=0,
                 jitter=0.0):

        super(DensitySphere, self).__init__(
            weighting_function=density_weights,
            f_kwargs={"dens": dens},
            normalization="volume",
            atomgroups=atomgroups,
            bin_width=bin_width,
            rmin=rmin,
            rmax=rmax,
            refgroup=refgroup,
            grouping=grouping,
            unwrap=unwrap,
            bin_method=bin_method,
            output=output,
            concfreq=concfreq,
            jitter=jitter)
