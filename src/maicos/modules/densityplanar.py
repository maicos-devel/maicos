#!/usr/bin/env python3
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
#
# Copyright (c) 2022 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the GNU Public Licence, v3 or any higher version
# SPDX-License-Identifier: GPL-3.0-or-later
"""Module for computing planar density profiles."""

import logging

from ..core import ProfilePlanarBase
from ..lib.util import render_docs
from ..lib.weights import density_weights


logger = logging.getLogger(__name__)


@render_docs
class DensityPlanar(ProfilePlanarBase):
    r"""Cartesian partial density profiles.

    ${DENSITY_DESCRIPTION}

    Parameters
    ----------
    ${PROFILE_PLANAR_CLASS_PARAMETERS}
    dens : str {'mass', 'number', 'charge'}
        density type to be calculated.

    Attributes
    ----------
    ${PROFILE_PLANAR_CLASS_ATTRIBUTES}

    Notes
    -----
    Partial mass density profiles can be used to calculate the ideal component
    of the chemical potential. For details, take a look at the corresponding
    :ref:`How-to guide<howto-chemical-potential>`.
    """

    def __init__(self,
                 atomgroups,
                 dens="mass",
                 dim=2,
                 zmin=None,
                 zmax=None,
                 bin_width=1,
                 refgroup=None,
                 sym=False,
                 grouping="atoms",
                 unwrap=True,
                 bin_method="com",
                 output="density.dat",
                 concfreq=0,
                 jitter=0.0):

        super(DensityPlanar, self).__init__(
            weighting_function=density_weights,
            f_kwargs={"dens": dens},
            normalization="volume",
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
