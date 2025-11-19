#!/usr/bin/env python
#
# Copyright (c) 2025 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the GNU Public Licence, v3 or any higher version
# SPDX-License-Identifier: GPL-3.0-or-later
r"""Module for computing spherical dipolar order parameters."""

import logging

import MDAnalysis as mda

from ..core import ProfileSphereBase
from ..lib.util import render_docs, unit_vectors_sphere
from ..lib.weights import diporder_weights


@render_docs
class DiporderSphere(ProfileSphereBase):
    r"""Spherical dipolar order parameters.

    ${DIPORDER_DESCRIPTION}

    ${CORRELATION_INFO_RADIAL}

    Parameters
    ----------
    ${ATOMGROUP_PARAMETER}
    ${ORDER_PARAMETER_PARAMETER}
    ${PROFILE_SPHERE_CLASS_PARAMETERS}
    ${OUTPUT_PARAMETER}

    Attributes
    ----------
    ${PROFILE_SPHERE_CLASS_ATTRIBUTES}

    """

    def __init__(
        self,
        atomgroup: mda.AtomGroup,
        order_parameter: str = "P0",
        rmin: float = 0,
        rmax: float | None = None,
        bin_width: float = 1,
        bin_method: str = "com",
        grouping: str = "residues",
        refgroup: mda.AtomGroup | None = None,
        unwrap: bool = True,
        pack: bool = True,
        jitter: float = 0.0,
        concfreq: int = 0,
        output: str = "diporder_sphere.dat",
    ) -> None:
        normalization = "volume" if order_parameter == "P0" else "number"

        def get_unit_vectors(atomgroup: mda.AtomGroup, grouping: str):
            return unit_vectors_sphere(
                atomgroup=atomgroup, grouping=grouping, bin_method=bin_method
            )

        super().__init__(
            atomgroup=atomgroup,
            unwrap=unwrap,
            pack=pack,
            jitter=jitter,
            refgroup=refgroup,
            concfreq=concfreq,
            rmin=rmin,
            rmax=rmax,
            bin_width=bin_width,
            grouping=grouping,
            bin_method=bin_method,
            output=output,
            weighting_function=diporder_weights,
            weighting_function_kwargs={
                "order_parameter": order_parameter,
                "get_unit_vectors": get_unit_vectors,
            },
            normalization=normalization,
        )

    def _prepare(self):
        logging.info("Analysis of the spherical dipolar order parameters.")
        super()._prepare()
