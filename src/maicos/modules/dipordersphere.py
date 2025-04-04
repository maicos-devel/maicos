#!/usr/bin/env python3
#
# Copyright (c) 2025 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the GNU Public Licence, v3 or any higher version
# SPDX-License-Identifier: GPL-3.0-or-later
r"""Module for computing spherical dipolar order parameters."""

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
    ${PROFILE_CYLINDER_CLASS_PARAMETERS}
    ${ORDER_PARAMETER_PARAMETER}

    Attributes
    ----------
    ${PROFILE_CYLINDER_CLASS_ATTRIBUTES}

    """

    def __init__(
        self,
        atomgroup: mda.AtomGroup,
        bin_width: float = 1,
        rmin: float = 0,
        rmax: float | None = None,
        refgroup: mda.AtomGroup | None = None,
        grouping: str = "residues",
        unwrap: bool = True,
        pack: bool = True,
        bin_method: str = "com",
        output: str = "diporder_sphere.dat",
        concfreq: int = 0,
        order_parameter: str = "P0",
        jitter: float = 0.0,
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
