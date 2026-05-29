#!/usr/bin/env python
#
# Copyright (c) 2026 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the GNU Public Licence, v3 or any higher version
# SPDX-License-Identifier: GPL-3.0-or-later
"""Module for computing cylindrical density profiles."""

import logging

import MDAnalysis as mda

from ..core import ProfileCylinderBase
from ..lib.util import render_docs, unit_vectors_cylinder
from ..lib.weights import density_weights, diporder_weights

logger = logging.getLogger(__name__)


@render_docs
class DensityCylinder(ProfileCylinderBase):
    r"""Cylindrical partial density profiles.

    ${DENSITY_CYLINDER_DESCRIPTION}

    ${CORRELATION_INFO_RADIAL}

    Parameters
    ----------
    ${ATOMGROUP_PARAMETER}
    ${DENS_PARAMETER}
    ${PDIM_RADIAL_PARAMETER}
    ${PROFILE_CYLINDER_CLASS_PARAMETERS}
    ${OUTPUT_PARAMETER}

    Attributes
    ----------
    ${PROFILE_CYLINDER_CLASS_ATTRIBUTES}

    """

    def __init__(
        self,
        atomgroup: mda.AtomGroup,
        dens: str = "mass",
        pdim: str = "r",
        dim: int = 2,
        zmin: float | None = None,
        zmax: float | None = None,
        rmin: float = 0,
        rmax: float | None = None,
        bin_width: float = 1,
        bin_method: str = "com",
        grouping: str = "atoms",
        refgroup: mda.AtomGroup | None = None,
        unwrap: bool = True,
        pack: bool = True,
        concfreq: int = 0,
        jitter: float = 0.0,
        output: str = "density.dat",
    ) -> None:
        self._locals = locals()

        if dens == "dipole":

            def get_unit_vectors(atomgroup: mda.AtomGroup, grouping: str):
                return unit_vectors_cylinder(
                    atomgroup=atomgroup,
                    grouping=grouping,
                    bin_method=bin_method,
                    dim=dim,
                    pdim=pdim,
                )

            weighting_function = diporder_weights
            weighting_function_kwargs = {
                "order_parameter": "P0",
                "get_unit_vectors": get_unit_vectors,
            }
        else:
            weighting_function = density_weights
            weighting_function_kwargs = {"dens": dens}

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
            rmin=rmin,
            rmax=rmax,
            refgroup=refgroup,
            grouping=grouping,
            bin_method=bin_method,
            output=output,
            weighting_function=weighting_function,
            weighting_function_kwargs=weighting_function_kwargs,
            normalization="volume",
        )

    def _prepare(self):
        logger.info(f"Analysis of the {self._locals['dens']} density profile.")
        super()._prepare()
