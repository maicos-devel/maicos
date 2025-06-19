#!/usr/bin/env python3
#
# Copyright (c) 2025 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the GNU Public Licence, v3 or any higher version
# SPDX-License-Identifier: GPL-3.0-or-later
r"""Module for computing planar dipolar order parameters."""

import logging

import MDAnalysis as mda

from ..core import ProfilePlanarBase
from ..lib.util import render_docs, unit_vectors_planar
from ..lib.weights import diporder_weights


@render_docs
class DiporderPlanar(ProfilePlanarBase):
    r"""Cartesian dipolar order parameters.

    ${DIPORDER_DESCRIPTION}

    ${CORRELATION_INFO_PLANAR}

    Parameters
    ----------
    ${PROFILE_PLANAR_CLASS_PARAMETERS}
    ${PDIM_PLANAR_PARAMETER}
    ${ORDER_PARAMETER_PARAMETER}

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
        bin_width: float = 1,
        refgroup: mda.AtomGroup | None = None,
        sym: bool = False,
        grouping: str = "residues",
        unwrap: bool = True,
        pack: bool = True,
        bin_method: str = "com",
        output: str = "diporder_planar.dat",
        concfreq: int = 0,
        pdim: int = 2,
        order_parameter: str = "P0",
        jitter: float = 0.0,
    ) -> None:
        self._locals = locals()
        normalization = "volume" if order_parameter == "P0" else "number"

        def get_unit_vectors(atomgroup: mda.AtomGroup, grouping: str):
            return unit_vectors_planar(
                atomgroup=atomgroup, grouping=grouping, pdim=pdim
            )

        super().__init__(
            atomgroup=atomgroup,
            unwrap=unwrap,
            pack=pack,
            refgroup=refgroup,
            jitter=jitter,
            concfreq=concfreq,
            dim=dim,
            zmin=zmin,
            zmax=zmax,
            bin_width=bin_width,
            sym=sym,
            sym_parity="odd",
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
        logging.info("Analysis of the cartesian dipolar order parameters.")
        super()._prepare()
