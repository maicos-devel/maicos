#!/usr/bin/env python
#
# Copyright (c) 2026 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the GNU Public Licence, v3 or any higher version
# SPDX-License-Identifier: GPL-3.0-or-later
"""Module for computing planar density profiles."""

import logging
from dataclasses import dataclass, field  # for typechecking

import MDAnalysis as mda
import numpy as np

from ..core import ProfilePlanarBase
from ..lib.util import render_docs
from ..lib.weights import density_weights

logger = logging.getLogger(__name__)


@dataclass
class DensityPlanarObs:
    """Observables for DensityPlanar, typed for mypy."""

    # From PlanarBase
    L: float = 0.0  #     _obs.L : float
    box_center: np.ndarray = field(
        default_factory=lambda: np.zeros(3)
    )  #     _obs.box_center : np.ndarray
    bin_edges: np.ndarray = field(
        default_factory=lambda: np.zeros(0)
    )  #    _obs.bin_edges : np.ndarray
    bin_width: float = 0.0  #    _obs.bin_width : float
    bin_pos: np.ndarray = field(
        default_factory=lambda: np.zeros(0)
    )  #    _obs.bin_pos : np.ndarray
    bin_area: np.ndarray = field(
        default_factory=lambda: np.zeros(0)
    )  #    _obs.bin_area : np.ndarray
    bin_volume: np.ndarray = field(
        default_factory=lambda: np.zeros(0)
    )  #    _obs.bin_volume : np.ndarray
    # From ProfileBase line 872 and 874
    profile: np.ndarray = field(default_factory=lambda: np.zeros(0))
    bincount: np.ndarray = field(default_factory=lambda: np.zeros(0))


@render_docs
class DensityPlanar(ProfilePlanarBase):
    r"""Cartesian partial density profiles.

    ${DENSITY_PLANAR_DESCRIPTION}

    ${CORRELATION_INFO_PLANAR}

    Parameters
    ----------
    ${ATOMGROUP_PARAMETER}
    ${DENS_PARAMETER}
    ${PROFILE_PLANAR_CLASS_PARAMETERS}
    ${OUTPUT_PARAMETER}

    Attributes
    ----------
    ${PROFILE_PLANAR_CLASS_ATTRIBUTES}

    Notes
    -----
    Partial mass density profiles can be used to calculate the ideal component of the
    chemical potential. For details, take a look at the corresponding :ref:`How-to
    guide<sphx_glr_generated_examples_basics_chemical-potential.py>`.

    """

    def __init__(
        self,
        atomgroup: mda.AtomGroup,
        dens: str = "mass",
        dim: int = 2,
        zmin: float | None = None,
        zmax: float | None = None,
        bin_width: float = 1,
        bin_method: str = "com",
        grouping: str = "atoms",
        sym: bool = False,
        refgroup: mda.AtomGroup | None = None,
        unwrap: bool = True,
        pack: bool = True,
        jitter: float = 0.0,
        concfreq: int = 0,
        output: str = "density.dat",
    ) -> None:
        self._locals = locals()
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
            refgroup=refgroup,
            sym=sym,
            sym_odd=False,
            grouping=grouping,
            bin_method=bin_method,
            output=output,
            weighting_function=density_weights,
            weighting_function_kwargs={"dens": dens},
            normalization="volume",
        )

    def _prepare(self):
        logger.info(f"Analysis of the {self._locals['dens']} density profile.")
        super()._prepare()
        # Helped by AI (Claude). Annotate _obs, means and sems so mypy knows
        # their types instead of relying on the generic Results() container.
        self._obs: DensityPlanarObs = DensityPlanarObs()
        self.means: DensityPlanarObs = DensityPlanarObs()
        self.sems: DensityPlanarObs = DensityPlanarObs()
