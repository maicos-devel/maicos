#!/usr/bin/env python
#
# Copyright (c) 2025 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the GNU Public Licence, v3 or any higher version
# SPDX-License-Identifier: GPL-3.0-or-later
"""Module for computing Radial Distribution functions for dipoles."""

import MDAnalysis as mda


class RDFDiporder:
    """Class has been moved to the `scatterkit` package."""

    def __init__(
        self,
        g1: mda.AtomGroup,  # noqa
        g2: mda.AtomGroup | None = None,  # noqa
        norm: str = "rdf",  # noqa
        rmin: float = 0.0,  # noqa
        rmax: float = 15.0,  # noqa
        bin_width: float = 0.1,  # noqa
        bin_method: str = "com",  # noqa
        grouping: str = "residues",  # noqa
        refgroup: mda.AtomGroup | None = None,  # noqa
        unwrap: bool = True,  # noqa
        pack: bool = True,  # noqa
        jitter: float = 0.0,  # noqa
        concfreq: int = 0,  # noqa
        output: str = "diporderrdf.dat",  # noqa
    ) -> None:
        raise ValueError(self.__doc__)
