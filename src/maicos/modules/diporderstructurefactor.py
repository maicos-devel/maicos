#!/usr/bin/env python
#
# Copyright (c) 2025 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the GNU Public Licence, v3 or any higher version
# SPDX-License-Identifier: GPL-3.0-or-later
"""Module for computing structure factor for dipoles."""

import MDAnalysis as mda


class DiporderStructureFactor:
    """Class has been moved to the `scatterkit` package."""

    def __init__(
        self,
        atomgroup: mda.AtomGroup,  # noqa
        qmin: float = 0,  # noqa
        qmax: float = 6,  # noqa
        dq: float = 0.01,  # noqa
        bin_method: str = "com",  # noqa
        grouping: str = "molecules",  # noqa
        refgroup: mda.AtomGroup | None = None,  # noqa
        unwrap: bool = True,  # noqa
        pack: bool = True,  # noqa
        jitter: float = 0.0,  # noqa
        concfreq: int = 0,  # noqa
        output: str = "sq.dat",  # noqa
    ) -> None:
        raise ValueError(self.__doc__)
