#!/usr/bin/env python
#
# Copyright (c) 2025 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the GNU Public Licence, v3 or any higher version
# SPDX-License-Identifier: GPL-3.0-or-later
r"""Module for computing Saxs structure factors and scattering intensities."""

import MDAnalysis as mda


class Saxs:
    """Class has been moved to the `scatterkit` package."""

    def __init__(
        self,
        atomgroup: mda.AtomGroup,  # noqa
        bin_spectrum: bool = True,  # noqa
        qmin: float = 0,  # noqa
        qmax: float = 6,  # noqa
        dq: float = 0.1,  # noqa
        thetamin: float = 0,  # noqa
        thetamax: float = 180,  # noqa
        refgroup: mda.AtomGroup | None = None,  # noqa
        unwrap: bool = False,  # noqa
        pack: bool = True,  # noqa
        jitter: float = 0.0,  # noqa
        concfreq: int = 0,  # noqa
        output: str = "sq.dat",  # noqa
    ) -> None:
        raise ValueError(self.__doc__)
