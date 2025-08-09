#!/usr/bin/env python
#
# Copyright (c) 2025 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the GNU Public Licence, v3 or any higher version
# SPDX-License-Identifier: GPL-3.0-or-later
"""Module for computing kinetic energy timeseries."""

import MDAnalysis as mda


class KineticEnergy:
    """Class has been removed and is no longer available."""

    def __init__(
        self,
        atomgroup: mda.AtomGroup,  # noqa
        refpoint: str = "com",  # noqa
        refgroup: mda.AtomGroup | None = None,  # noqa
        unwrap: bool = False,  # noqa
        pack: bool = True,  # noqa
        jitter: float = 0.0,  # noqa
        concfreq: int = 0,  # noqa
        output: str = "ke.dat",  # noqa
    ) -> None:
        raise ValueError(self.__doc__)
