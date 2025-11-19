#!/usr/bin/env python
#
# Copyright (c) 2025 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the GNU Public Licence, v3 or any higher version
# SPDX-License-Identifier: GPL-3.0-or-later
"""Module for computing dielectric spectra for bulk systems."""

import MDAnalysis as mda


class DielectricSpectrum:
    """Class has been moved to the `spectrakit` package."""

    def __init__(
        self,
        atomgroup: mda.AtomGroup,  # noqa
        temperature: float = 300,  # noqa
        segs: int = 20,  # noqa
        df: float | None = None,  # noqa
        bins: int = 200,  # noqa
        binafter: float = 20,  # noqa
        nobin: bool = False,  # noqa
        refgroup: mda.AtomGroup | None = None,  # noqa
        unwrap: bool = True,  # noqa
        pack: bool = True,  # noqa
        jitter: float = 0.0,  # noqa
        concfreq: int = 0,  # noqa
        output_prefix: str = "",  # noqa
    ) -> None:
        raise ValueError(self.__doc__)
