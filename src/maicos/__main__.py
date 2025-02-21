#!/usr/bin/env python
#
# Copyright (c) 2025 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the GNU Public Licence, v3 or any higher version
# SPDX-License-Identifier: GPL-3.0-or-later
"""Analyse molecular dynamics simulation of interfacial and confined systems."""

from mdacli import cli

from maicos import __version__
from maicos.core import AnalysisBase


def main():
    """Execute main CLI entry point."""
    cli(
        name="MAICoS",
        module_list=["maicos"],
        base_class=AnalysisBase,
        version=__version__,
        description=__doc__,
        ignore_warnings=True,
    )


if __name__ == "__main__":
    main()
