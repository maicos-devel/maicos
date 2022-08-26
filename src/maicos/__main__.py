#!/usr/bin/env python3
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
#
# Copyright (c) 2022 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the GNU Public Licence, v3 or any higher version
# SPDX-License-Identifier: GPL-3.0-or-later
"""Analyse molecular dynamics simulation of interfacial and confined systems."""

import os

from mdacli import cli

from maicos import __version__
from maicos.core import (
    AnalysisBase,
    CylinderBase,
    PlanarBase,
    ProfileCylinderBase,
    ProfilePlanarBase,
    )
from maicos.modules import __all__


def main():
    """Execute main CLI entry point."""
    # These module are currently not supported. Either due a different
    # structure or due parameters that are not supported by our parser.
    skip_mods = ['base']
    base_cls = [AnalysisBase,
                PlanarBase,
                ProfilePlanarBase,
                CylinderBase,
                ProfileCylinderBase]
    available_mods = [f"maicos.modules.{m}" for m in __all__]
    if os.path.isfile(os.path.join(os.path.expanduser("~"),
                                   ".maicos",
                                   "maicos_custom_modules.py")):
        available_mods += ["maicos_custom_modules"]

    cli(name="MAICoS",
        module_list=available_mods,
        base_class=base_cls,
        version=__version__,
        description=__doc__,
        skip_modules=skip_mods,
        ignore_warnings=True)


if __name__ == '__main__':
    main()
