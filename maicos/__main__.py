#!/usr/bin/env python3
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
#
# Copyright (c) 2019 Authors and contributors
# (see the file AUTHORS for the full list of names)
#
# Released under the GNU Public Licence, v2 or any higher version
# SPDX-License-Identifier: GPL-2.0-or-later
"""
    Analyse molecular dynamics simulations of interfacial and confined systems.
"""
import os

from maicos import __version__
from maicos.modules import __all__
from maicos.modules.base import AnalysisBase, PlanarBase

from mdacli import cli

def main():
    """Execute main CLI entry point."""
    # Thse module are currently not supported. Either due a different
    # structure or due parameters that aree not supported by our parser.
    skip_mods = ['base']
    balse_cls = [AnalysisBase, PlanarBase]
    available_mods = [f"maicos.modules.{m}" for m in __all__]
    if os.path.isfile(os.path.join(os.path.expanduser("~"),
                                   ".maicos",
                                   "maicos_costum_modules.py")):
        available_mods += ["maicos_costum_modules"]


    cli(name="MAICoS",
        module_list=available_mods,
        base_class=balse_cls,
        version=__version__,
        description=__doc__,
        skip_modules=skip_mods,
        ignore_warnings=True)


if __name__ == '__main__':
    main()
