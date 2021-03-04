#!/usr/bin/env python3
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
#
# Copyright (c) 2020 Authors and contributors
# (see the file AUTHORS for the full list of names)
#
# Released under the GNU Public Licence, v2 or any higher version
# SPDX-License-Identifier: GPL-2.0-or-later

import warnings
import sys
import os

from .modules import *

__all__ = [
    'density_planar',
    'density_cylinder',
    'epsilon_bulk',
    'epsilon_planar',
    'epsilon_cylinder',
    'dielectric_spectrum',
    'dielectric_spectrum_ion',
    'saxs',
    'debye',
    'dipole_angle',
    'kinetic_energy',
    'dipole_trajectory',
    'diporder',
    'velocity',
]

try:
    sys.path.append(os.path.join(os.path.expanduser("~"),
                                 ".maicos/"))
    from maicos_costum_modules import *
    __all__ += custom_modules
except ImportError:
    pass

__authors__ = "Philip Loche et. al."
__version__ = "0.4-dev"  # NOTE: keep in sync with VERSION in setup.py

# Print maicos DeprecationWarnings
warnings.filterwarnings(action='once', category=DeprecationWarning, module='maicos')