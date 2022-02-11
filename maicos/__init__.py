#!/usr/bin/env python3
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
#
# Copyright (c) 2019 Authors and contributors
# (see the file AUTHORS for the full list of names)
#
# Released under the GNU Public Licence, v2 or any higher version
# SPDX-License-Identifier: GPL-2.0-or-later

__all__ = [
    'DensityPlanar',
    'DensityCylinder',
    'EpsilonBulk',
    'EpsilonPlanar',
    'EpsilonCylinder',
    'DielectricSpectrum',
    'Saxs',
    'Debye',
    'Diporder',
    'DipoleAngle',
    'KineticEnergy',
    'Velocity',
]

import warnings
import sys
import os

from .modules.density import DensityPlanar, DensityCylinder
from .modules.epsilon import (
    EpsilonBulk,
    EpsilonPlanar,
    EpsilonCylinder,
    DielectricSpectrum
)
from .modules.structure import Saxs, Debye, Diporder
from .modules.timeseries import DipoleAngle, KineticEnergy
from .modules.transport import Velocity

try:
    sys.path.append(os.path.join(os.path.expanduser("~"),
                                 ".maicos/"))
    from maicos_costum_modules import *
    __all__ += custom_modules
except ImportError:
    pass

__authors__ = "Philip Loche et. al."
# NOTE: keep in sync with VERSION in setup.py
# NOTE: keep in sync with version in docs/source/conf.py
__version__ = "0.5-dev"

# Print maicos DeprecationWarnings
warnings.filterwarnings(action='once', category=DeprecationWarning, module='maicos')
