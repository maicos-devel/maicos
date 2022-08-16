#!/usr/bin/env python3
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
#
# Copyright (c) 2022 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the GNU Public Licence, v3 or any higher version
# SPDX-License-Identifier: GPL-3.0-or-later

__all__ = [
    'ChemicalPotentialPlanar',
    'TemperaturePlanar',
    'DensityPlanar',
    'DensityCylinder',
    'EpsilonPlanar',
    'EpsilonCylinder',
    'DielectricSpectrum',
    'Saxs',
    'Diporder',
    'DipoleAngle',
    'KineticEnergy',
    'Velocity',
    'RDFPlanar',
    ]

import os
import sys
import warnings

from .modules.density import (
    ChemicalPotentialPlanar,
    DensityCylinder,
    DensityPlanar,
    TemperaturePlanar,
    )
from .modules.epsilon import DielectricSpectrum, EpsilonCylinder, EpsilonPlanar
from .modules.structure import Diporder, RDFPlanar, Saxs
from .modules.timeseries import DipoleAngle, KineticEnergy
from .modules.transport import Velocity
from .version import __version__


try:
    sys.path.append(os.path.join(os.path.expanduser("~"),
                                 ".maicos/"))
    from maicos_custom_modules import *
    __all__ += custom_modules
except ImportError:
    pass

__authors__ = "Philip Loche et. al."

# Print maicos DeprecationWarnings
warnings.filterwarnings(action='once', category=DeprecationWarning, module='maicos')
