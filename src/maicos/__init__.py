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
    'DensitySphere',
    'DielectricPlanar',
    'DielectricCylinder',
    'DielectricSphere',
    'DielectricSpectrum',
    'Saxs',
    'Diporder',
    'DipoleAngle',
    'KineticEnergy',
    'VelocityPlanar',
    'VelocityCylinder',
    'RDFPlanar',
    ]

import os
import sys
import warnings

from ._version import get_versions
from .modules.density import (
    ChemicalPotentialPlanar,
    DensityCylinder,
    DensityPlanar,
    DensitySphere,
    TemperaturePlanar,
    )
from .modules.dielectric import (
    DielectricCylinder,
    DielectricPlanar,
    DielectricSpectrum,
    DielectricSphere,
    )
from .modules.structure import Diporder, RDFPlanar, Saxs
from .modules.timeseries import DipoleAngle, KineticEnergy
from .modules.transport import VelocityCylinder, VelocityPlanar


try:
    sys.path.append(os.path.join(os.path.expanduser("~"),
                                 ".maicos/"))
    from maicos_custom_modules import *
    __all__ += custom_modules
except ImportError:
    pass

__authors__ = "MAICoS Developer Team"
 #: Version information for MAICoS, following :pep:`440`
 #: and `semantic versioning <http://semver.org/>`_.
__version__ = get_versions()['version']
del get_versions

# Print maicos DeprecationWarnings
warnings.filterwarnings(action='once', category=DeprecationWarning, module='maicos')

from . import _version


__version__ = _version.get_versions()['version']
