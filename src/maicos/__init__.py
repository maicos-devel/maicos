#!/usr/bin/env python
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
#
# Copyright (c) 2023 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the GNU Public Licence, v3 or any higher version
# SPDX-License-Identifier: GPL-3.0-or-later
import sys
import warnings
from pathlib import Path

from ._version import get_versions
from .modules import *
from .modules import __all__


try:
    sys.path.append(str(Path().home() / ".maicos/"))
    from maicos_custom_modules import *
    from maicos_custom_modules import custom_modules

    __all__ += custom_modules
except ImportError:
    pass

__authors__ = "MAICoS Developer Team"
#: Version information for MAICoS, following :pep:`440`
#: and `semantic versioning <http://semver.org/>`_.
__version__ = get_versions()["version"]
del get_versions

# Print maicos DeprecationWarnings
warnings.filterwarnings(action="once", category=DeprecationWarning, module="maicos")
