#!/usr/bin/env python
#
# Copyright (c) 2025 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the GNU Public Licence, v3 or any higher version
# SPDX-License-Identifier: GPL-3.0-or-later
"""MAICoS: Molecular Analysis of Interfacial and COnfined Systems."""

import warnings

from ._version import get_versions
from .modules import *  # noqa: F403
from .modules import __all__ as __all__

__authors__ = "MAICoS Developer Team"
#: Version information for MAICoS, following :pep:`440`
#: and `semantic versioning <http://semver.org/>`_.
__version__ = get_versions()["version"]
del get_versions

# Print maicos DeprecationWarnings
warnings.filterwarnings(action="once", category=DeprecationWarning, module="maicos")
