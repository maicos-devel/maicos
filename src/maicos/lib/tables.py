#!/usr/bin/env python
#
# Copyright (c) 2025 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the GNU Public Licence, v3 or any higher version
# SPDX-License-Identifier: GPL-3.0-or-later
"""The module contains static lookup tables for atom typing etc.

The tables are dictionaries that are indexed by elements.
"""

from pathlib import Path

import numpy as np

_share_path = Path(__file__).parents[1] / "share"


#: Translation of
#: :py:class:`MDAnalysis.AtomGroup.types <MDAnalysis.core.topologyattrs.Atomtypes>` to
#: chemical elements.
atomtypes = {}
with open(_share_path / "atomtypes.dat") as f:
    for line in f:
        if line[0] != "#":
            elements = line.split()
            atomtypes[elements[0]] = elements[1]

#: Cromer-Mann X-ray scattering factors computed from numerical
#: Hartree-Fock wave functions. See Acta Cryst. A 24 (1968) p. 321
CM_parameters = {}
with open(_share_path / "sfactor.dat") as f:
    for line in f:
        if line[0] != "#":
            elements = line.split()
            CM_parameters[elements[0]] = type("CM_parameter", (object,), {})()
            CM_parameters[elements[0]].a = np.array(elements[2:6], dtype=np.double)
            CM_parameters[elements[0]].b = np.array(elements[6:10], dtype=np.double)
            CM_parameters[elements[0]].c = float(elements[10])
