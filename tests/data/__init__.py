#!/usr/bin/env python3
"""init file for datafiles."""
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
#
# Copyright (c) 2022 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the GNU Public Licence, v3 or any higher version
# SPDX-License-Identifier: GPL-3.0-or-later

from pkg_resources import resource_filename


# bulk water
WATER_GRO = resource_filename(__name__, "../../docs/static/water/water.gro")
WATER_TRR = resource_filename(__name__, "../../docs/static/water/water.trr")
WATER_TPR = resource_filename(__name__, "../../docs/static/water/water.tpr")

# air-water interface
AIRWATER_GRO = resource_filename(__name__, "airwater/airwater.gro")
AIRWATER_TRR = resource_filename(__name__, "airwater/airwater.trr")
AIRWATER_TPR = resource_filename(__name__, "airwater/airwater.tpr")

SALT_WATER_GRO = resource_filename(__name__, "salt_water/salt_water.gro")
SALT_WATER_TPR = resource_filename(__name__, "salt_water/salt_water.tpr")

# NVE bulk water
NVE_WATER_TPR = resource_filename(__name__, "kineticenergy/nve.tpr")
NVE_WATER_TRR = resource_filename(__name__, "kineticenergy/nve.trr")

# MICA slab
MICA_XTC = resource_filename(__name__, "mica/mica.xtc")
MICA_TPR = resource_filename(__name__, "mica/mica.tpr")
MICA_GRO = resource_filename(__name__, "mica/mica.gro")

LAMMPS10WATER = resource_filename(__name__, "lammps10water.data")
SPCE_ITP = resource_filename(__name__, 'spce.itp')
SPCE_GRO = resource_filename(__name__, 'spce.gro')

DIPOLE_ITP = resource_filename(__name__, 'dipole.itp')
DIPOLE_GRO = resource_filename(__name__, 'dipole.gro')
