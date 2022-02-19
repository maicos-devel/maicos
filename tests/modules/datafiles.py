#!/usr/bin/env python
"""Import datafiles."""
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
#
# Copyright (c) 2022 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the GNU Public Licence, v3 or any higher version
# SPDX-License-Identifier: GPL-3.0-or-later

from pkg_resources import resource_filename


# bulk water
WATER_GRO = resource_filename(__name__, "../data/water/confout.gro")
WATER_TRR = resource_filename(__name__, "../data/water/traj.trr")
WATER_TPR = resource_filename(__name__, "../data/water/topol.tpr")

# air-water interface
AIRWATER_GRO = resource_filename(__name__, "../data/airwater/confout.gro")
AIRWATER_TRR = resource_filename(__name__, "../data/airwater/traj.trr")
AIRWATER_TPR = resource_filename(__name__, "../data/airwater/topol.tpr")

SALT_WATER_GRO = resource_filename(
    __name__, "../data/salt_water/salt_water.gro")
SALT_WATER_TPR = resource_filename(
    __name__, "../data/salt_water/salt_water.tpr")

# NVE bulk water
NVE_WATER_TPR = resource_filename(__name__, "../data/kineticenergy/nve.tpr")
NVE_WATER_TRR = resource_filename(__name__, "../data/kineticenergy/nve.trr")

LAMMPS10WATER = resource_filename(__name__, "../data/lammps10water.data")
