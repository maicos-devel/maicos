#!/usr/bin/env python
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
#
# Copyright (c) 2019 Authors and contributors
# (see the file AUTHORS for the full list of names)
#
# Released under the GNU Public Licence, v2 or any higher version
# SPDX-License-Identifier: GPL-2.0-or-later

from pkg_resources import resource_filename

# bulk water
WATER_GRO = resource_filename(__name__, "../data/water/confout.gro")
WATER_TRR = resource_filename(__name__, "../data/water/traj.trr")
WATER_TPR = resource_filename(__name__, "../data/water/topol.tpr")

# air-water interface
AIRWATER_GRO = resource_filename(__name__, "../data/airwater/confout.gro")
AIRWATER_TRR = resource_filename(__name__, "../data/airwater/traj.trr")
AIRWATER_TPR = resource_filename(__name__, "../data/airwater/topol.tpr")

# NaCl solution
SALT_GRO = resource_filename(__name__, "../data/salt/confout.gro")
SALT_TRR = resource_filename(__name__, "../data/salt/traj.xtc")
SALT_TPR = resource_filename(__name__, "../data/salt/topol.tpr")
