#!/usr/bin/env python
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
#
# Copyright (c) 2019 Authors and contributors
# (see the file AUTHORS for the full list of names)
#
# Released under the GNU Public Licence, v2 or any higher version
# SPDX-License-Identifier: GPL-2.0-or-later

from pkg_resources import resource_filename

WATER_GRO = resource_filename(__name__, "../data/water/confout.gro")
WATER_TRR = resource_filename(__name__, "../data/water/traj.trr")
WATER_TPR = resource_filename(__name__, "../data/water/topol.tpr")
