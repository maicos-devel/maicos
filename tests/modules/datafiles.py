#!/usr/bin/env python
# coding: utf8

from pkg_resources import resource_filename

WATER_GRO = resource_filename(__name__, "../data/water/confout.gro")
WATER_TRR = resource_filename(__name__, "../data/water/traj.trr")
WATER_TPR = resource_filename(__name__, "../data/water/topol.tpr")
