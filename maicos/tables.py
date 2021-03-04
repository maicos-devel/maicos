#!/usr/bin/env python3
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
#
# Copyright (c) 2020 Authors and contributors
# (see the file AUTHORS for the full list of names)
#
# Released under the GNU Public Licence, v2 or any higher version
# SPDX-License-Identifier: GPL-2.0-or-later

import os

import numpy as np

_share_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "share")

# Translation of atomnames to types/element
atomtypes = {}
with open(os.path.join(_share_path, "atomtypes.dat")) as f:
    for line in f:
        if line[0] != '#':
            elements = line.split()
            atomtypes[elements[0]] = elements[1]

# Cromer-Mann X-ray scattering factors computed from numerical Hartree-Fock wave functions
# See Acta Cryst. A 24 (1968) p. 321
CM_parameters = {}
with open(os.path.join(_share_path, "sfactor.dat")) as f:
    for line in f:
        if line[0] != '#':
            elements = line.split()
            CM_parameters[elements[0]] = type('CM_parameter', (object,), {})()
            CM_parameters[elements[0]].a = np.array(
                elements[2:6], dtype=np.double)
            CM_parameters[elements[0]].b = np.array(
                elements[6:10], dtype=np.double)
            CM_parameters[elements[0]].c = float(elements[10])
