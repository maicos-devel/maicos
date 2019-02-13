#!/usr/bin/env python
# coding: utf-8

from __future__ import absolute_import, division, print_function

import os

import numpy as np

from .utils import _share_path

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
