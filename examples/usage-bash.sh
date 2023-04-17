#!/bin/bash
# -*- Mode: bash; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
#
# Copyright (c) 2023 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the GNU Public Licence, v3 or any higher version
# SPDX-License-Identifier: GPL-3.0-or-later

maicos densityplanar -s slit_flow.tpr \
                     -f slit_flow.trr \
                     -atomgroup 'type OW HW'

# The density profile has been written in a file named ``density.dat`` in the current
# directory. The written file starts with the following lines

head -n 20 density.dat 

# For lengthy analysis, use the ``concfreq`` option to update the result during the run

maicos densityplanar -s slit_flow.tpr \
                    -f slit_flow.trr \
                    -atomgroup 'type OW HW' \
                    -concfreq '10'

# The general help of MAICoS can be accessed using

maicos -h

# Package-specific page can also be accessed from the cli

maicos densityplanar -h
