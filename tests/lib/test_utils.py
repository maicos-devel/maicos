#!/usr/bin/env python3
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
#
# Copyright (c) 2020 Authors and contributors
# (see the file AUTHORS for the full list of names)
#
# Released under the GNU Public Licence, v2 or any higher version
# SPDX-License-Identifier: GPL-2.0-or-later

import sys
from unittest.mock import patch

from MDAnalysisTests.core.util import UnWrapUniverse
import maicos.lib.utils


def test_check_compound():
    u = UnWrapUniverse()
    assert maicos.lib.utils.check_compound(u.atoms) == "molecules"

    u = UnWrapUniverse(have_molnums=False, have_bonds=True)
    assert maicos.lib.utils.check_compound(u.atoms) == "fragments"

    u = UnWrapUniverse(have_molnums=False, have_bonds=False)
    assert maicos.lib.utils.check_compound(u.atoms) == "residues"


def test_get_cli_input():
    testargs = ['maicos', 'foo', "foo bar"]
    with patch.object(sys, 'argv', testargs):
        assert maicos.lib.utils.get_cli_input() == 'Command line was: maicos foo "foo bar"'
