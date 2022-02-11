#!/usr/bin/env python3
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
#
# Copyright (c) 2021 Authors and contributors
# (see the file AUTHORS for the full list of names)
#
# Released under the GNU Public Licence, v2 or any higher version
# SPDX-License-Identifier: GPL-2.0-or-later

import subprocess

from maicos import __all__ as available_modules
import pytest


class Test_parse_args(object):

    def test_required_args(self):
        with pytest.raises(subprocess.CalledProcessError):
            subprocess.check_call(['maicos'])

    def test_wrong_module(self):
        with pytest.raises(subprocess.CalledProcessError):
            subprocess.check_call(['maicos', 'foo'])

    @pytest.mark.parametrize("module", tuple(available_modules))
    def test_available_modules(self, module):
        subprocess.check_call(['maicos', module, "--help"])

    @pytest.mark.parametrize('args', ("version", "help"))
    def test_extra_options(self, args):
        subprocess.check_call(['maicos', '--' + args])