#!/usr/bin/env python
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
#
# Copyright (c) 2023 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the GNU Public Licence, v3 or any higher version
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for main."""

import subprocess
import sys

import pytest

from maicos import __all__ as available_modules


@pytest.mark.skipif(
    sys.platform == "win32", reason="CLI not yet available on platform Windows"
)
class Test_parse_args(object):
    """Tests for the parse argurment."""

    def test_required_args(self):
        """Test required arguments."""
        with pytest.raises(subprocess.CalledProcessError):
            subprocess.check_call(["maicos"])

    def test_wrong_module(self):
        """Test wrong module."""
        with pytest.raises(subprocess.CalledProcessError):
            subprocess.check_call(["maicos", "foo"])

    @pytest.mark.parametrize("module", tuple(available_modules))
    def test_available_modules(self, module):
        """Test available modules."""
        subprocess.check_call(["maicos", module, "--help"])

    @pytest.mark.parametrize("args", ("version", "help"))
    def test_extra_options(self, args):
        """Test extra options."""
        subprocess.check_call(["maicos", "--" + args])
