#!/usr/bin/env python
#
# Copyright (c) 2025 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the GNU Public Licence, v3 or any higher version
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for setup.py functions."""

import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).parents[1]))
from setup import detect_openmp  # noqa: E402


@pytest.mark.xfail(
    sys.platform != "linux", reason="OpenMP detection should at least work on linux"
)
def test_openmp_detection():
    """Test OpenMP detection."""
    assert detect_openmp()
