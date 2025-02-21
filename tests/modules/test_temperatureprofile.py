#!/usr/bin/env python
#
# Copyright (c) 2025 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the GNU Public Licence, v3 or any higher version
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for the TemperaturePlanar class."""

import pytest
from numpy.testing import assert_allclose
from test_densityplanar import ReferenceAtomGroups

from maicos import TemperaturePlanar


class TestTemperaturProfile(ReferenceAtomGroups):
    """Tests for the TemperaturePlanar class."""

    @pytest.mark.parametrize("dim", [0, 1, 2])
    def test_dens(self, ag, dim):
        """Test mean temperature."""
        dens = TemperaturePlanar(ag, dim=dim).run()
        assert_allclose(dens.results.profile.mean(), 291.6, rtol=1e1)

    def test_wrong_grouping(self, ag):
        """Test a wrong grouping choice."""
        with pytest.raises(ValueError, match="Invalid choice"):
            TemperaturePlanar(ag, grouping="molecules").run()
