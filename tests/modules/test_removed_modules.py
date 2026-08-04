# Copyright (c) 2026 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the GNU Public Licence, v3 or any higher version
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for modules that have been removed or moved to other packages."""

import sys
from pathlib import Path

import MDAnalysis as mda
import pytest

sys.path.append(str(Path(__file__).parents[1]))
from data import SPCE_GRO, SPCE_ITP


@pytest.fixture
def ag():
    """A minimal AtomGroup for testing."""
    return mda.Universe(SPCE_ITP, SPCE_GRO, topology_format="itp").atoms


def test_dielectric_spectrum_removed(ag):
    """Test that DielectricSpectrum raises ValueError since it was moved."""
    from maicos.modules.dielectricspectrum import DielectricSpectrum

    with pytest.raises(ValueError, match="spectrakit"):
        DielectricSpectrum(ag)


def test_dipole_angle_removed(ag):
    """Test that DipoleAngle raises ValueError since it was removed."""
    from maicos.modules.dipoleangle import DipoleAngle

    with pytest.raises(ValueError, match="no longer available"):
        DipoleAngle(ag)


def test_diporder_structure_factor_removed(ag):
    """Test that DiporderStructureFactor raises ValueError since it was moved."""
    from maicos.modules.diporderstructurefactor import DiporderStructureFactor

    with pytest.raises(ValueError, match="scatterkit"):
        DiporderStructureFactor(ag)


def test_kinetic_energy_removed(ag):
    """Test that KineticEnergy raises ValueError since it was removed."""
    from maicos.modules.kineticenergy import KineticEnergy

    with pytest.raises(ValueError, match="no longer available"):
        KineticEnergy(ag)


def test_rdf_diporder_removed(ag):
    """Test that RDFDiporder raises ValueError since it was moved."""
    from maicos.modules.rdfdiporder import RDFDiporder

    with pytest.raises(ValueError, match="scatterkit"):
        RDFDiporder(ag)


def test_saxs_removed(ag):
    """Test that Saxs raises ValueError since it was moved."""
    from maicos.modules.saxs import Saxs

    with pytest.raises(ValueError, match="scatterkit"):
        Saxs(ag)
