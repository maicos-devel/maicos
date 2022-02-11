import MDAnalysis as mda
import pytest

from maicos import kinetic_energy
import numpy as np
from numpy.testing import assert_almost_equal

from datafiles import NVE_WATER_TPR, NVE_WATER_TRR


class Test_kinetic_energy(object):

    @pytest.fixture()
    def ag(self):
        u = mda.Universe(NVE_WATER_TPR, NVE_WATER_TRR)
        return u.atoms

    def test_ke_trans(self, ag):
        ke = kinetic_energy(ag)
        ke.run()
        assert_almost_equal(np.mean(ke.results.trans), 2156, decimal=0)

    def test_ke_rot(self, ag):
        ke = kinetic_energy(ag)
        ke.run()
        assert_almost_equal(np.mean(ke.results.rot), 2193, decimal=0)
