# functions for calculating the polarization of an MD trajectory from a
# selection of atoms from an MDAnalysis universe.
# by Shane Carlson

import numpy as np

from ..utils import repairMolecules

def TotalPolarizationInst(selection):

    repairMolecules(selection)  # reunites molecules that straddle PB's

    Pinst = np.zeros(3)

    for atm in selection.atoms:
        Pinst += atm.charge * atm.position

    # divide by 10, since the distances given by MDAnlaysis are in Angstroms, and we prefer nm
    return Pinst / 10


def TotalPolarizationTraj(selection):

    # Polarization vectors for all timesteps
    Ptraj = np.zeros((len(selection.trajectory), 3))

    for ts in selection.trajectory:
        Ptraj[ts.frame, :] = TotalPolarizationInst(selection)

    return Ptraj
