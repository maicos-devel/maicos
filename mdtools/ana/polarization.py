# functions for calculating the polarization of an MD trajectory from a 
# selection of atoms from an MDAnalysis universe.
# by Shane Carlson

import numpy as np

def TotalPolarizationInst(selection):
    
    pbctools.repairMolecules(selection) # reunites molecules that straddle PB's

    Pinst = np.zeros(3)

    for atm in selection.atoms:
        Pinst += atm.charge*atm.position
    
    return Pinst/10 #divide by 10, since the distances given by MDAnlaysis are in Angstroms, and we prefer nm

def TotalPolarizationTraj(selection):

    Ptraj = np.zeros((len(selection.trajectory), 3)) # Polarization vectors for all timesteps
    
    for ts in selection.trajectory:
        Ptraj[ts.frame, :] = TotalPolarizationInst(selection)
    
    return Ptraj