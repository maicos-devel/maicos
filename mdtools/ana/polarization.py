# functions for calculating the polarization of an MD trajectory from an MDAnalysis universe -- by Shane Carlson

import numpy as np

def TotalPolarizationInst(selection):
    
    Pinst = np.zeros(3)

    for atm in selection.atoms:
        Pinst += atm.charge*atm.position
    
    return Pinst

def TotalPolarizationTraj(selection):

    Ptraj = np.zeros((len(selection.trajectory), 3)) # Polarization vectors for all timesteps
    
    for ts in selection.trajectory:
        Ptraj[ts.frame, :] = TotalPolarizationInst(selection)
    
    return Ptraj