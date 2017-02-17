#!/usr/bin/env python

from __future__ import division

import numpy as np
import MDAnalysis
import sys

from mdanahelper import pbctools

# parse command line options
import argparse
parser = argparse.ArgumentParser(description="Calculate dipolar order parameters.", \
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-s', dest='topology', type=str,
        default='topol.tpr')
parser.add_argument('-f', dest='trajectory', type=str, nargs='+',
        default=['traj_comp.xtc'])
parser.add_argument('-dz', dest='binwidth', type=float,\
        default=0.01, help='specify the binwidth [nm]')
parser.add_argument('-dt', dest='skipframes', type=int,\
        default=1, help='skip every N frames')
parser.add_argument('-b', dest='begin', type=float,\
        default=0, help='starting time for evaluation')
parser.add_argument('-e', dest='end', type=float,\
        default=-1, help='ending time for evaluation')
parser.add_argument('-d', dest='dim', type=int,\
        default=2, help='direction normal to the surface (x,y,z=0,1,2, default: z)')
parser.add_argument('-o', dest='output', type=str,\
        default='diporder', help='Prefix for output filenames')
parser.add_argument('-sel', dest='sel', type=str,
                   help='atom group selection',
                   default='resname SOL')
parser.add_argument('-dout', dest='outfreq', type=float,\
        default='10000', help='Default number of frames after which output files are refreshed (10000)')
parser.add_argument('-sym', dest='bsym', action='store_const',
                   const=True, default=False,
                   help='symmetrize the profiles')
parser.add_argument('-shift', dest='membrane_shift', action='store_const',
                   const=True, default=False,
                   help='shift system by half a box length (useful for membrane simulations)')
parser.add_argument('-com', dest='com', action='store_const',
                   const=True, default=False,
                   help='shift system such that the water COM is centered')
parser.add_argument('-bin', dest='binmethod', type=str,
                   default='COM',
                   help='binning method: center of Mass (COM), center of charge (COC) or oxygen position (OXY)')

def output():
    avL = Lz/frame/10

    z = np.linspace(0,avL, len(diporder)) + avL/nbins/2

    outdata = np.hstack([ z[:,np.newaxis], diporder/framecount[:,np.newaxis] ])

    if (args.bsym):
        for i in range(len(outdata)-1):
            outdata[i+1] = .5*(outdata[i+1]+outdata[i+1][-1::-1])

    np.savetxt(args.output+'.dat', outdata, header="z\tP_0 rho(z) cos(Theta(z))\tcos(Theta(z))\trho(z)")

    return


args = parser.parse_args()

u = MDAnalysis.Universe(args.topology, args.trajectory)
sol = u.select_atoms(args.sel)
atomsPerMolecule = sol.n_atoms // sol.n_residues

dim = args.dim
xydims = np.roll(np.arange(3),-dim)[1:] # Assume a threedimensional universe...
dz = args.binwidth * 10 # Convert to Angstroms
nbins=int(u.dimensions[dim]/dz) # CAVE: binwidth varies in NPT !

begin=args.begin

if (args.end != -1):
    end = args.end
else:
    end=u.trajectory.totaltime

'''
data structure:
    diporder saves the (summed) projected polarization density:
        0 P_0 rho(z) cos(Theta(z))
        1 cos(Theta(z))
        2 rho(z)
'''

diporder = np.zeros((nbins,3))
framecount = np.zeros((nbins)) # only count frames when there was a water!

Lz = 0

# unit normal vector
unit = np.zeros(3)
unit[dim] += 1

print 'Using', nbins, 'bins.'

frame = 0
print "Evaluating frame: ", u.trajectory.frame, "\ttime: ", int(u.trajectory.time), '\r',

startframe = int(begin // u.trajectory.dt)
endframe = int(end // u.trajectory.dt)

for ts in u.trajectory[startframe:endframe:args.skipframes]:

    if args.membrane_shift:
        # shift membrane
        ts.positions[:,dim] += ts.dimensions[dim]/2
        ts.positions[:,dim] %= ts.dimensions[dim]
    if args.com:
        # put water COM into center
        waterCOM = np.sum(sol.atoms.positions[:,2]*sol.atoms.masses) / sol.atoms.masses.sum()
        ts.positions[:,dim] += ts.dimensions[dim]/2 - waterCOM
        ts.positions[:,dim] %= ts.dimensions[dim]

    # make broken molecules whole again!
    pbctools.repairMolecules(u)

    A = np.prod(ts.dimensions[xydims])
    dz_frame = ts.dimensions[dim]/nbins

    chargepos = sol.atoms.positions*sol.atoms.charges[:,np.newaxis]
    dipoles = np.sum( chargepos[i::atomsPerMolecule] for i in range(atomsPerMolecule) ) / 10 # convert to e nm

    if args.binmethod == 'COM':
        # Calculate the centers of the objects ( i.e. Molecules )
        masses = np.sum( sol.atoms.masses[i::atomsPerMolecule] for i in range(atomsPerMolecule) )
        masspos = sol.atoms.positions * sol.atoms.masses[:,np.newaxis]
        coms = np.sum( masspos[i::atomsPerMolecule] for i in range(atomsPerMolecule) ) / masses[:,np.newaxis]
        bins=(coms[:,dim]/dz_frame).astype(int)
    elif args.binmethod == 'COC':
        abschargepos = sol.atoms.positions*np.abs(sol.atoms.charges)[:,np.newaxis]
        charges = np.sum( np.abs(sol.atoms.charges)[i::atomsPerMolecule] for i in range(atomsPerMolecule) )
        cocs = np.sum( abschargepos[i::atomsPerMolecule] for i in range(atomsPerMolecule) ) / charges[:,np.newaxis]
        bins=(cocs[:,dim]/dz_frame).astype(int)
    elif args.binmethod == 'OXY':
        bins=(sol.atoms.positions[::3,dim]/dz_frame).astype(int)
    else:
        raise ValueError('Unknown binning method: %s' % args.binmethod)

    bincount = np.bincount(bins, minlength=nbins)

    diporder[:,0] += np.histogram(bins, bins=np.arange(nbins+1),
            weights = np.dot(dipoles, unit))[0] / (A*dz_frame/1e3)
    with np.errstate(divide='ignore', invalid='ignore'):
        diporder[:,1] += np.nan_to_num(np.histogram(bins, bins=np.arange(nbins+1), 
            weights=np.dot(dipoles/np.linalg.norm(dipoles,axis=1)[:,np.newaxis],unit))[0] / bincount)
    diporder[:,2] += bincount / (A*dz_frame/1e3)
    framecount += bincount>0

    Lz += ts.dimensions[dim]

    if (ts.frame % 250 == 0):
        print "Evaluating frame: %12d    time: %12d\r" %(frame, int(ts.time)),
        sys.stdout.flush()
        # call for output
    if (frame % args.outfreq == 0 and frame >= args.outfreq):
        output()
    frame+=1

print '\n'
output()
