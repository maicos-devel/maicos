#!/usr/bin/env python

from __future__ import division
from __future__ import print_function
import MDAnalysis as mda
import numpy as np
import sys
import argparse
import os

#========== PARSER ===========
#=============================
parser = argparse.ArgumentParser(description="""
    Computes partial densities or tempertaure profiles across the box.
    For group selections use strings in the MDAnalysis selection command style
    found here:
    https://pythonhosted.org/MDAnalysis/documentation_pages/selections.html""",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-s', dest='topology', type=str,
    default='topol.tpr',help="the topolgy file")
parser.add_argument('-f', dest='trajectory', type=str, nargs='+',
        default=['traj.xtc'], help="A single or multiple trajectory files.")

parser.add_argument('-b', dest='begin', type=float,
    default=0, help='start time (ps) for evaluation')
parser.add_argument('-e', dest='end', type=float,
    default=None, help='end time (ps) for evaluation')
parser.add_argument('-dt', dest='skipframes', type=int,
    default=1, help='skip every N frames')
parser.add_argument('-o', dest='output', type=str,
    default='density', help='Prefix for output filenames')
parser.add_argument('-dout', dest='outfreq', type=float,\
        default='1000', help='Default time after which output files are refreshed (1000 ps).')


parser.add_argument('-d', dest='dim', type=int,\
    help='dimension for binning (0=X, 1=Y, 2=Z)', default=2)

parser.add_argument('-dz', dest='binwidth', type=float,\
    default=1, help='binwidth (Angstrom)')

parser.add_argument('-muo', dest='muout', type=str,\
    default='dens', help='Prefix for output filename for chemical potential')
parser.add_argument('-temp', dest='temperature', type=float,\
        default=300, help='temperature (K) for chemical potential')
parser.add_argument('-zpos', dest='zpos', type=float,
    default=None, help='position at which the chemical potential will be computed. By default average over box.')

parser.add_argument('-dens', dest='density', type=str,
                   help='Density: mass, number, charge, temp', default='mass')
parser.add_argument('-gr', dest='groups', type=str, nargs='+',
                   help='Atoms for which to compute the density profile', default=['resname SOL'])

#======== DEFINITIONS ========
#=============================

args = parser.parse_args()

def output():
    """Saves the current profiles to a file."""

    dens_mean = density_mean / frames
    dens_mean_sq = density_mean_sq / frames

    dens_std = np.nan_to_num(np.sqrt(dens_mean_sq-dens_mean**2))
    dens_err = dens_std/np.sqrt(frames)

    dz = av_box_length / (frames*nbins)
    z = np.linspace(0, av_box_length/frames, nbins, endpoint=False)+dz/2

    #write header
    if args.density == "mass":
        units = "kg m^(-3)"
    elif args.density == "number":
        units = "nm^(-3)"
    elif args.density == "charge":
        units = "e nm^(-3)"
    elif args.density == "temp":
        units = "K"

    if args.density == 'temp':
        columns  = "temperature profile [%s]" % (units)
    else:
        columns  = "%s density profile [%s]" % (args.density,units)
    columns += "\nstatistics over %d picoseconds \npositions [nm]" % ((end-begin+1)*dt)
    for group in args.groups:
        columns +="\t"+group
    for group in args.groups:
        columns +="\t"+group+" error"

    # save density profile
    np.savetxt(args.output+'.dat',
        np.hstack(((z[:,np.newaxis])/10,dens_mean,dens_err)),
        header=columns)

    # save chemcial potential
    if (args.zpos != None):
        this = (args.zpos / av_box_size*nbins).astype(int)
        np.savetxt(args.muout+'.dat',
            np.hstack((mu(dens_mean[this]), dmu(dens_mean[this], dens_err[this])))[None])
    else:
        np.savetxt(args.muout+'.dat',
            np.array((np.mean(mu(dens_mean)), np.mean(dmu(dens_mean, dens_err))))[None])

def weight(selection):
    """Calculates the weights for the histogram depending on the choosen type of density."""
    if args.density == "mass":
        # amu in kg -> kg/m^3
        return selection.atoms.masses*1.66053892
    elif args.density == "number":
        return np.ones(selection.atoms.n_atoms)
    elif args.density == "charge":
        return selection.atoms.charges
    elif args.density == "temp":
        # ((1 amu * (Angstrom^2)) / (picoseconds^2)) / Boltzmann constant = 1.20272362 Kelvin
        return ((selection.atoms.velocities**2).sum(axis=1)*selection.atoms.masses/2*1.20272362)

def mu(rho):
  #db = 1.00394e-1  # De Broglie (converted to nm)
  mu = kT*np.log(rho)
  return mu

def dmu(rho, drho):
  return (kT/rho*drho)


#======= PREPERATIONS =======
#============================
args = parser.parse_args()
if      args.density !='mass' and args.density != 'number' \
    and args.density !='charge' and args.density != 'temp':
    parser.error('Unknown density type: valid are mass, number, charge, temp')

dim = args.dim
if args.density == 'temp':
    print ('Computing temperature profile along %s-axes.' % ('XYZ'[dim]))
else:
    print ('Computing %s density profile along %s-axes.' % (args.density,'XYZ'[dim]))

print("Loading trajectory...")
u = mda.Universe(args.topology,args.trajectory)

dt = u.trajectory.dt
kT = 0.00831446215*args.temperature

begin=int(args.begin // dt)
if args.end != None:
    end = int(args.end // dt)
else:
    end = int(u.trajectory.totaltime // dt)

if begin > end:
    sys.exit("Start time is larger than end time!")

ngroups = len(args.groups)
nbins = int(np.ceil(u.dimensions[dim]/args.binwidth))

density_mean = np.zeros(( nbins, ngroups ))
density_mean_sq = np.zeros(( nbins, ngroups ))
frames = 0
av_box_length = 0

print("\nCalcualate profile for the following group(s):")
sel = []
for i,gr in enumerate(args.groups):
    sel.append(u.select_atoms(gr))
    print("%s: %i atoms" % (gr,sel[i].n_atoms))


print('\nUsing', nbins, 'bins.')

#======== MAIN LOOP =========
#============================
for ts in u.trajectory[begin:end+1:args.skipframes]:
    curV = u.dimensions[:3].prod()/1000
    av_box_length += u.dimensions[dim]

    for index,selection in enumerate(sel):
        bins = ( selection.atoms.positions[:,dim] / (u.dimensions[dim]/nbins) ).astype(int)%nbins
        density_ts = np.histogram(bins, bins=np.arange(nbins+1), weights=weight(selection))[0]

        bincount = np.bincount(bins, minlength=nbins)

        if args.density == 'temp':
            density_mean[:,index] += density_ts/bincount
            density_mean_sq[:,index] += (density_ts/bincount)**2
        else:
            density_mean[:,index] += density_ts/curV*nbins
            density_mean_sq[:,index] += (density_ts/curV*nbins)**2



    frames += 1
    if (frames < 100):
        print ("\rEvaluating frame: %12d    time: %12d ps" %(ts.frame, int(ts.time)), end="")
    elif (frames < 1000 and frames % 10 == 1):
        print ("\rEvaluating frame: %12d    time: %12d ps" %(ts.frame, int(ts.time)), end="")
    elif (frames % 250 == 1):
        print ("\rEvaluating frame: %12d    time: %12d ps" %(ts.frame, int(ts.time)), end="")
    # call for output
    if ( int(ts.time) % args.outfreq == 0 and ts.time-args.begin >= args.outfreq):
        output()
    sys.stdout.flush()

output()
print("\n")
