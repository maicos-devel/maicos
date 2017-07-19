#!/usr/bin/env python3

from __future__ import print_function, division
import MDAnalysis as mda
import numpy as np
from scipy.stats import binned_statistic
import sys
import argparse
import os
import multiprocessing
from sfactor import sfactor #is this fails build the sfactor libaray with 'python3 setup-sfactor.py build_ext --inplace'



#========== PARSER ===========
#=============================
parser = argparse.ArgumentParser(description="""
    Computes SAXS scattering intensities according to the Debye scattering equation
    for all atom types from the given trajectory.
    For the scattering factor the structure fator is multiplied by a atom type specific form factor
    based on Cromer-Mann parameters. By using the -sel option atoms can be selected for which the
    profile is calculated. The selection uses the MDAnalysis selection commands found here:
    http://www.mdanalysis.org/docs/documentation_pages/selections.html""",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-s',     dest='topology',    type=str,   default='topol.tpr',            help='the topolgy file')
parser.add_argument('-f',     dest='trajectory',  type=str,   default=['traj.xtc'],nargs='+', help='A single or multiple trajectory files.')
parser.add_argument('-sel',   dest='sel',         type=str,   default='all',                  help='Atoms for which to compute the profile', )
parser.add_argument('-b',     dest='begin',       type=float, default=0,                      help='First frame (ps) to read from trajectory')
parser.add_argument('-e',     dest='end',         type=float, default=None,                   help='Last frame (ps) to read from trajectory')
parser.add_argument('-skip',  dest='skipframes',  type=int,   default=1,                      help='Evaluate every Nth frames')
parser.add_argument('-dout',  dest='outfreq',     type=float, default='100',                  help='Number of frames after which the output is updated.')
parser.add_argument('-sq',    dest='output',      type=str,   default='./sq',                 help='Prefix/Path for output file')
parser.add_argument('-startq',dest='startq',      type=float, default=0,                      help='Starting q (1/nm)')
parser.add_argument('-endq',  dest='endq',        type=float, default=60,                     help='Ending q (1/nm)')
parser.add_argument('-dq',    dest='dq',          type=float, default=0.05,                   help='binwidth (1/nm)')
parser.add_argument('-nt',    dest='nt',          type=int,   default=0,                   help='Total number of threads to start (0 is guess)')

args = parser.parse_args()

#======== DEFINITIONS ========
#=============================

def output():
    """Saves the current profiles to a file."""
    nonzeros = np.where(struct_factor != 0)[0]
    scat_factor = struct_factor[nonzeros]
    wave_vectors = q[nonzeros]

    scat_factor = scat_factor/frames

    np.savetxt(args.output+'.dat',
        np.vstack([wave_vectors,scat_factor]).T,
        header="q (1/nm)\tS(q)_tot (arb. units)",fmt='%.8e')

def get_base_path():
    return os.path.dirname(os.path.realpath(sys.argv[0]))

type_dict = {}
with open("{}/share/atomtypes.dat".format(get_base_path())) as f:
    for line in f:
        if line[0] != '#':
            elements = line.split()
            type_dict[elements[0]] = elements[1]

CM_parameters = {};
with open("{}/share/sfactor.dat".format(get_base_path())) as f:
    for line in f:
        if line[0] != '#':
            elements = line.split();
            CM_parameters[elements[0]] = np.array(elements[2:11], dtype=np.double);

#======= PREPERATIONS =======
#============================
print('Command line was: %s\n' % ' '.join(sys.argv))

print("Loading trajectory...\n")
u = mda.Universe(args.topology,args.trajectory)
sel = u.select_atoms(args.sel)
sel = sel.atoms.select_atoms("not name 'DUM'")
n_atoms = sel.atoms.n_atoms

types = np.unique(sel.types.astype(str))
CMFP = np.zeros((len(types),9), dtype=np.float32)
nh = np.zeros(len(types), dtype=np.int32) #number of hydrogens for united atom force fields

#initialize arrays for later calculation in the cython code
dist_mat = np.zeros((n_atoms, n_atoms), dtype=np.float32);
form_factors = np.zeros(len(types), dtype=np.float32);

for i,atom_type in enumerate(types):
    try:
        element = type_dict[atom_type]
    except KeyError:
        sys.exit("No suitable element for '{0}' found. You can add '{0}' together with a suitable element to 'share/atomtypes.dat'.".format(atom_type))
    if element != "DUM":
        if element in ["CH1","CH2","CH3","CH4","NH","NH2","NH3"]:
            CMFP[i] = CM_parameters[element[0]]
            nh[i] = element[-1]
        else:
            CMFP[i] = CM_parameters[element]

indices = np.zeros(n_atoms, dtype=np.int32)
for i,atom_type in enumerate(sel.types.astype(str)):
    indices[i] = np.where(atom_type == types)[0][0]

dt = u.trajectory.dt

begin=int(args.begin // dt)
if args.end != None:
    end = int(args.end // dt)
else:
    end=int(u.trajectory.totaltime // u.trajectory.dt)

if begin > end:
    print("Start time is larger than end time!")

if args.nt == 0: args.nt = multiprocessing.cpu_count()

nbins = int(np.ceil((args.endq - args.startq)/args.dq))
q = np.arange(args.startq,args.endq,args.dq) + 0.5*args.dq
struct_factor = np.zeros(nbins)
frames = 0

#======== MAIN LOOP =========
#============================
for ts in u.trajectory[begin:end+1:args.skipframes]:

    box = np.diag(mda.lib.mdamath.triclinic_vectors(ts.dimensions))

    q_ts, S_ts = sfactor.compute_scattering_intensity(
                                    sel.atoms.positions/10, n_atoms,
                                    indices, CMFP, nh, box/10,
                                    dist_mat, form_factors,
                                    args.startq, args.endq, args.nt)

    q_ts = np.asarray(q_ts).flatten()
    S_ts = np.asarray(S_ts).flatten()
    nonzeros = np.where(S_ts != 0)[0]

    q_ts = q_ts[nonzeros]
    S_ts = S_ts[nonzeros]

    struct_ts = binned_statistic(q_ts,S_ts,bins=nbins,range=(args.startq,args.endq))[0]
    struct_factor[:] += np.nan_to_num(struct_ts)

    frames += 1
    if (frames < 100):
        print ("\rEvaluating frame: {:>12} time: {:>12} ps".format(frames, round(ts.time)), end="")
    elif (frames < 1000 and frames % 10 == 1):
        print ("\rEvaluating frame: {:>12} time: {:>12} ps".format(frames, round(ts.time)), end="")
    elif (frames % 250 == 1):
        print ("\rEvaluating frame: {:>12} time: {:>12} ps".format(frames, round(ts.time)), end="")
    # call for output
    if ( frames % args.outfreq == 0 ):
        output()
    sys.stdout.flush()

output()
print("\n")
