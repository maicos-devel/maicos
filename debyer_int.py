#!/usr/bin/env python3

from __future__ import print_function, division
import MDAnalysis as mda
import numpy as np
import argparse
import os
import sys
import subprocess


#========== PARSER ===========
#=============================
parser = argparse.ArgumentParser(description="""
    An interface to the Deyer library. By using the -sel option atoms can be selected for which the
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
parser.add_argument('-startq',dest='startq',      type=float, default=0,                      help='Starting q (1/A)')
parser.add_argument('-endq',  dest='endq',        type=float, default=60,                     help='Ending q (1/A)')
parser.add_argument('-dq',    dest='dq',          type=float, default=0.02,                   help='binwidth (1/A)')
parser.add_argument('-d',     dest='debyer',      type=str,   default="~/repos/debyer/debyer/debyer", help='path to the debyer executable')


args = parser.parse_args()


def cleanup():
    """averages over all dat file and removes them"""
    datfiles = [f for f in os.listdir("tmp") if f.endswith(".dat")]
    xyzfiles = [f for f in os.listdir("tmp") if f.endswith(".xyz")]

    for i, f in enumerate(datfiles):
        path = "tmp/{}".format(f)
        if i == 0:
            s_tmp = np.loadtxt(path)
        else:
            s_tmp[:,1] += np.loadtxt(path)[:,1]

        os.remove(path)
        os.remove("tmp/{}".format(xyzfiles[i]))

    s_tmp[:,1] /= len(datfiles)

    if frames > args.outfreq:
        s_tmp[:,1] *= len(datfiles)
        s_tmp[:,1] += (frames-len(datfiles))*np.loadtxt("tmp/{}".format(f))[:,1]

    s_tmp /= frames
    np.savetxt(args.output+'.dat',s_tmp,header="q (1/A)\tS(q)_tot (arb. units)",fmt='%.8e')

def get_base_path():
    return os.path.dirname(os.path.realpath(sys.argv[0]))

type_dict = {}
with open("{}/share/atomtypes.dat".format(get_base_path())) as f:
    for line in f:
        if line[0] != '#':
            elements = line.split()
            type_dict[elements[0]] = elements[1]

#======= PREPERATIONS =======
#============================
print('Command line was: %s\n' % ' '.join(sys.argv))

print("Loading trajectory...\n")
u = mda.Universe(args.topology,args.trajectory)
sel = u.select_atoms(args.sel)
sel = sel.atoms.select_atoms("not name 'DUM'")
n_atoms = sel.atoms.n_atoms

for i, atom_type in enumerate(sel.types.astype(str)):
    atom = sel.atoms[i]
    atom.name = type_dict[atom_type]

startq = args.startq
dt = u.trajectory.dt

begin=int(args.begin // dt)
if args.end != None:
    end = int(args.end // dt)
else:
    end=int(u.trajectory.totaltime // u.trajectory.dt)

if begin > end:
    print("Start time is larger than end time!")

#create tmp directory for saving datafiles
try:
    os.mkdir("tmp")
except OSError as e:
    if e.errno == 17: #pass if directory exist
        pass

frames = 0
for ts in u.trajectory[begin:end+1:args.skipframes]:

    box = np.diag(mda.lib.mdamath.triclinic_vectors(ts.dimensions))
    sel.atoms.positions = sel.atoms.positions \
                            - box*np.round(sel.atoms.positions/box) # minimum image

    sel.atoms.write("tmp/{}.xyz".format(frames))

    ref_q = 4*np.pi/np.min(box)
    if ref_q > args.startq: startq = ref_q

    command = "-x -f {0} -t {1} -s {2} -o tmp/{3}.dat tmp/{3}.xyz".format(
                                              startq, args.endq,args.dq,frames)

    subprocess.run("{} {}".format(args.debyer,command),shell=True)

    frames += 1
    if (frames < 100):
        print ("\rEvaluating frame: {:>12} time: {:>12} ps".format(frames, round(ts.time)), end="")
    elif (frames < 1000 and frames % 10 == 1):
        print ("\rEvaluating frame: {:>12} time: {:>12} ps".format(frames, round(ts.time)), end="")
    elif (frames % 250 == 1):
        print ("\rEvaluating frame: {:>12} time: {:>12} ps".format(frames, round(ts.time)), end="")
    # call for output
    if ( frames % args.outfreq == 0 ):
        cleanup()
    sys.stdout.flush()

cleanup()
try:
    os.mkdir("tmp")
except OSError as e:
    if e.errno == 66: #pass if directory is not empty
        pass
print("\n")
