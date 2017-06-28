#!/usr/bin/env python

from __future__ import print_function
from __future__ import division
import numpy as np
import MDAnalysis
import copy
import sys, getopt, os
import moleculeInsertion

def main(argv=None):
  if argv is None:
    argv = sys.argv
    # etc., replacing sys.argv with argv in the getopt() call.

  ingrofile=argv[-1]
  print ("Loading", ingrofile)

  global zmin, zmax, u
  u=MDAnalysis.Universe(ingrofile)
  zmin=0.
  zmax=u.dimensions[2]
  Nw=1
  distance=1.25

  #parse command line options
  try:
    opts, args = getopt.getopt(sys.argv[1:-1], "h", ["zmin=", "zmax=", "Nw=", "help", "dist="])
  except getopt.error, msg:
    print (msg)
    print ("for help use --help")
    return 2
  for o, a in opts:
    if (o in ("--zmin")):
      zmin = float(a)*10.
    if (o in ("--zmax")):
      zmax = float(a)*10.
    if (o in ("--Nw")):
      Nw = int(a)
    if (o in ("--dist")):
      distance = float(a)


  print ("Inserting", Nw, "water molecules bewteen z=", zmin, "and z=", zmax)
  water = MDAnalysis.Universe('/home/aschlaich/repos/dielectric/resources/watermolecule.gro')
  for i in range(Nw):
    u = moleculeInsertion.box(u,water,zmax=zmax,zmin=zmin,distance=distance)
    print("%i of %i watermolecules placed." % (i,Nw), end="\r")
    sys.stdout.flush()
    if (i+1)%250 == 0:
      u.atoms.write('solvate.gro')

  u.atoms.write('solvate.gro')


if __name__ == "__main__":
  sys.exit(main())
