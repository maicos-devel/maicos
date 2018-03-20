#!/usr/bin/env python
# coding: utf-8

from __future__ import print_function

import sys

import MDAnalysis


def initilize_universe(argobj):
    """Reads the trajectory data and returns an MDAnalysis universe."""

    print('\nCommand line was: {}\n'.format(' '.join(sys.argv)))
    print("Loading trajectory...")
    u = MDAnalysis.Universe(argobj.topology, argobj.trajectory)

    argobj.dt = u.trajectory.dt

    argobj.beginframe = int(argobj.begin // argobj.dt)
    if argobj.end != None:
        argobj.endframe = int(argobj.end // argobj.dt)
    else:
        argobj.endframe = int(u.trajectory.totaltime // argobj.dt)

    if argobj.beginframe > argobj.endframe:
        sys.exit("Start time is larger than end time!")

    if argobj.box != None:
        assert (len(argobj.box) == 6 or len(argobj.box) == 3),\
            'The boxdimensions must contain 3 entries for the box vectors and possibly 3 more for the angles.'
        if len(argobj.box) == 6:
            u.dimensions = np.array(argobj.box)
        else:
            u.dimensions[:2] = np.array(argobj.box)

    return u


def print_frameinfo(ts, frame):
    """Prints the current frame information during evaulation."""
    if (frame < 100):
        print("\rEvaluating frame: {:>12} time: {:>12} ps".format(
            ts.frame, round(ts.time)), end="")
        sys.stdout.flush()
    elif (frame < 1000 and frame % 10 == 1):
        print("\rEvaluating frame: {:>12} time: {:>12} ps".format(
            ts.frame, round(ts.time)), end="")
        sys.stdout.flush()
    elif (frame % 250 == 1):
        print("\rEvaluating frame: {:>12} time: {:>12} ps".format(
            ts.frame, round(ts.time)), end="")
        sys.stdout.flush()
