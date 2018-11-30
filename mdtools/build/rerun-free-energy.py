#!/usr/bin/env python
# coding: utf-8

# ===================================================================================================
# IMPORTS
# ===================================================================================================

from __future__ import absolute_import, division, print_function

import os
import shutil
import subprocess
import sys

import gromacs

from .. import initilize_parser
from ..utils import copy_itp, nlambdas, replace, submit_job

# ===================================================================================================
# INPUT OPTIONS
# ===================================================================================================
parser = initilize_parser()
parser.description = """ 
Set up folders and files for GROMACS free energy simulations to
recaculate dH/dl for a new topology.

With the -q option the rerun can be done for different final charges. 
To use this option replace the final charge in the topoly file by <XXX> and give 
the original charge."""

parser.add_argument(
    '-f',
    dest='mdp',
    type=str,
    default='grompp.mdp',
    help="grompp input file with MD parameters")
parser.add_argument(
    '-b',
    dest='begin',
    type=int,
    default=0,
    help="First lambda state from which the new FINAL charge is taken.")
parser.add_argument(
    '-e',
    dest='end',
    type=int,
    default=None,
    help=
    "Last lambda state from which the new FINAL charge is taken. If None last state from original mdp file is taken"
)
parser.add_argument(
    '-c',
    dest='gro',
    type=str,
    default='conf.gro',
    help="Structure file: gro g96 pdb brk ent esp tpr")
parser.add_argument(
    '-n', dest='index', type=str, default=None, help="Index file")
parser.add_argument(
    '-p', dest='top', type=str, default='topol.top', help="Topology file.")
parser.add_argument(
    '-q',
    dest='charge',
    type=float,
    default=None,
    help=
    "Final charge of original topology. <XXX> will be replaced with an adjusted charge depending on the lambda state."
)
parser.add_argument(
    '-nl',
    dest='nolink',
    action='store_false',
    help="If set files will NOT linked into an ordered folder in a dhdl dir")
parser.add_argument(
    '-d',
    dest='dir',
    type=str,
    default="lambda_",
    help="Directory prefix, where trajectories are stored.")
parser.add_argument(
    '-x',
    dest='prefix',
    type=str,
    default="md",
    help="file prefix for trajectory etc.")
parser.add_argument(
    '-sub',
    dest='sub',
    type=str,
    default=None,
    help="Use a SLURM submission script. If None calculation is done locally.")
parser.add_argument(
    '-o',
    dest='output',
    type=str,
    default="",
    help="prefix for all output files and folders")
parser.add_argument(
    '-mdrun',
    dest='mdrun',
    type=str,
    default="gmx mdrun",
    help="Command line to run a simulation, e.g. 'gmx mdrun' or 'mdrun_mpi'")

# ===================================================================================================
# MAIN
# ===================================================================================================


def main(firstarg=2, DEBUG=False):

    args = parser.parse_args(args=sys.argv[firstarg:])

    # Convert all paths to absolute paths
    args.mdp = os.path.abspath(args.mdp)
    args.gro = os.path.abspath(args.gro)
    if args.index != None:
        args.index = os.path.abspath(args.index)
    if args.sub != None:
        args.sub = os.path.abspath(args.sub)
    args.top = os.path.abspath(args.top)

    if len(args.output) != 0:
        args.output = "{}_".format(args.output)

    if len(args.prefix) != 0:
        args.prefix = "{}_".format(args.prefix)

    projectname = os.getcwd().split('/')[-1]
    mdp = gromacs.fileformats.mdp.MDP(args.mdp)

    if args.end == None:
        end = utilities.nlambdas(mdp)
    else:
        end = args.end + 1

    if args.charge != None:
        args.charge == float(args.charge)
        print("Write new topologies in top directory.")
        try:
            os.mkdir("{}top".format(args.output))
        except OSError:
            pass

        os.chdir("{}top".format(args.output))
        utilities.copy_itp(args.top)

        for lfinal in range(args.begin, end):
            new_name = "topol_{}.top".format(lfinal)
            new_charge = mdp['coul-lambdas'][lfinal] * args.charge
            utilities.replace(args.top, "XXX", new_charge, new_name)

        os.chdir("..")

    for l in range(end):
        os.chdir(args.dir + str(l))
        if args.sub != None:
            command = ""
        # reload mdp file... ugly....
        mdp = gromacs.fileformats.mdp.MDP(args.mdp)
        mdp['init-lambda-state'] = l
        if args.begin > l:
            begin = args.begin
        else:
            begin = l

        if args.charge != None:
            for lfinal in range(begin, end)[::-1]:
                if lfinal == 0:
                    # senseles because no TI if this is the final step...
                    continue
                # First reduce all lambda states
                for t in "fep", "vdw", "bonded", "restraint", "mass", "temperature":
                    try:
                        mdp["{}-lambdas".format(t)] = mdp["{}-lambdas".format(
                            t)][:lfinal + 1]
                    except KeyError:
                        pass
                # Then reduce and rescale coul lambdas.
                qfinal = mdp["coul-lambdas"][lfinal]
                mdp["coul-lambdas"] = mdp["coul-lambdas"][:lfinal + 1] / qfinal

                title = "{}{}{}_{}".format(args.output, args.prefix, l, lfinal)
                mdp.write("{}".format(title))
                command += "gmx grompp -maxwarn 2 -f {0} -c {1} -p {2} -o {0}".format(
                    title, os.path.relpath(args.gro),
                    "../{}top/topol_{}".format(args.output, lfinal))
                if args.index != None:
                    command += " -n {}".format(os.path.relpath(args.index))
                command += "\n"

                command += "{} -deffnm {} -rerun {}{}\n".format(
                    args.mdrun, title, args.prefix, l)

        else:
            title = "{}{}{}".format(args.output, args.prefix, l)
            mdp.write("{}".format(title))
            command += "gmx grompp -maxwarn 2 -f {0} -c {1} -p {2} -o {0}".format(
                title, os.path.relpath(args.gro), os.path.relpath(args.top))
            if args.index != None:
                command += " -n {}".format(os.path.relpath(args.index))
            command += "\n"

            command += "{} -deffnm {} -rerun {}{}\n".format(
                os.path.relpath(args.mdrun), title, args.prefix, l)

        if args.sub != None:
            utilities.submit_job(
                args.sub, "{}{}_{}_rerun".format(args.output, projectname, l),
                command)
        else:
            subprocess.call(command, shell=True)

        os.chdir("..")

    if args.nolink:
        print("linking xvg files...")
        try:
            os.mkdir("{}dhdl".format(args.output))
        except OSError:
            pass
        os.chdir("{}dhdl".format(args.output))

        for lfinal in range(args.begin, end):
            if args.charge != None:
                try:
                    os.mkdir("lambda_{}".format(lfinal))
                except OSError:
                    pass
                os.chdir("lambda_{}".format(lfinal))

                for l in range(lfinal + 1):
                    os.symlink(
                        "../../lambda_{0}/{1}{2}{0}_{3}.xvg".format(
                            l, args.output, args.prefix, lfinal),
                        "{}{}{}.xvg".format(args.output, args.prefix, l))

                os.chdir("..")
            else:
                os.symlink(
                    "../lambda_{0}/{1}{2}{0}.xvg".format(
                        lfinal, args.output, args.prefix), "{}{}{}.xvg".format(
                            args.output, args.prefix, lfinal))

    print("\n")
    if DEBUG:
        # Inject local variables into global namespace for debugging.
        for key, value in locals().items():
            globals()[key] = value


if __name__ == "__main__":
    main(firstarg=1)
