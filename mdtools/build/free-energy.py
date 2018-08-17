#!/usr/bin/env python
# coding: utf-8

# ===================================================================================================
# IMPORTS
# ===================================================================================================

from __future__ import absolute_import, division, print_function

import os
import subprocess
import sys

import gromacs

from ..import initilize_parser
from ..utils import copy_itp, nlambdas, replace, submit_job

# ===================================================================================================
# INPUT OPTIONS
# ===================================================================================================
parser = initilize_parser()
parser.description = """
Set up folders and files for GROMACS free energy simulations.
The script will read the lambda states from the given mdp files and creates a subfolder for every lambda step.
For every given mdp file it will grommp and run a simulation in the given order.
The first simulation will use the given gro file. All following files are taken from the prior simulation.

If the -sub flag is choosen all commands will be appended to the given file and 
submitted to the SLURM workload manager."""
parser.add_argument('-f',     dest='mdp',   type=str,  default=[
                    'grompp.mdp'], nargs='+', help="One or several grompp input files with MD parameters")
parser.add_argument('-c',     dest='gro',   type=str,  default='conf.gro',
                    help="Structure file: gro g96 pdb brk ent esp tpr")
parser.add_argument('-n',     dest='index', type=str,
                    default=None,           help="Index file")
parser.add_argument('-p',     dest='top',   type=str,
                    default='topol.top',    help="Topology file")
parser.add_argument('-sub',   dest='sub',   type=str,
                    default=None,           help="Use a SLURM submission script.")
parser.add_argument('-sp',    dest='split', action="store_true",
                    help="Split each lambda state into a single submission.")
parser.add_argument('-mdrun', dest='mdrun', type=str,  default="gmx mdrun",
                    help="Command line to run a simulation, e.g. 'gmx mdrun' or 'mdrun_mpi'")
parser.add_argument('-d',     dest='fdepth', type=int,  default=1,
                    help="Number of folders from the file tree to include in the name for the submission.")

# ===================================================================================================
# MAIN
# ===================================================================================================


def main(firstarg=2, DEBUG=False):

    args = parser.parse_args(args=sys.argv[firstarg:])

    # Check if all files exist
    files = [args.mdp, [args.top], [args.gro]]
    for f in [item for sublist in files for item in sublist]:
        if not os.path.isfile(f):
            sys.exit("{} does not exists or is not accessible.".format(f))

    # Convert all Pathes to absolute Pathes
    args.mdp = [os.path.abspath(mdp) for mdp in args.mdp]
    if args.index != None:
        args.index = os.path.abspath(args.index)
    args.top = os.path.abspath(args.top)
    if args.sub != None:
        args.sub = os.path.abspath(args.sub)

    # get number of lambda states
    nlambdas = nlambdas(gromacs.fileformats.mdp.MDP(args.mdp[0]))
    if nlambdas == 0:
        sys.exit("No lambda states found. Can't continue...")

    print("Create input folders and files for {} lambda states...".format(nlambdas))

    projectname = "-".join(os.getcwd().split('/')[-args.fdepth:])

    command = ""
    for l in range(nlambdas):

        gro = os.path.abspath(args.gro)
        workdir = 'lambda_{}'.format(l)

        try:
            os.mkdir(workdir)
        except OSError:
            pass
        os.chdir(workdir)

        for mdp_path in args.mdp:
            mdp = gromacs.fileformats.mdp.MDP(mdp_path)
            mdp['init-lambda-state'] = l  # set lambda state

            mdp_split = os.path.basename(mdp_path).split(".")
            mdp_name = "{}_{}.{}".format(mdp_split[0], l, mdp_split[1])
            mdp.write(mdp_name)

            if args.sub != None and not args.split:
                command += "\n\ncd {}\n".format(workdir)

            # grompp
            command += "gmx grompp -maxwarn 2 -f {} -c {} -p {} -o {}_{}".format(
                os.path.relpath(mdp_name), os.path.relpath(gro),
                os.path.relpath(args.top), os.path.relpath(mdp_split[0]), l)
            if args.index != None:
                command += " -n {}".format(os.path.relpath(args.index))
            command += "\n"

            # mdrun
            # use standard mdrun and only 1 thread for energy minimization.
            if mdp["integrator"] == "steep":
                command += "gmx mdrun -nt 1"
            else:
                command += "{}".format(os.path.relpath(args.mdrun))
            command += " -deffnm {}_{}\n".format(mdp_split[0], l)

            # set new gro file
            gro = "{}_{}.gro".format(mdp_split[0], l)

        if args.sub != None and args.split:
            print("\r{}/{}".format(l + 1, nlambdas), end="")
            submit_job(args.sub, "{}_{}".format(projectname, l), command)
            command = ""

        elif args.sub != None and not args.split:
            command += "cd ..\n"

        elif args.sub == None:
            subprocess.call(command, shell=True)

        os.chdir('..')

    if args.sub != None and not args.split:
        submit_job(args.sub, "{}".format(projectname), command)

    print("\n")
    if DEBUG:
        # Inject local variables into global namespace for debugging.
        for key, value in locals().items():
            globals()[key] = value


if __name__ == "__main__":
    main(firstarg=1)
