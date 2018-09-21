#!/bin/bash

#SBATCH --job-name=job_name
#SBATCH --mail-user=ploche@physik.fu-berlin.de
#SBATCH --mail-type=FAIL

#SBATCH --output=job_%j.out
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=1G
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --partition=main

set -e #Exit immediately if a command exits with a non-zero status

sleep $((RANDOM%10))
module load gromacs/single/2016.5

