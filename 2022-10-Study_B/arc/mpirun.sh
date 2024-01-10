#!/bin/bash

#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --partition=devel
#SBATCH --job-name=PyHello
#SBATCH --time=00:10:00

module purge
module load Anaconda3/2020.11
module load foss/2020a

source activate $DATA/venv_b

mpirun python ./mpihello.py
