#!/bin/bash

#SBATCH --nodes=1
#SBATCH --partition=short
#SBATCH --time=1:00:00
#SBATCH --job-name=B-preprocess
#SBATCH --clusters=htc
#SBATCH --gres=gpu:1
#SBATCH --output=/home/ball4321/2022-10-Study_B/logs/%A.log
#SBATCH --mail-type=ALL
#SBATCH --mail-user=hubert.au@oii.ox.ac.uk


module purge
module load Anaconda3/2020.11
module load foss/2020a

source activate $DATA/venv_b

cd /home/ball4321/DPhil_Studies/2022-10-Study_B/src

python -m newsanalysis -- duplicate-check \
  $DATA/data_b/01_raw/data.jsonl \
  $DATA/data/03_processed/preprocessing/