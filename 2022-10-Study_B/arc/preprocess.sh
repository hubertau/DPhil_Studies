#!/bin/bash

#SBATCH --nodes=1
#SBATCH --partition=medium
#SBATCH --time=48:00:00
#SBATCH --job-name=B-preprocess
#SBATCH --clusters=htc
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=350G
#SBATCH --output=/home/ball4321/DPhil_Studies/2022-10-Study_B/logs/preprocess-%A.log
#SBATCH --mail-type=ALL
#SBATCH --mail-user=hubert.au@oii.ox.ac.uk


module purge
module load Anaconda3/2020.11
module load foss/2020a

source activate $DATA/venv_b

cd /home/ball4321/DPhil_Studies/2022-10-Study_B/src

python -m newsanalysis --gpu duplicate-check \
  $DATA/data_b/01_raw/data_ndid.jsonl \
  $DATA/data_b/03_processed/preprocessing/