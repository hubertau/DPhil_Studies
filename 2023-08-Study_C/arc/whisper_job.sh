#!/bin/bash

#SBATCH --cpus-per-task=48
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=2
#SBATCH --partition=short
#SBATCH --mem-per-cpu=7G
#SBATCH --time=12:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=hubert.au@oii.ox.ac.uk

module load Anaconda3/2022.10
module load FFmpeg/6.0-GCCcore-12.3.0

conda activate $DATA/venv_legalpythia

# Calculate the total number of cores
total_cores=$((SLURM_JOB_NUM_NODES * SLURM_CPUS_ON_NODE-10))
echo "Total cores available: $total_cores"

cd $DATA/data_c/YT/

cat files_audio_test96.txt | xargs -P $total_cores -I {} whisper {} --model medium -f json --verbose False -o audio_transcribed/
