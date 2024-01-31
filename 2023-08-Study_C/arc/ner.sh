#!/bin/bash

#SBATCH --nodes=1
#SBATCH --partition=medium
#SBATCH --mem=300G
#SBATCH --clusters=htc
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mail-type=START,FAIL
#SBATCH --mail-user=hubert.au@oii.ox.ac.uk
#SBATCH --output=ner-%A.log

module load Anaconda3/2023.09-0

conda activate $DATA/venv_legalpythia

# Function to send email
send_email() {
    log_file="ner-${SLURM_JOB_ID}.log"  # Ensure this matches the output file name
    mutt -s "${SLURM_JOB_ID} Log" -a "${log_file}" -- hubert.au@oii.ox.ac.uk < /dev/null
}

trap send_email EXIT

#echo ${DATA}
#echo `which python`

# Check for mandatory positional argument
if [ $# -lt 1 ]; then
    echo "Usage: $0 <arg1> [size]"
    exit 1
fi

# Set the positional arguments
arg1=$1  # Mandatory argument
size=${2:-4000000}  # Optional size argument with default

python simple_ner.py $arg1 /data/inet-large-scale-twitter-diffusion/ball4321/data_c/YT/raw_ner --size $size


