#!/bin/bash

#SBATCH --nodes=1
#SBATCH --partition=short
#SBATCH --mem=300G
#SBATCH --clusters=arc
#SBATCH --time=12:00:00
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=hubert.au@oii.ox.ac.uk
#SBATCH --output=logs/whisper-%A-%a.log
#SBATCH --array=0-18%19   # Create 70 jobs (0-69), where %N limits the number of simultaneous jobs

module load Anaconda3/2023.09-0

conda activate $DATA/venv_legalpythia

# Function to send email
send_email() {
    log_file="logs/whisper-${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}.log"  # Ensure this matches the output file name
    mutt -s "${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID} Log" -a "${log_file}" -- hubert.au@oii.ox.ac.uk < /dev/null
}

trap send_email EXIT

# Calculate start and end numbers
START=$(( $SLURM_ARRAY_TASK_ID * 5 ))
END=$(( START + 4 ))

# Print or use these numbers as needed
echo "Job ID: $SLURM_ARRAY_TASK_ID, Start: $START, End: $END"

# Here, call your actual processing script or command, passing $START and $END
python whisper_arc.py \
  -f /data/inet-large-scale-twitter-diffusion/ball4321/data_c/YT/files_remaining.txt \
  -s $START \
  -e $END \
  -o /data/inet-large-scale-twitter-diffusion/ball4321/data_c/YT/audio_transcribed
