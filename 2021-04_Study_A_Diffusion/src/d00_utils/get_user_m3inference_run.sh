#!/bin/bash

group_num=3

python3.9 get_user_m3inference.py \
    --data_dirs ../../data/01_raw/ \
    --output_dir ../../data/03_processed/ \
    --group_num ${group_num} \
    --log_dir ../../logs/

ls -alht ../../data/03_processed/ | mail -s "m3 ${group_num} complete" hubert.au@oii.ox.ac.uk