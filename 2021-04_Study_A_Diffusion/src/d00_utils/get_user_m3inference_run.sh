#!/bin/bash

python3.9 get_user_m3inference.py \
    --data_dirs ../../data/01_raw/ \
    --output_dir ../../data/03_processed/ \
    --group_num 1 \
    --log_dir ../../logs/