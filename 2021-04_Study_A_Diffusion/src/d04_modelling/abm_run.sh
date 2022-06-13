#!/usr/bin/bash

kernprof -l abm.py \
    --search_hashtags /home/hubert/DPhil_Studies/2021-04_Study_A_Diffusion/references/search_hashtags.txt \
    --group_num 2 \
    --data_path /home/hubert/DPhil_Studies/2021-04_Study_A_Diffusion/data/ \
    --activity_file /home/hubert/DPhil_Studies/2021-04_Study_A_Diffusion/data/03_processed/activity_counts.hdf5 \
    --output_path /home/hubert/DPhil_Studies/2021-04_Study_A_Diffusion/data/06_reporting \
    --log_dir /home/hubert/DPhil_Studies/2021-04_Study_A_Diffusion/logs \
    --params_file params.json \
    --max_workers 2 \
    --line_profiler \
    --debug_len 10

