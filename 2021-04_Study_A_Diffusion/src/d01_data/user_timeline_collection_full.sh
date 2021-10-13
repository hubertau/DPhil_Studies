#!/bin/bash

# 2021-09-20: used for the full data collection.

python3.9 /home/hubert/DPhil_Studies/2021-04_Study_A_Diffusion/user_twarc_timelines.py \
    /home/hubert/DPhil_Studies/2021-04_Study_A_Diffusion/collection_results_2021_06_19_16_21/user_list_full_0_.txt \
    --output_dir /home/hubert/DPhil_Studies/2021-04_Study_A_Diffusion/collection_results_2021_06_19_16_21 \
    --start_time 2017-11-05T00:00:00 \
    --end_time 2018-05-29T00:00:00 \
    --min_tweets 10 \
    --limit 25000 \
    --omit_hashtags \
    --omit_retweets \
    --continue_from_user_id 560707426
