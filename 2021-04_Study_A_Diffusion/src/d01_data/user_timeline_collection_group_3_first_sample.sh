#!/bin/bash

# 2021-09-20: used for the full data collection.

group_num=3
group_index=$((group_num-1))

python3.9 user_twarc_timelines.py \
    ../../data/02_intermediate/0${group_num}_group/group_${group_num}_sampled_weight_False_users.txt \
    --output_dir ../../data/01_raw/0${group_num}_group/ \
    --log_dir ../../logs/ \
    --log_level INFO \
    --start_time 2020-02-03T00:00:00 \
    --end_time 2020-03-16T00:00:00 \
    --limit 25000 \
    --omit_hashtags \
    --omit_retweets

ls -alht ../../data/01_raw/0${group_num}_group/ | mail -s "group ${group_num} first collection complete" hubert.au@oii.ox.ac.uk
