#!/bin/bash

# 2021-09-20: used for the full data collection.

python3.9 user_twarc_timelines.py \
    ../../data/02_intermediate/02_group/group_2_snowball_1.txt \
    --output_dir ../../data/01_raw/02_group/ \
    --log_dir ../../logs/ \
    --log_level DEBUG \
    --start_time 2018-11-21T00:00:00 \
    --end_time 2019-01-02T00:00:00 \
    --limit 25000 \
    --omit_hashtags \
    --omit_retweets

ls -alht ../../data/01_raw/02_group/ | mail -s "group 2 first snowball collection complete" hubert.au@oii.ox.ac.uk
