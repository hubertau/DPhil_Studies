#!/bin/bash

# 2021-09-20: used for the full data collection.

python3.9 user_twarc_timelines.py \
    ../../data/02_intermediate/user_list_full_0_.txt \
    --output_dir ../../logs/ \
    --data_dir ../../data/01_raw/ \
    --start_time 2017-11-05T00:00:00 \
    --end_time 2018-05-29T00:00:00 \
    --min_tweets 10 \
    --limit 25000 \
    --omit_hashtags \
    --omit_retweets
