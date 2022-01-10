#!/bin/bash

group_num=1
group_index=$((group_num-1))

python3.9 followers.py \
    ../../data/02_intermediate/user_list_full_${group_index}_nocount.txt \
    --data_dir ../../data/01_raw/0${group_num}_group/ \
    --twarc_credentials twarc_credentials.json \
    --log_dir ../../logs/ \
    --log_level INFO

collected_follower_files=$(ls ../../data/01_raw/0${group_num}_group/follower* | wc -l)
collected_following_files=$(ls ../../data/01_raw/0${group_num}_group/following* | wc -l)

echo "done" | mail -s "Followers: ${collected_follower_files}, Following: ${collected_following_files}" hubert.au@oii.ox.ac.uk