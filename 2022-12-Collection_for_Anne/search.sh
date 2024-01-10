#!/bin/bash

python3.9 ../2021-04_Study_A_Diffusion/src/d01_data/full_archive_search.py \
    --search_query_txt search_query.txt \
    --start_time 2020-06-01 \
    --end_time 2022-12-22 \
    --output_dir /home/hubert/DPhil_Studies/2022-12-Collection_for_Anne \
    --data_dir /home/hubert/DPhil_Studies/2022-12-Collection_for_Anne \
    --raw


echo "Collection for Anne complete" | mail -s "[DPhil Server] FAS Collection Complete" hubert.au@oii.ox.ac.uk
