#!/bin/bash

group_num=3

group_index=$((group_num-1))

./empirical_distribution_sampling.py \
  ../../data/01_raw/ \
  ../../data/02_intermediate/0${group_num}_group/ \
  ../../references/search_hashtags.txt \
  ${group_num} \
  ../../data/02_intermediate/user_list_full_${group_index}_nocount.txt \
  --verbose \
  --group_daterange_file /home/hubert/DPhil_Studies/2021-04_Study_A_Diffusion/data/02_intermediate/FAS_peak_analysis.hdf5 \
  --max_total 2500

ls -alht ../../data/02_intermediate/0${group_num}_group/ | mail -s "empirical sampling complete for group ${group_num}" hubert.au@oii.ox.ac.uk