#!/bin/bash

./empirical_distribution_sampling.py \
  ../../data/01_raw/ \
  ../../data/02_intermediate/02_group/ \
  ../../references/search_hashtags.txt \
  2 \
  ../../data/02_intermediate/user_list_full_1_nocount.txt \
  --verbose \
  --group_daterange_file /home/hubert/DPhil_Studies/2021-04_Study_A_Diffusion/data/02_intermediate/FAS_peak_analysis.hdf5 \
  --max_total 2500

ls -alht ../../data/02_intermediate/02_group/ | mail -s "empirical sampling complete" hubert.au@oii.ox.ac.uk