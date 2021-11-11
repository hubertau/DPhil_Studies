#!/bin/bash

./empirical_distribution_sampling.py \
  ../../data/01_raw/ \
  ../../data/02_intermediate/02_group/ \
  ../../references/search_hashtags.txt \
  2 \
  ../../data/02_intermediate/user_list_full_1_nocount.txt \
  --verbose \
  --max_total 5000

ls -alht ../../data/02_intermediate/02_group/ | mail -s "empirical sampling complete" hubert.au@oii.ox.ac.uk