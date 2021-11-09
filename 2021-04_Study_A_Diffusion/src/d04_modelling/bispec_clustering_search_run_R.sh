#!/bin/bash

./bispec_clustering_search.py ../../data/02_intermediate/bispec_ready_counts_23.csv \
  --output_dir ../../data/05_model_output/ \
  --implementation r \
  --range 100-105 \
  --interval 1 \
  --min_user 50 | mail -s "clustering complete" hubert.au@oii.ox.ac.uk