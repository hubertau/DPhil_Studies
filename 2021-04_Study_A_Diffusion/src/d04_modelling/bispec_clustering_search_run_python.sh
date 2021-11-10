#!/bin/bash

./bispec_clustering_search.py 23 \
  ../../data/02_intermediate/01_group/ \
  --output_dir ../../data/05_model_output/01_group/ \
  --implementation python \
  --range 505-750 \
  --interval 5 \
  --min_user 100 \

ls -alht ../../data/05_model_output/01_group/ | head -n 50 | mail -s "test" hubert.au@oii.ox.ac.uk