#!/bin/bash

./generate_user_to_hashtag_matrix.py \
  ../../data/01_raw/ \
  ../../data/02_intermediate/01_group/ \
  --ngram_range 34 \
  --low_memory \
  --verbose

ls -alht ../../data/02_intermediate/01_group/ | mail -s "vectorizer complete" hubert.au@oii.ox.ac.uk