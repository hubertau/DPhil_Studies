#!/bin/bash

./bispec_clustering_eval.py \
  ../../data/05_model_output/01_group \
  ../../data/06_reporting/01_group \
  ../../data/02_intermediate/01_group/user_count_mat_ngram_23.obj \
  ../../data/02_intermediate/01_group/mapping_ngram_23.obj \
| tee /dev/stderr | mail -s "Clustering Eval Complete" hubert.au@oii.ox.ac.uk