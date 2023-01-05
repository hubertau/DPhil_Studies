#!/bin/bash

infile="collect_2014-10-17_to_2017-10-17"

python3 enrich_media.py \
    --infile ../data/01_raw/${infile}.jsonl \
    --outfile ../data/01_raw/${infile}_enriched.jsonl \
    --multithread \
    --workers 40 \
    --log_dir ../logs/ \
    --log_handler_level both \
    --log_level INFO \
    # --continue_from 1387283816

echo "Enrichment complete: ${infile}" | mail -s "[DPhil Server] Mediacloud Enrichment Complete" hubert.au@oii.ox.ac.uk