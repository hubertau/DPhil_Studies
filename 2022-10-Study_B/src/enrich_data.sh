#!/bin/bash

python3.9 enrich_media.py \
    --infile ../data/01_raw/pilot.jsonl \
    --outfile ../data/01_raw/pilot_enriched.jsonl \
    --log_dir ../logs/ \
    --log_handler_level both \

echo "Enrichment complete" | mail -s "[DPhil Server] Mediacloud Enrichment Complete" hubert.au@oii.ox.ac.uk