import jsonlines
import gzip
import click
import os
import numpy as np
import pandas as pd

@click.command()
@click.option('--file', '-f', required=True, help='Compressed JSONL file with enriched data to evaluate.')
def main(file):
    text_lengths = []
    with gzip.open(file, 'rb') as f:
        reader = jsonlines.Reader(f)

        for story in reader:
            if 'query' in story:
                continue
            text_lengths.append(len(story.get('text')))

    df = pd.DataFrame()
    df['text_lengths'] = text_lengths
    print(df.describe())


if __name__ == '__main__':
    main()