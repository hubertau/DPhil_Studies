'''Functions to handle post processing
'''

import click
from loguru import logger
import pandas as pd
from collections import Counter
from datasets import Dataset
import pickle

@click.group(help='Commands for postprocessing')
@click.pass_context
def postprocess(ctx):
    pass

@postprocess.command()
@click.argument('file')
@click.option('--outfile', '-o', required=True)
@click.option('--original_data', '-d', required=True)
@click.option('--mcsourceinfo', '-m', required=True)
@click.option('--nerinfo', '-n', required=False)
def consolidatesubs(file, outfile, original_data, mcsourceinfo, nerinfo):
    '''Conslidate substance annotations from split back into stories'''

    with open(file, 'rb') as f:
        original_dict = pickle.load(f)

    new_dict = {}
    for k, v in original_dict.items():
        id, _ = k.split('_')
        if id not in new_dict:
            new_dict[id] = {v: 1}
        else:
            if v in new_dict[id]:
                new_dict[id][v] += 1
            else:
                new_dict[id][v] = 1

    # Convert the new dictionary to a DataFrame
    df = pd.DataFrame.from_dict({k: Counter(v) for k, v in new_dict.items()}, orient='index')
    df.fillna(0, inplace=True)
    df = df.astype(int)

    df.index.names=['processed_stories_id']

    # supply media id information
    original_raw = Dataset.load_from_disk(original_data)
    original_df = original_raw.to_pandas()
    original_df = original_df.set_index('processed_stories_id')
    original_df.index = original_df.index.map(str)

    df = df.join(original_df[['media_id', 'language', 'publish_date']])

    # supply media locale information:
    with open(mcsourceinfo, 'rb') as f:
        mc = pickle.load(f)
    df['country'] = df['media_id'].apply(mc)

    if nerinfo:
        with open(mcsourceinfo, 'rb') as f:
            ner = pickle.load(f)

    df.to_csv(outfile)

@postprocess.command()
@click.argument('file')
def consolidatener(file):
    with open(file, 'rb') as f:
        ner_annot = pickle.dump(f)

