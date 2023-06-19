'''Functions to handle post processing
'''

import click
from loguru import logger
import pandas as pd
from collections import Counter
from itertools import repeat
from datasets import Dataset
from pathlib import Path
import pickle

import pickle
from pathlib import Path
import rapidfuzz
from concurrent.futures import ProcessPoolExecutor

@click.group(help='Commands for postprocessing')
@click.pass_context
def postprocess(ctx):
    pass

@postprocess.command()
@click.argument('subsfile')
@click.option('--outfolder', '-o', required=True)
@click.option('--original_data', '-d', required=False)
@click.option('--relfile', '-r', required=False)
@click.option('--mcsourceinfo', '-m', required=False)
@click.option('--substhresh', '-st', type=float, default=None)
@click.option('--relthresh', '-rt', type=float, default=None)
def consolidatesubs(subsfile, outfolder, original_data, mcsourceinfo, relfile, substhresh, relthresh):
    '''Conslidate substance annotations from split back into stories'''

    # Load in substance annotation file
    with open(subsfile, 'rb') as f:
        original_dict = pickle.load(f)

    # Load in relevance annotation file
    with open(relfile, 'rb') as f:
        rel_dict = pickle.load(f)

    for k,v in original_dict.items():
        has_logits = isinstance(v, tuple)
        break
    logger.info(f'Logits is {has_logits}')

    # get the substance annotations back into stories and not parts
    new_dict = {}
    if has_logits:
        for k, predict_tuple in original_dict.items():
            id, _ = k.split('_')

            # check relevance threshold (N.B. don't need to check if predict is 1 because otherwise it wouldn't be in subs dict):
            if rel_dict[k][0] < relthresh:
                continue

            # check subs threshold
            logit, v  = predict_tuple
            if logit < substhresh:
                continue

            if id not in new_dict:
                new_dict[id] = {v: 1}
            else:
                if v in new_dict[id]:
                    new_dict[id][v] += 1
                else:
                    new_dict[id][v] = 1

    else:
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

    if original_data:
        # supply media id information
        original_raw = Dataset.load_from_disk(original_data)
        original_df = original_raw.to_pandas()
        original_df = original_df.set_index('processed_stories_id')
        original_df.index = original_df.index.map(str)

        df = df.join(original_df[['media_id', 'language', 'publish_date']])

    # supply media locale information:
    if mcsourceinfo:
        with open(mcsourceinfo, 'rb') as f:
            mc = pickle.load(f)
        df['country'] = df['media_id'].replace(mc)

    df.to_csv(f'{outfolder}/{Path(subsfile).stem}{"_" if any([original_data, mcsourceinfo]) else ""}{"d" if original_data else ""}{"m" if mcsourceinfo else ""}{f"_r{relthresh}" if relthresh else ""}{f"_s{substhresh}" if substhresh else ""}.csv')

def process_one_token(number, token, names_ref):
    if number % 10000 == 0:
        logger.info(f'{number}')
    return (token, rapidfuzz.process.extractOne(token.upper(), names_ref))

@postprocess.command()
@click.argument('ner_file')
@click.argument('outfile')
@click.option('--dataset', '-d', required=True)
@click.option('--names', '-n', required=True)
@click.option('--surnames', '-s', required=True)
@click.option('--up_to', '-u', type=int)
def consolidatener(ner_file, outfile, dataset, names, surnames, up_to):
    with open(ner_file, 'rb') as f:
        ner_annot = pickle.load(f)

    logger.info('NER loaded in')

    # get languages of articles
    og_dataset = Dataset.load_from_disk(dataset)
    og_dataset = og_dataset.to_pandas()
    og_dataset = og_dataset.set_index('processed_stories_id')
    logger.info("Dataset loaded in")

    ner_filtered = {}
    for k, v in ner_annot.items():
        lang = og_dataset.loc[k]['language']
        if lang in ['zh', 'ja', 'ko', 'bn', 'hi', 'ta', 'gu', 'kn', 'te', 'ml', 'th', 'my', 'mr', 'vi', 'ne', 'lo', 'ur', 'si', 'or', 'ceb']:
            ner_filtered[k] = v
        else:

            ner_filtered[k] = [i for i in v if 1 < len(i.split())<5 ]
    logger.info('Dataset filtered')

    # import name list to compare against
    name_database = pd.read_csv(names,header=None, names = ['name', 'frequency', 'males', 'females'])
    surname_database = pd.read_csv(surnames, header=None, names = ['name', 'frequency'])

    #combine
    allname_database = pd.concat((name_database['name'], surname_database['name'])).to_list()
    logger.info('Names loaded in')

    unique_tokens = set()
    for k, v in ner_filtered.items():
        lang = og_dataset.loc[k]['language']
        if lang in ['zh', 'ja', 'ko', 'bn', 'hi', 'ta', 'gu', 'kn', 'te', 'ml', 'th', 'my', 'mr', 'vi', 'ne', 'lo', 'ur', 'si', 'or', 'ceb']:
            pass
        else:
            for i in v:
                unique_tokens.update(i.split())
    unique_tokens = list(unique_tokens)
    if up_to:
        logger.info(f'DEBUGGING UP TO {up_to}')
        unique_tokens = unique_tokens[:up_to]
    logger.info('Unique tokens extracted')
    logger.info(f'Beginning ProcessPoolExecutor')
    processpoolout = []
    with ProcessPoolExecutor(max_workers=None) as executor:
        # for number, token in zip(range(len(unique_tokens)), unique_tokens):
        #     if number % 10000 == 0:
        #         logger.info(f'{number}')
        #     processpoolout.append(executor.submit(process_one_token, token, allname_database))
        processpoolout = executor.map(process_one_token, range(len(unique_tokens)), unique_tokens, repeat(allname_database))
    logger.info('ProcessPool done')
    # processpoolout = [i.result() for i in processpoolout]

    final_dict = {}
    for tok, corrected in processpoolout:
        final_dict[tok] = corrected[0]

    with open(outfile, 'wb') as f:
        pickle.dump(final_dict, f)
    logger.info(f'Saved to {outfile}')