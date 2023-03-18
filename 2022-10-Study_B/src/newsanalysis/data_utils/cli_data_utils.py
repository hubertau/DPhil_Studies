import click
import glob
import jsonlines
from loguru import logger

from ..dataviz import show_date_range
from . import *

@click.group(help='Commands relating to the processing of data')
@click.pass_context
def preprocess(ctx):
    pass

@preprocess.command()
@click.pass_context
@click.argument('glob_command')
@click.argument('outfile')
def consolidate(ctx, glob_command, outfile):
    """Consolidate separate collected files in to one data file.

    Args:
        glob_command (str): glob command
        outfile (str): save file

    Raises:
        ValueError: if no files to consolidate, bad glob command
    """

    if ctx.obj['DEBUG']:
        logger.info(f'Glob command: {glob_command}')
        logger.info(f'Outfile: {outfile}')
    files_to_consolidate = glob.glob(glob_command)
    if ctx.obj['DEBUG']:
        logger.info(files_to_consolidate)
    if not files_to_consolidate:
        raise ValueError

    # check all queries are the same
    if ctx.obj['DEBUG']:
        logger.info('Performing query check')
    query = None
    for file in files_to_consolidate:
        with jsonlines.open(file, 'r') as reader:
            for story in reader.iter(skip_empty=True, skip_invalid=True):
                if 'query' in story:
                    if query is not None:
                        assert story.get('query') == query, 'Queries in files do not match.'
                    else:
                        query = story.get('query')
                    break

    if ctx.obj['DEBUG']:
        logger.info('Collecting max start and end date')

    min_date, max_date = None, None
    for file in files_to_consolidate:
        min_date, max_date = show_date_range(file, min_date, max_date)
    if ctx.obj['DEBUG']:
        logger.info(f'Total date range is {min_date} to {max_date}')

    query_written = False
    with jsonlines.open(outfile, 'w') as writer:
        for index, file in enumerate(files_to_consolidate):
            logger.info(f'Processing {file}, {index+1} out of {len(files_to_consolidate)}')
            with jsonlines.open(file, 'r') as reader:
                for story in reader.iter(skip_empty=True, skip_invalid=True):
                    if 'query' in story:
                        if not query_written:
                            story = {
                                'query':story.get('query'),
                                'start_date': min_date.strftime('%Y-%m-%d'),
                                'end_date':max_date.strftime('%Y-%m-%d'),
                                'date_of_collection': datetime.datetime.now().strftime('%Y-%m-%d')
                            }
                            writer.write(story)
                            query_written = True
                        else:
                            continue
                    writer.write(story)


@preprocess.command()
@click.pass_context
@click.argument('file')
@click.argument('savepath')
@click.option('--by', default='id', type=click.Choice(['id', 'url']))
def remove_redundant_items(ctx, file, savepath, by):
    '''Removal of redundant items, e.g. repeated unique ids or urls'''
    # from . import remove_redundant
    remove_redundant(file, savepath, by)

@preprocess.command()
@click.pass_context
@click.argument('file')
@click.argument('savepath')
@click.option('--threshold',default=0.9)
def duplicate_check(ctx, file, savepath, threshold = 0.9):
    """Deducplication. Given an ENRICHED file, generate a similarity matrix on a bag-of-words representation

    Args:
        file (_type_): _description_

    Returns:
        _type_: _description_
    """
    logger.info(f'threshold: {threshold}')
    logger.info(f'Savepath: {savepath}')
    logger.info(f'GPU flag is {ctx.obj["GPU"]}')
    # from . import deduplicate
    # from transformers import BertTokenizerFast
    # import re
    # import functools
    # from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
    # import faiss
    # import numpy as np
    # import h5py
    # import os
    # from time import perf_counter
    # from ..dataviz import retrieve_story_and_lang, retrieve_story_lens
    deduplicate(file, savepath, gpu=ctx.obj['GPU'])
    logger.info('done')

@preprocess.command()
@click.pass_context
@click.argument('file')
@click.argument('savepath')
@click.option('--up_to', '-u', type=int, default=None)
@click.option('--progress_check', '-p', type=int, default=500000)
def embed(ctx, file, savepath, up_to, progress_check):
    '''Generate embeddings of documents with multiple GPUs
    '''
    if not ctx.obj['GPU']:
        logger.warning("GPU flag not set. Torch will try to embed with 4 cpus")
    # from . import embed_docs
    # from sentence_transformers import SentenceTransformer
    embed_docs(file, savepath, up_to, progress_check)

@preprocess.command()
@click.pass_context
@click.argument('file')
@click.argument('savepath')
@click.option('--up_to', '-u', type=int, default=None)
@click.option('--progress_check', '-p', type=int, default=500000)
@click.option('--embedding_file', '-e', default=None)
def obtain_clusters(ctx, file, savepath, up_to, progress_check, embedding_file):
    if embedding_file:
        #Load sentences & embeddings from disc
        with open(embedding_file, "rb") as fIn:
            stored_data = pickle.load(fIn)
            stored_ids = stored_data['ids']
            stored_embeddings = stored_data['embeddings']
        logger.info('loaded in embeddings')
    else:
        stored_embeddings = None
    _ = filter_by_cluster(
        file,
        savepath,
        up_to=up_to,
        progress_check=progress_check,
        embeddings=stored_embeddings
    )

@preprocess.command()
@click.pass_context
@click.argument('dedup_faiss_file')
@click.argument('data_file')
@click.argument('savepath')
@click.option('--skip/--no_skip', default=False, help='skip reading from hdf5 file if discard list already present')
@click.option('--threshold', '-t', default=0.15, type=float, help='threshold of distance below which articles are considered duplicates')
def remove_dup(ctx, dedup_faiss_file, data_file, savepath, skip, threshold):
    '''remove duplicates given a file of calculated distances'''
    # from . import remove_duplicates
    # import os
    # import h5py
    # import numpy as np
    # import pickle
    remove_duplicates(dedup_faiss_file, data_file, savepath, skip_hdf5_read = skip, threshold=threshold)


@preprocess.command()
@click.pass_context
@click.argument('data_file')
@click.option('--lo', type=int, default=0)
@click.option('--hi')
def remove_by_length(ctx, data_file, lo, hi):
    '''remove by length.'''
    remove_by_len(data_file, lo, hi)


@preprocess.command()
@click.argument('data_file')
@click.option('--outpath', help='folder to save outfile')
@click.option('--id', help='specific id of story to export', default=None, type=int)
@click.option('--format', '-f', help='format to save in.', default='txt')
@click.option('--count', '-c', help= 'number of stories to output', type=int)
def export(data_file, outpath, id, format, count):
    '''export for dedoose use'''
    export_to(data_file, outpath = outpath, id=id, format=format, count=count)

@preprocess.command()
@click.argument('data_file')
@click.argument('savepath')
@click.option('--by', '-b', help='What to sample on', multiple=True)
@click.option('--lang', '-l', help='Languages to sample on. Can be multiple', multiple = True)
@click.option('--total')
def gen_sample(data_file, savepath, by, lang, total):
    '''Sample data for dedoose'''
    sample(data_file, savepath)

if __name__ == '__main__':
    preprocess()