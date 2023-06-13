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
@click.option('--nr_topics', '-n', type=int, default=None)
@click.option('--top_n_words', '-t', type=int, default=10)
@click.option('--embedding_file', '-e', default=None)
def obtain_clusters(ctx, file, savepath, up_to, progress_check, nr_topics, top_n_words, embedding_file):
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
        nr_topics = nr_topics,
        top_n_words=top_n_words,
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
@click.option('--source', '-s')
@click.option('--outpath', help='folder to save outfile')
@click.option('--id', help='specific id of story to export', default=None, type=int)
@click.option('--format', '-f', help='format to save in.', default='txt')
@click.option('--count', '-c', help= 'number of stories to output', type=int)
def export(data_file, source, outpath, id, format, count):
    '''export for dedoose use'''
    export_to(data_file, source = source, outpath = outpath, id=id, format=format, count=count)

@preprocess.command()
@click.argument('data_file')
@click.argument('savepath')
@click.option('--by', '-b', help='What to sample on', multiple=True)
@click.option('--lang', '-l', help='Languages to sample on. Can be multiple', multiple = True)
@click.option('--total', '-t', help='Total number of articles to have sampled by the end.', type=int)
@click.option('--exclude', '-e', help='File of existing sample to exclude', multiple=True)
@click.option('--min_date', help='min date')
@click.option('--max_date', help='min date')
def gen_sample(data_file, savepath, by, lang, total, exclude, min_date, max_date):
    '''Sample data for dedoose'''
    sample(data_file, savepath, lang=lang, total=total, exclude=exclude, min_date= min_date, max_date = max_date)


@preprocess.command()
@click.argument('data_file')
@click.argument('bertopic_file')
@click.argument('outfile')
@click.option('--remove', '-r', help='ids for each group to remove', multiple=True, required=True, type=int)
def remove_by_bertopic(data_file, bertopic_file, outfile, remove):
    '''remove by bertopic result'''

    remove_by_bt(data_file, bertopic_file, outfile, remove)

@preprocess.command()
@click.argument('jsonl_file')
@click.argument('dataset_out_path')
@click.option('--keys', '-k', multiple=True)
@click.option('--split/--no-split', default=True)
@click.option('--up_to', type=int)
def to_dataset(jsonl_file, dataset_out_path, keys, split, up_to):
    '''Convert jsonl to Datasets object'''
    print(keys)
    logger.info(f'Splitting is {split}')
    jsonl_to_dataset(
        jsonl_file=jsonl_file,
        dataset_out=dataset_out_path,
        keys_to_read=keys,
        splitting=split,
        up_to=up_to
    )

@preprocess.command()
@click.argument('dataset_path')
@click.argument('outpath')
@click.option('--model', '-m', default='51la5/roberta-large-NER')
@click.option('--tok', '-t', default=None)
@click.option('--num_batches', '-n', default=None, type=int)
@click.option('--kind', '-k', default = 'ner')
@click.option('--batchsizepergpu', '-b', default=800, type=int, help='Batch Size per GPU')
@click.option('--from_batch', '-f', default=None, type=int)
@click.option('--rel_filter', '-r', default=None)
@click.option('--no-logits/logits', '-l', default=True)
def annot(dataset_path, outpath, model, tok, num_batches, kind, batchsizepergpu, from_batch, rel_filter):
    '''Apply NER'''
    annotate(
        dataset_path,
        outpath,
        model=model,
        tok = tok,
        rel_filter = rel_filter,
        num_batches=num_batches,
        kind = kind,
        batch_size_per_gpu = batchsizepergpu,
        from_batch = from_batch
    )

@preprocess.command()
@click.argument('ner_batch_path')
@click.argument('outpath')
@click.option('--omit_tokens', '-o', default=['<pad>'])
def combine_ner(ner_batch_path, outpath, omit_tokens):
    '''Combine resulting NER batches'''

    collate_ner(
        ner_batch_dir=ner_batch_path,
        outpath = outpath,
        omit_tokens = omit_tokens
    )

if __name__ == '__main__':
    preprocess()