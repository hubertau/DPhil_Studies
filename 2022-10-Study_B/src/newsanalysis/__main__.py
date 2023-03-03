import newsanalysis.data_utils
import newsanalysis.dataviz
import jsonlines
import click
import os
import numpy as np
import pickle
import datetime
from loguru import logger
import sys
from iso639 import Lang as isolang
from iso639.exceptions import InvalidLanguageValue
import glob
from pprint import PrettyPrinter
import h5py
import pandas as pd
from collections import Counter, defaultdict
import logging

# intercept default logging package from submodules (e.g. SentenceTrasnformers)
# from recipe in loguru: https://loguru.readthedocs.io/en/stable/overview.html
class InterceptHandler(logging.Handler):
    def emit(self, record):
        # Get corresponding Loguru level if it exists
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Find caller from where originated the logged message
        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())

logging.basicConfig(handlers=[InterceptHandler()], level=0)

@click.group()
@click.pass_context
@click.option('--debug/--no-debug', default=False)
@click.option('--gpu/--no-gpu', default=False)
@click.option('--log_file')
def cli(ctx, debug, gpu, log_file):
    """News Analysis package.

    """
    logger.info(f"Debug mode is {'on' if debug else 'off'}")
    if not debug:
        logger.remove()
        logger.add(sys.stderr, level="INFO", backtrace=True, diagnose=True)
    if log_file:
        logger.add(log_file, backtrace=True, diagnose=True)

    ctx.obj = {}
    ctx.obj['DEBUG'] = debug
    ctx.obj['GPU'] = gpu

@cli.command()
@click.argument('file')
def get_date_range(file) -> None:
    '''Print date range of data file supplied'''
    min_date, max_date = newsanalysis.dataviz.show_date_range(file)
    logger.info(f'Date range for file is {min_date} to {max_date}')


@cli.command()
@click.argument('file')
def get_example(file) -> None:
    '''Print an example from a data jsonl file'''
    with jsonlines.open(file, 'r') as reader:
        for story in reader.iter(skip_empty=True, skip_invalid=True):
            if 'query' in story:
                continue
            else:
                p = PrettyPrinter(indent=4)
                p.pprint(story)
                break

@cli.command()
@click.argument('unenriched')
@click.argument('enriched')
@click.argument('savedir')
def get_enrichment_rate(unenriched, enriched, savedir):
    '''Save enrichment success rate data to csv'''

    df = newsanalysis.dataviz.show_enrichment_rate(unenriched, enriched)
    df.to_csv(os.path.join(savedir, 'desc_enrichment.csv'))


@cli.command()
@click.argument('enrichment_csv')
def show_enrichment_rate(enrichment_csv):
    """Print overall enrichment rate and by language"""
    df = pd.read_csv(enrichment_csv)
    total_unenriched = len(df)
    total_enriched   = sum(df['enriched'] == 1)

    by_lang = df.groupby('language').agg(
        enriched_sum   = pd.NamedAgg(column='enriched', aggfunc='sum'),
        enriched_count = pd.NamedAgg(column='enriched', aggfunc='count')
    )
    by_lang['success_rate'] = 100*by_lang['enriched_sum']/by_lang['enriched_count']

    logger.info(f'Total unenriched: {total_unenriched}')
    logger.info(f'Total enriched: {total_enriched}')
    logger.info(f'Overall enrichment success rate: {100*total_enriched/total_unenriched:.2f}%')
    logger.info(f'Success rate by language:')
    logger.info(f'{by_lang}')

@cli.command()
@click.argument('enrichment_csv')
@click.argument('savedir')
@click.option('--min_count', default=0)
def plot_success_rate(enrichment_csv, savedir, min_count):
    df = pd.read_csv(enrichment_csv)

    by_lang = df.groupby('language').agg(
        enriched_sum   = pd.NamedAgg(column='enriched', aggfunc='sum'),
        enriched_count = pd.NamedAgg(column='enriched', aggfunc='count')
    )
    by_lang['success_rate'] = 100*by_lang['enriched_sum']/by_lang['enriched_count']

    newsanalysis.dataviz.plot_success_rate_by_lang(by_lang.reset_index(), savedir, min_count = min_count)

@cli.command()
@click.argument('enrichment_csv')
@click.argument('savedir')
@click.option('--min_count', default=0)
def plot_success_rate_time(enrichment_csv, savedir, min_count):
    df = pd.read_csv(enrichment_csv)
    df['date'] = pd.to_datetime(df['date'], format = '%Y-%m-%d %H:%M:%S')

    df['month'] = df.date.dt.to_period('M')
    # df.sort_values('month', ascending=False, inplace=True)

    by_lang = df.groupby(['language', 'month']).agg(
        enriched_sum   = pd.NamedAgg(column='enriched', aggfunc='sum'),
        enriched_count = pd.NamedAgg(column='enriched', aggfunc='count')
    )
    by_lang['success_rate'] = 100*by_lang['enriched_sum']/by_lang['enriched_count']
    by_lang = by_lang.reset_index()
    by_lang['month'] = by_lang['month'].astype(str)
    newsanalysis.dataviz.plot_success_rate_by_lang_and_time(by_lang,savedir, min_count=min_count)

@cli.command()
@click.argument('enrichment_csv')
@click.argument('savedir')
@click.option('--min_count', default=0)
def plot_enriched_count_by_lang_and_time(enrichment_csv, savedir, min_count):
    df = pd.read_csv(enrichment_csv)
    df['date'] = pd.to_datetime(df['date'], format = '%Y-%m-%d %H:%M:%S')

    df['month'] = df.date.dt.to_period('M')
    # df.sort_values('month', ascending=False, inplace=True)

    by_lang = df.groupby(['language', 'month']).agg(
        enriched_sum   = pd.NamedAgg(column='enriched', aggfunc='sum'),
        enriched_count = pd.NamedAgg(column='enriched', aggfunc='count')
    )
    by_lang['success_rate'] = 100*by_lang['enriched_sum']/by_lang['enriched_count']
    by_lang = by_lang.reset_index()
    by_lang['month'] = by_lang['month'].astype(str)
    newsanalysis.dataviz.plot_enriched_count_by_lang_and_time(by_lang,savedir, min_count=min_count)

@cli.command()
@click.argument('file')
@click.argument('outdir')
def get_pub_list(file, outdir):
    '''Save pub list to csv file'''
    df = newsanalysis.dataviz.retrieve_pubs(file)
    savepath = os.path.join(outdir, 'desc_pubs.csv')
    df.to_csv(savepath)
    logger.info(f'Saved to {savepath}')

@cli.command()
@click.argument('file')
def show_pub_summary(file):
    '''Show publisher distribution summary
    '''
    if file.endswith('.csv'):
        df = pd.read_csv(file)
    else:
        df = newsanalysis.dataviz.retrieve_pubs(file)
    logger.info(df.describe())

@cli.command()
@click.argument('file')
def show_top_pubs(file, top_n = 5):
    '''Show top n publishers by language'''

    if file.endswith('.csv'):
        df = pd.read_csv(file)
    else:
        df = newsanalysis.dataviz.retrieve_pubs(file)

    df = df.groupby(['language', 'id']).agg(
        name=pd.NamedAgg(column='name', aggfunc='first'),
        story_count = pd.NamedAgg(column='story_count', aggfunc='sum')
    ).reset_index()
    df.sort_values('story_count', ascending=False, inplace=True)

    agged = df.groupby('language').head(top_n)

    logger.info(agged)

@cli.command()
@click.argument('file')
def show_pub_by_lang(file):
    '''Show summary of publishers by language
    '''
    if file.endswith('.csv'):
        df = pd.read_csv(file)
    else:
        df = newsanalysis.dataviz.retrieve_pubs(file) 
    logger.info(df.groupby('language').sum()['story_count'])

@cli.command()
@click.argument('file')
@click.argument('savedir')
@click.option('--min_count', default=0, required=False)
def plot_pubs_hist(file, savedir, min_count):
    '''Plot publisher histogram. Not by language
    '''
    if file.endswith('.csv'):
        df = pd.read_csv(file)
    else:
        df = newsanalysis.dataviz.retrieve_pubs(file)
    newsanalysis.dataviz.plot_pub_hist(df, savedir, min_count=min_count)

@cli.command()
@click.argument('file')
@click.argument('savedir')
@click.option('--min_count', default=0, required=False)
def plot_pubs_by_lang(file, savedir, min_count):
    '''Plot publisher histogram. Not by language
    '''
    if file.endswith('.csv'):
        df = pd.read_csv(file)
    else:
        df = newsanalysis.dataviz.retrieve_pubs(file)
    newsanalysis.dataviz.plot_pub_by_lang(df, savedir, min_count=min_count)

# @get_media.command()
# @click.pass_context
# def collect(ctx):
#     pass

@cli.command()
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
        min_date, max_date = newsanalysis.dataviz.show_date_range(file, min_date, max_date)
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

@cli.command()
@click.pass_context
@click.argument('file')
@click.argument('savepath')
def remove_redundant_ids(ctx, file, savepath):
    newsanalysis.data_utils.remove_redundant_ids(file, savepath)

@cli.command()
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
    newsanalysis.data_utils.deduplicate(file, savepath, gpu=ctx.obj['GPU'])
    logger.info('done')


@cli.command()
@click.pass_context
@click.argument('file')
def show_langs(ctx, file):
    df = newsanalysis.data_utils.retrieve_story_and_lang(file)

    # MediaCloud say they support the following languages:
    langdict = {
        "ca":"Catalan",
        "da":"Danish",
        "de":"German",
        "en":"English",
        "es":"Spanish",
        "fi":"Finnish",
        "fr":"French",
        "ha":"Hausa",
        "hi":"Hindi",
        "hu":"Hungarian",
        "it":"Italian",
        "ja":"Japanese",
        "lt":"Lithuanian",
        "nl":"Dutch",
        "no":"Norwegian",
        "pt":"Portuguese",
        "ro":"Romanian",
        "ru":"Russian",
        "sv":"Swedish",
        "tr":"Turkish",
        "zh":"Chinese"
    }

    for empirical_lang in df['lang'].unique():
        if empirical_lang is None:
            logger.info('"None" language present.')
            continue
        try:
            this_lang = isolang(empirical_lang)
            logger.info(f'{empirical_lang}: {this_lang.name} present. {"YES MediaCloud" if empirical_lang in langdict else "NO MediaCloud"}')
        except InvalidLanguageValue:
            logger.info(f'{empirical_lang} not a valid iso639 code')


@cli.command()
@click.pass_context
@click.argument('file')
@click.argument('savepath')
@click.option('--up_to', '-u', type=int, default=None)
@click.option('--progress_check', '-p', type=int, default=100000)
def embed(ctx, file, savepath, up_to, progress_check):
    '''Generate embeddings of documents with multiple GPUs
    '''
    if not ctx.obj['GPU']:
        logger.warning("GPU flag not set. Torch will try to embed with 4 cpus")

    newsanalysis.data_utils.embed_docs(file, savepath, up_to, progress_check)

@cli.command()
@click.pass_context
@click.argument('file')
@click.argument('savepath')
@click.option('--up_to', '-u', type=int, default=None)
@click.option('--progress_check', '-p', type=int, default=100000)
@click.option('--embedding_file', '-e', default=None)
def obtain_clusters(ctx, file, savepath, up_to, progress_check, embedding_file):
    if embedding_file:
        #Load sentences & embeddings from disc
        with open(embedding_file, "rb") as fIn:
            stored_data = pickle.load(fIn)
            stored_ids = stored_data['ids']
            stored_embeddings = stored_data['embeddings']
        logger.info('loaded in embeddings')
    topics, probs = newsanalysis.data_utils.filter_by_cluster(
        file,
        savepath,
        up_to=up_to,
        progress_check=progress_check,
        embeddings=stored_embeddings
    )
    savename = os.path.join(savepath, 'bertopic_cluster.hdf5')
    with h5py.File(savename, 'w') as f:
        f.create_dataset('topics', data=topics)
        f.create_dataset('probs', data=probs)

@cli.command()
@click.pass_context
@click.argument('dedup_faiss_file')
@click.argument('data_file')
@click.argument('savepath')
def remove_duplicates(ctx, dedup_faiss_file, data_file, savepath):
    newsanalysis.data_utils.remove_duplicates(dedup_faiss_file, data_file, savepath)

if __name__ == '__main__':
    cli()