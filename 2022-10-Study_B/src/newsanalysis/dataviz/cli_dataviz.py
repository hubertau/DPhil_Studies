'''CLI commands for dataviz
'''

import click
from loguru import logger
import jsonlines
from pprint import PrettyPrinter
import os
from pathlib import Path
import pandas as pd
from iso639 import Lang as isolang
from iso639.exceptions import InvalidLanguageValue

from . import *

@click.group(help='Commands relating to visualisation')
@click.pass_context
def viz(ctx):
    pass

@viz.command()
@click.argument('file')
def get_date_range(file) -> None:
    '''Print date range of data file supplied'''
    min_date, max_date = show_date_range(file)
    logger.info(f'Date range for file is {min_date} to {max_date}')

@viz.command()
@click.argument('file')
@click.option('--id', '-i', help='specific id to retrieve', type=int)
def show_example(file, id) -> None:
    '''Print an example from a data jsonl file'''
    with jsonlines.open(file, 'r') as reader:
        for story in reader.iter(skip_empty=True, skip_invalid=True):
            if 'query' in story:
                continue
            if id is None:
                p = PrettyPrinter(indent=4)
                p.pprint(story)
                break
            else:
                story_id = story.get('processed_stories_id')
                if story_id == id:
                    p = PrettyPrinter(indent=4)
                    p.pprint(story)
                    break

@viz.command()
@click.argument('unenriched')
@click.argument('enriched')
@click.argument('savedir')
def get_enrichment_rate(unenriched, enriched, savedir):
    '''Save enrichment success rate data to csv'''

    df = show_enrichment_rate_data(unenriched, enriched)
    df.to_csv(Path(savedir) / 'desc_enrichment.csv')


@viz.command()
@click.argument('enrichment_csv')
def show_enrichment_rate(enrichment_csv):
    """Print overall enrichment rate and by language. Must be run after get-enrichment-rate"""
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


@viz.command()
@click.argument('enrichment_csv')
@click.argument('savedir')
@click.option('--min_count', default=0)
def plot_success_rate(enrichment_csv, savedir, min_count):
    """Plot the success rate by language."""
    df = pd.read_csv(enrichment_csv)

    by_lang = df.groupby('language').agg(
        enriched_sum   = pd.NamedAgg(column='enriched', aggfunc='sum'),
        enriched_count = pd.NamedAgg(column='enriched', aggfunc='count')
    )
    by_lang['success_rate'] = 100*by_lang['enriched_sum']/by_lang['enriched_count']

    plot_success_rate_by_lang(by_lang.reset_index(), savedir, min_count = min_count)

@viz.command()
@click.argument('enrichment_csv')
@click.argument('savedir')
@click.option('--min_count', default=0)
def plot_success_rate_time(enrichment_csv, savedir, min_count):
    '''Plot success rate by time and language'''
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
    plot_success_rate_by_lang_and_time(by_lang,savedir, min_count=min_count)

@viz.command()
@click.argument('enrichment_csv')
@click.argument('savedir')
@click.option('--min_count', default=0)
def plot_enriched_count_lt(enrichment_csv, savedir, min_count):
    '''Plot enriched count by language and time'''
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
    plot_enriched_count_by_lang_and_time(by_lang,savedir, min_count=min_count)

@viz.command()
@click.argument('file')
@click.argument('outdir')
def get_pub_list(file, outdir):
    '''Save pub list to csv file'''
    df = retrieve_pubs(file)
    savepath = os.path.join(outdir, 'desc_pubs.csv')
    df.to_csv(savepath)
    logger.info(f'Saved to {savepath}')

@viz.command()
@click.argument('file')
def show_pub_summary(file):
    '''Show publisher distribution summary'''
    if file.endswith('.csv'):
        df = pd.read_csv(file)
    else:
        df = retrieve_pubs(file)
    logger.info(df.describe())

@viz.command()
@click.argument('file')
def show_top_pubs(file, top_n = 5):
    '''Show top n publishers by language'''

    if file.endswith('.csv'):
        df = pd.read_csv(file)
    else:
        df = retrieve_pubs(file)

    df = df.groupby(['language', 'id']).agg(
        name=pd.NamedAgg(column='name', aggfunc='first'),
        story_count = pd.NamedAgg(column='story_count', aggfunc='sum')
    ).reset_index()
    df.sort_values('story_count', ascending=False, inplace=True)

    agged = df.groupby('language').head(top_n)

    logger.info(agged)

@viz.command()
@click.argument('file')
def show_pub_by_lang(file):
    '''Show summary of publishers by language
    '''
    if file.endswith('.csv'):
        df = pd.read_csv(file)
    else:
        df = retrieve_pubs(file) 
    logger.info(df.groupby('language').sum()['story_count'])

@viz.command()
@click.argument('file')
@click.argument('savedir')
@click.option('--min_count', default=0, required=False)
def plot_pubs_histogram(file, savedir, min_count):
    '''Plot publisher histogram. Not by language
    '''
    if file.endswith('.csv'):
        df = pd.read_csv(file)
    else:
        df = retrieve_pubs(file)
    plot_pub_hist(df, savedir, min_count=min_count)

@viz.command()
@click.argument('file')
@click.argument('savedir')
@click.option('--min_count', default=0, required=False)
def plot_pubs_lang(file, savedir, min_count):
    '''Plot publisher histogram. Not by language'''
    if file.endswith('.csv'):
        df = pd.read_csv(file)
    else:
        df = retrieve_pubs(file)
    plot_pub_by_lang(df, savedir, min_count=min_count)

@viz.command()
@click.argument('file')
def show_count(file):
    '''Show count of stories in data file'''
    counter = 0
    with jsonlines.open(file, 'r') as reader:
        for story in reader.iter(skip_invalid=True, skip_empty=True):
            if 'query' in story:
                continue
            else:
                counter += 1
    logger.info(f'{counter} stories in file')

@viz.command()
@click.pass_context
@click.argument('file')
def show_langs(ctx, file):
    '''Show languages prsent in data file
    '''
    df = retrieve_story_and_lang(file)

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

@viz.command()
@click.argument('data_file')
@click.argument('savedir')
def get_story_and_lang(data_file, savedir):
    '''Save stories with language information'''

    '''Show languages prsent in data file
    '''
    df = retrieve_story_and_lang(data_file)

    savename = Path(savedir) / f'{Path(data_file).stem}_stories_and_langs.csv'
    df.to_csv(savename)



if __name__ == '__main__':
    viz()