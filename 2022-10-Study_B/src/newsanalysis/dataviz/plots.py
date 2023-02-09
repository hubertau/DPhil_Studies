import matplotlib.pyplot as plt
import seaborn as sns
import jsonlines
import pandas as pd
import os
import numpy as np
from dataclasses import dataclass, asdict, field
from collections import Counter
import unicodedata

@dataclass(frozen=True)
class publisher:
    name: str = field(compare=False)
    id: str = field(compare=True)
    url: str = field(compare=False)
    language: str = field(compare=True)

@dataclass(frozen=True)
class storyclass:
    id: str=field(compare=True)
    lang: str=field(compare=False)
    length: int=field(compare=False, default=None)

def retrieve_pubs(file):

    all_pubs = Counter()
    with jsonlines.open(file, 'r') as reader:
        for story in reader.iter(skip_empty=True, skip_invalid=True):
            if 'query' in story:
                continue
            else:
                all_pubs[publisher(
                    name     = story.get('media_name'),
                    id       = story.get('media_id'),
                    url      = story.get('media_url'),
                    language = story.get('language'))
                ] += 1

    records = [{**asdict(k),**{'story_count':v}} for k, v in all_pubs.items()]

    df = pd.DataFrame.from_records(records)

    return df

def calc_story_len(text, lang):

    length = 0
    if lang in ['zh', 'ko', 'ja']:
        for i in text:
            if unicodedata.name(i, False):
                length+=1
    else:
        length = len(text.split(' '))
    return length

def retrieve_story_and_lang(file):
    '''Extract just language and story id from data file.'''

    all_stories = []
    with jsonlines.open(file, 'r') as reader:
        for story in reader.iter(skip_empty=True, skip_invalid=True):
            if 'query' in story:
                continue
            else:
                all_stories.append(storyclass(
                    id     = story.get('processed_stories_id'),
                    lang   = story.get('language')
                ))

    df = pd.DataFrame.from_records([asdict(k) for k in all_stories])

    return df

def retrieve_story_lens(file):
    '''Extract information about story length from data file.

    Note that different languages will need different counting strategies
    '''

    all_stories = []
    with jsonlines.open(file, 'r') as reader:
        for story in reader.iter(skip_empty=True, skip_invalid=True):
            if 'query' in story:
                continue
            else:
                all_stories.append(storyclass(
                    id     = story.get('processed_stories_id'),
                    lang   = story.get('language'),
                    length = calc_story_len(
                        story.get('text'),
                        story.get('language')
                    )
                ))

    df = pd.DataFrame.from_records([asdict(k) for k in all_stories])

    return df

def plot_pub_hist(df, savedir, min_count = 0):

    fig = plt.figure(figsize=(15,8))
    plot_df = df[df['story_count']>min_count]
    plot_df = plot_df.groupby('id').sum()
    sns.set_theme()
    sns.histplot(
        data=plot_df,
        x = 'story_count',
        bins=30,
        log_scale=(True,True)
    )

    plt.savefig(os.path.join(savedir, 'desc_pub_hist.png'), bbox_inches='tight', dpi=300)

def plot_pub_by_lang(df, savedir, min_count=0):

    fig = plt.figure(figsize=(15,8))
    plot_df = df[df['story_count']>min_count]
    plot_df = plot_df.groupby('language').sum().reset_index()
    plot_df.sort_values('story_count', ascending=False, inplace=True)

    sns.set_theme()
    g = sns.barplot(
        data=plot_df,
        x = 'language',
        y = 'story_count'
    )
    g.set_yscale("log")

    plt.savefig(os.path.join(savedir, 'desc_pub_by_lang.png'), bbox_inches='tight', dpi=300)

def plot_success_rate_by_lang(df, savedir, min_count = 0):

    df = df.sort_values('success_rate', ascending=False)
    df = df[df['enriched_count'] > min_count]

    fig  = plt.figure(figsize=(15,8))
    sns.set_theme()
    g = sns.barplot(
        data=df,
        x='language',
        y = 'success_rate'
    )
    g.set_xticklabels(g.get_xticklabels(), rotation=45, horizontalalignment='right')
    plt.savefig(os.path.join(savedir, 'desc_success_rate_by_lang.png'), bbox_inches='tight', dpi=300)


def plot_success_rate_by_lang_and_time(df, savedir, min_count = 0):

    df = df[df['enriched_count'] > min_count]
    df.loc[:,'month'] = pd.to_datetime(df.loc[:,'month'])

    fig  = plt.figure(figsize=(15,8))
    sns.set_theme()
    g = sns.lineplot(
        data=df,
        x = 'month',
        y = 'success_rate',
        hue = 'language'
    )
    # g.set_xticklabels(g.get_xticklabels(), rotation=45)
    plt.xticks(rotation=45, ha='right')
    plt.savefig(os.path.join(savedir, 'desc_success_rate_by_lang_and_time.png'), bbox_inches='tight', dpi=300)

def plot_enriched_count_by_lang_and_time(df, savedir, min_count = 0):

    df = df[df['enriched_count'] > min_count]
    df.loc[:,'month'] = pd.to_datetime(df.loc[:,'month'])

    fig  = plt.figure(figsize=(15,8))
    sns.set_theme()
    g = sns.lineplot(
        data=df,
        x = 'month',
        y = 'enriched_count',
        hue = 'language'
    )
    g.set_yscale("log")
    # g.set_xticklabels(g.get_xticklabels(), rotation=45)
    plt.xticks(rotation=45, ha='right')
    plt.savefig(os.path.join(savedir, 'desc_enrichment_count_by_lang_and_time.png'), bbox_inches='tight', dpi=300)

def plot_story_length_boxplot(df, savedir):

    fig = plt.figure(figsize=(15,8))
    sns.set_theme()
    g = sns.boxplot(
        x = df['length']
    )
    plt.savefig(os.path.join(savedir, 'desc_story_len_boxplot.png'), bbox_inches='tight', dpi=300)

def plot_story_length_barplot_by_lang(df, savedir):
    pass