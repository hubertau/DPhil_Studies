import jsonlines
import gzip
import click
import os
import numpy as np
import pandas as pd
import datetime
from collections import Counter, defaultdict

def show_date_range(file, existing_min = None, existing_max = None):
    """Simple function to empirically determine min and max date of jsonl data file.

    Args:
        file (str): path to the file in question.
    """
    if existing_min and existing_max:
        first=False
        min_date = existing_min
        max_date = existing_max
    else:
        first = True
    with jsonlines.open(file, 'r') as reader:
        for story in reader.iter(skip_invalid=True, skip_empty=True):
            if 'query' in story:
                continue
            if 'publish_date' in story:
                story_pub_date = story.get('publish_date')
                if story_pub_date is None:
                    continue
                # sometimes weird formats occur like 2018-02-23 04:36:25.929714
                if '.' in story_pub_date:
                    story_pub_date = story_pub_date.split('.')[0]
                story_pub_date = datetime.datetime.strptime(story_pub_date, "%Y-%m-%d %H:%M:%S")
                if first:
                    min_date = story_pub_date
                    max_date = story_pub_date
                    first = False
                if story_pub_date < min_date:
                    min_date = story_pub_date
                elif story_pub_date > max_date:
                    max_date = story_pub_date

    return min_date, max_date

def show_enrichment_rate(unenriched, enriched):
    '''Print enrichment success rate'''

    results_dict  = Counter()
    language_dict = {}
    date_dict     = {}

    with jsonlines.open(unenriched, 'r') as reader:
        for story in reader.iter(skip_empty=True, skip_invalid=True):
            if 'query' in story:
                continue
            else:
                date = story.get('publish_date')
                if date is None:
                    continue
                if '.' in date:
                    date = date.split('.')[0]
                id = story.get('processed_stories_id')
                results_dict[id]  = 0
                language_dict[id] = story.get('language')
                date_dict[id]     = datetime.datetime.strptime(date, '%Y-%m-%d %H:%M:%S')

    with jsonlines.open(enriched, 'r') as reader:
        for story in reader.iter(skip_empty=True, skip_invalid=True):
            if 'query' in story:
                continue
            else:
                id = story.get('processed_stories_id')
                if id not in results_dict:
                    continue
                # assert id in results_dict, id
                results_dict[id]  = 1

    records = [{
        'id':k,
        'enriched': v,
        'language':language_dict[k],
        'date':date_dict[k]
    } for k, v in results_dict.items()]

    df = pd.DataFrame.from_records(records)

    return df


def main(file):
    text_lengths = []
    with jsonlines.open(file, 'r') as reader:
        for story in reader.iter(skip_empty=True, skip_invalid=True):
            if 'query' in story:
                continue
            text_lengths.append(len(story.get('text')))

    df = pd.DataFrame()
    df['text_lengths'] = text_lengths
    print(df.describe())