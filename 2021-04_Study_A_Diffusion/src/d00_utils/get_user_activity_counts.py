#!/usr/bin/python3.9

'''
Return user activity levels, on a daily count.
'''

import argparse
import datetime
import glob
import pandas as pd
import logging
import os
import re
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from typing import NamedTuple
import numpy as np

import h5py
import jsonlines


class OutputTuple(NamedTuple):
    author_id: str
    normal: Counter
    hashtagged: defaultdict

class DateRange(NamedTuple):
    start: datetime.datetime
    end: datetime.datetime

# set up recursive defaultdict
def nested_dd():
    return defaultdict(int)

def one_user_activity(timeline_file, augmented_file, date_range, search_hashtags):

    tweet_counts = []
    output_hashtagged = defaultdict(nested_dd)
    with jsonlines.open(timeline_file) as reader:
        for tweet_json in reader:
            for tweet_data in tweet_json['data']:
                tweet_created_at = datetime.datetime.fromisoformat(tweet_data['created_at'][:-1]).date()
                if date_range.start <= tweet_created_at <= date_range.end:
                    tweet_counts.append(tweet_created_at)
        author_id = tweet_data['author_id']
    with jsonlines.open(augmented_file) as reader:
        for tweet in reader:
            tweet_created_at = datetime.datetime.fromisoformat(tweet['created_at'][:-1]).date()
            try:
                tweet_hts = [i['tag'].lower() for i in tweet['entities']['hashtags']]
            except KeyError:
                continue
            tags = [i for i in tweet_hts if i in search_hashtags]
            if date_range.start <= tweet_created_at <= date_range.end:
                # output_hashtagged.append(tweet_created_at)
                for t in tags:
                    output_hashtagged[tweet_created_at][t] += 1
            author_id = tweet['author_id']

    output = Counter(tweet_counts)

    logging.info(f'Processed {author_id}')

    return OutputTuple(author_id, output, output_hashtagged)

def main(args):

    # collect data files
    timeline_file_list = sorted(glob.glob(os.path.join(args.data_dir, 'timeline*.jsonl')))
    augmented_file_list = sorted(glob.glob(os.path.join(args.data_dir, 'augmented*.jsonl')))

    assert os.path.isdir(args.output_dir)

    # extract start and stop times from FAS peak analysis and group number
    with h5py.File(args.FAS_peak_analysis_file, 'r') as f:
        x = f['segments']['selected_ranges'][args.group_num-1]
        args.date_range_start = x[0].decode()
        args.date_range_end = x[1].decode()

    # process date range
    date_range = DateRange(
        datetime.datetime.strptime(args.date_range_start, "%Y-%m-%d").date(),
        datetime.datetime.strptime(args.date_range_end, "%Y-%m-%d").date()
    )
    logging.debug(f'Processed start time: {date_range.start}')
    logging.debug(f'Processed end time: {date_range.end}')

    # sanity checks
    try:
        assert len(timeline_file_list) == len(augmented_file_list)
        assert all([
            re.split('[_.]', timeline_file_list[i])[-2] == re.split('[_.]',augmented_file_list[i])[-2] for i in range(len(timeline_file_list))
        ])
    except AssertionError:
        logging.error(f'Assertion error on timeline and augmented jsonl files', exc_info=True)
        raise

    with open(args.search_hashtags, 'r') as f:
        search_hashtags = f.readlines()
        search_hashtags = [i.rstrip('\n') for i in search_hashtags]
        search_hashtags = [i.lower() for i in search_hashtags]
        search_hashtags = [i.replace('#', '') for i in search_hashtags]

    with ProcessPoolExecutor() as executor:
        results = executor.map(one_user_activity,
            timeline_file_list,
            augmented_file_list,
            repeat(date_range),
            repeat(search_hashtags)
        )

    # generate daterange for writing to file
    date_range_enumerated = pd.date_range(
        start=args.date_range_start,
        end=args.date_range_end
    ).to_pydatetime().tolist()

    date_range_enumerated = [i.date() for i in date_range_enumerated]

    output_filename = os.path.join(args.output_dir, 'activity_counts.hdf5')
    with h5py.File(output_filename, 'a') as f:
        if args.overwrite:
            if f'group_{args.group_num}' in f.keys():
                del f[f'group_{args.group_num}']
        g = f.create_group(f'group_{args.group_num}')
        for t in results:
            normal_npy = np.array([t.normal[date] for date in date_range_enumerated])

            metadata = {
                'author_id': t.author_id,
                'date_range_start': args.date_range_start,
                'date_range_end': args.date_range_end
            }

            h = g.create_group(t.author_id)
            normal_dset = h.create_dataset('normal', data=normal_npy)
            normal_dset.attrs.update(metadata)

            # create numpy array, rows = hashtags, columns = dates
            hashtagged_npy = np.zeros((len(search_hashtags), len(date_range_enumerated)))

            hashtagged_metadata = {
                'author_id': t.author_id,
                'date_range_start': args.date_range_start,
                'date_range_end': args.date_range_end,
                'feature_order': ';'.join(search_hashtags)
            }

            # fill each row, i.e. by hashtag
            for date_index, date in enumerate(date_range_enumerated):
                for search_ht_index, search_ht in enumerate(search_hashtags):
                    hashtagged_npy[search_ht_index][date_index] = t.hashtagged[date][search_ht]

            hashtagged_dset = h.create_dataset('hashtagged', data=hashtagged_npy)
            hashtagged_dset.attrs.update(hashtagged_metadata)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Extract user tweet counts from timeline and augmented files.')

    parser.add_argument(
        'data_dir',
        help='directory of timeline and augmented files'
    )

    parser.add_argument(
        'output_dir',
        help='output directory of results'
    )

    parser.add_argument(
        'group_num',
        help='Group number',
        type=int
    )

    parser.add_argument(
        'FAS_peak_analysis_file',
        help='FAS peak analysis hdf5'
    )

    parser.add_argument(
        '--overwrite',
        help='overwrite existing group',
        action='store_true'
    )

    parser.add_argument(
        '--search_hashtags',
        help='search hastags file',
        default = '../../references/search_hashtags.txt'
    )

    parser.add_argument(
        '--log_dir',
        help='director to place log in. Defaults to $HOME',
        default='$HOME'
    )

    parser.add_argument(
        '--log_level',
        help='logging_level',
        type=str.upper,
        choices=['INFO','DEBUG','WARNING','CRITICAL','ERROR','NONE'],
        default='DEBUG'
    )

    args = parser.parse_args()

    logging_dict = {
        'NONE': None,
        'CRITICAL': logging.CRITICAL,
        'ERROR': logging.ERROR,
        'WARNING': logging.WARNING,
        'INFO': logging.INFO,
        'DEBUG': logging.DEBUG
    }

    logging_level = logging_dict[args.log_level]

    if logging_level is not None:

        logging_fmt   = '[%(levelname)s] %(asctime)s - %(message)s'
        today_datetime = str(datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S'))
        logging_file  = os.path.join(args.log_dir, f'{today_datetime}_user_activity.log')
        logging.basicConfig(
            handlers=[
                logging.FileHandler(filename=logging_file,mode='w'),
                logging.StreamHandler()
            ],
            format=logging_fmt,
            level=logging_level,
            datefmt='%m/%d/%Y %I:%M:%S %p'
        )

        logging.info(f'Start time of user activity script is {today_datetime}')

    logging.info(f'group num is {args.group_num}')
    logging.info(f'Overwrite flag is {args.overwrite=}')

    main(args)
