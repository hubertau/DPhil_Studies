#!/usr/bin/python3.9

'''
Script to fill out stats df columns with activity counts.
'''

import logging
import argparse
import os
import datetime
import h5py
from typing import NamedTuple
from concurrent.futures import ProcessPoolExecutor, process
import numpy as np
from time import perf_counter
import pandas as pd
from itertools import repeat

class PPE_output(NamedTuple):
    ht: str
    data: pd.Series

class daterange(NamedTuple):
    start: str
    end: str

# insert activity of users
def date_to_array_index(date, daterange):
    return (date - datetime.datetime.strptime(daterange.start, '%Y-%m-%d').date()).days

def insert_activity_of_user(row,
        daterange=None,
        activity_file = None,
        activity='hashtagged',
        hashtag=None,
        group_num = 1,
        ht_row_mapping = None
    ):

        if activity == 'hashtagged':
            if row[f'peak_{hashtag}']:
                after = date_to_array_index(row['created_at'], daterange)
                # get ht row index
                assert hashtag in ht_row_mapping
                ht_index = ht_row_mapping.index(hashtag)
                result = activity[f'group_{group_num}'][row['author_id']][activity][ht_index,after:]
            else:
                after = date_to_array_index(row['created_at'], daterange)
                # get ht row index
                assert hashtag in ht_row_mapping
                ht_index = ht_row_mapping.index(hashtag)
                result = activity[f'group_{group_num}'][row['author_id']][activity][ht_index,after:]
        else:
            if row[f'peak_{hashtag}']:
                after = date_to_array_index(row['created_at'], daterange)
                # print(f'group_{group_num}', row['author_id'], after)
                result = activity_file[f'group_{group_num}'][row['author_id']][activity][after:]
            else:
                up_to = date_to_array_index(row['created_at'], daterange)
                # print(f'group_{group_num}', row['author_id'], up_to)
                result = activity_file[f'group_{group_num}'][row['author_id']][activity][:up_to]

        if len(result) > 0:
            return np.sum(result)/len(result)
        else:
            return 0

def process_one_hashtag(df, group_daterange, hashtag):

    start_time = perf_counter()
    logging.info(f'Begin processing {hashtag}')

    with h5py.File(args.activity_file, 'r') as f:
        res = df.apply(
                insert_activity_of_user,
                daterange = group_daterange,
                activity_file = f,
                hashtag = hashtag,
                activity='normal',
                axis=1
            )

    end_time = perf_counter()
    logging.info(f'End processing {hashtag} after {end_time-start_time}')
    return PPE_output(ht=hashtag, data=res)


def main(args):

    logging.info(f'Reading in df from {args.df_file}')
    df = pd.read_hdf(args.df_file, f'group_{args.group_num}')
    logging.info('Done')

    def daterange_from_group_num(group_num):
        with h5py.File(args.FAS_peak_analysis_file, 'r') as f:
            x = f['segments']['selected_ranges'][int(group_num)-1]
            res = daterange(
                start = x[0].decode(),
                end = x[1].decode()
            )
        return res

    group_daterange = daterange_from_group_num(args.group_num)
    logging.info(f'group num is {args.group_num}')
    logging.info(f'Associated date range is {group_daterange}')

    with open(args.search_hashtags, 'r') as f:
        search_hashtags = f.readlines()
        search_hashtags = [i.replace('\n', '') for i in search_hashtags]
        search_hashtags = [i.replace('#', '') for i in search_hashtags]
        search_hashtags = [i.lower() for i in search_hashtags]
        search_hashtags.remove('وأناكمان')

    with ProcessPoolExecutor(max_workers=5) as executor:
        results = executor.map(process_one_hashtag, repeat(df), repeat(group_daterange), search_hashtags)

    logging.info('ProcessPoolExecutor complete. Assigning df columns')
    for i in results:
        df[f'activity_{i.ht}_normal'] = i.data

    with open(args.df_file, 'a') as f:
        del f[f'group_{args.group_num}']

    logging.info('Saving to same df file')
    df.to_hdf(args.df_file, f'group_{args.group_num}')

    logging.info('Verify - new columns of df:')
    cols = [i for i in df.columns if 'activity' in i]
    logging.info(f'{cols}')
    logging.info('End of script.')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='append activity counts to df')

    parser.add_argument(
        '--df_file',
        help='preprocessed df file'
    )

    parser.add_argument(
        '--activity_file',
        help='activity file'
    )

    parser.add_argument(
        '--FAS_peak_analysis_file'
    )

    parser.add_argument(
        '--group_num',
        type=int
    )

    parser.add_argument(
        '--search_hashtags'
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

    parser.add_argument(
        '--log_handler_level',
        help='log handler setting. "both" for file and stream, "file" for file, "stream" for stream',
        default='both',
        choices = ['both','file','stream']
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
        logging_file  = os.path.join(args.log_dir, f'{today_datetime}_activity_counts_stats.log')
        if args.log_handler_level == 'both':
            handlers = [
                logging.FileHandler(filename=logging_file,mode='w'),
                logging.StreamHandler()
            ]
        elif args.log_handler_level == 'file':
            handlers = [logging.FileHandler(filename=logging_file,mode='w')]
        elif args.log_handler_level == 'stream':
            handlers = [logging.StreamHandler()]
        logging.basicConfig(
            handlers=handlers,
            format=logging_fmt,
            level=logging_level,
            datefmt='%m/%d/%Y %I:%M:%S %p'
        )

        logging.info(f'Start time of script is {today_datetime}')

    main(args)