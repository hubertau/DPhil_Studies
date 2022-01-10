#!/usr/bin/python3.9

'''
Script to collect followers and following with a supplied list of user ids.

This script uses twarc1 and Twitter API 1.1, because its limit on followers is higher.

'''

import argparse
import datetime
import glob
import os
import re
import subprocess
import logging
import json
from concurrent.futures import ThreadPoolExecutor
from typing import NamedTuple
from itertools import repeat

class save_names(NamedTuple):
    user_id: str
    followers_file: str
    following_file: str

def time_function(func):
    """
    Wrapper function to time execution.
    """

    def inner(*args, **kwargs):
        timefunc_start_time = datetime.datetime.now()
        logging.info('Start Time: {}'.format(timefunc_start_time))
        result = func(*args, **kwargs)
        logging.info('Total Time Taken: {}'.format(datetime.datetime.now()-timefunc_start_time))
        return result
    return inner

def check_existing_downloaded_timelines(args):

    all_timeline_files = glob.glob(os.path.join(args.data_dir, 'timeline*.jsonl'))

    all_timeline_file_ids = [re.split('[_.]', i) for i in all_timeline_files]
    all_timeline_file_ids = [i[-2] for i in all_timeline_file_ids]

    with open(args.user_list_file, 'r') as f:
        p = f.readlines()
        p = [i.replace('\n','') for i in p]
        user_list = p

    # Use of hybrid method for intersection check
    temp = set(all_timeline_file_ids)
    intersection = [value for value in user_list if value in temp]

    return intersection

def generate_filename(args, user_id, followers = True):

    if followers:
        return os.path.join(
            os.path.abspath(args.output_dir),
            f'followers_{user_id}.txt'
        )
    else:
        return os.path.join(
            os.path.abspath(args.output_dir),
            f'following_{user_id}.txt'
        )

def twarc_follow(args, save_names, credentials, worker_num, followers = True):

    if followers:
        # save_filename = generate_filename(args, user_id, followers)
        if os.path.isfile(save_names.followers_file):
            logging.warning(f'{os.path.split(save_names.followers_file)[-1]} already exists. Continuing...')
            return None
        # screen_name = get_screen_name_from_file(os.path.join(args.data_dir, f'timeline_{user_id}.jsonl'))
        # this is to collect the followers of a user
        with open(save_names.followers_file, 'w') as f:
            subprocess.run([
                'twarc',
                'followers',
                str(save_names.user_id),
                '--consumer_key',
                credentials['API_key'],
                '--consumer_secret',
                credentials['API_secret'],
                '--access_token',
                credentials['Access_Token'],
                '--access_token_secret',
                credentials['Access_Token_Secret']
            ], stdout=f)
        logging.info(f'Worker {worker_num}: saving at {os.path.split(save_names.followers_file)[-1]}')

    else:
        # save_filename = generate_filename(args, user_id, followers)
        # this is to collect the users that the one in question is following
        if os.path.isfile(save_names.following_file):
            logging.warning(f'{os.path.split(save_names.following_file)[-1]} already exists. Continuing...')
            return None
        with open(save_names.following_file, 'w') as f:
            subprocess.run([
                'twarc',
                'friends',
                str(save_names.user_id),
                '--consumer_key',
                credentials['API_key'],
                '--consumer_secret',
                credentials['API_secret'],
                '--access_token',
                credentials['Access_Token'],
                '--access_token_secret',
                credentials['Access_Token_Secret']
            ], stdout=f)
        logging.info(f'Worker {worker_num}: saving at {os.path.split(save_names.following_file)[-1]}')

def process_one_list(args, save_names_list, worker_num, twarc_credentials):

    for index, item in enumerate(save_names_list):
        logging.info(f'Worker {worker_num}: Collecting user {index+1} of {len(save_names_list)}: {100*(index+1)/len(save_names_list):.2f}%')
        twarc_follow(args, item, twarc_credentials, worker_num, followers=True)
        twarc_follow(args, item, twarc_credentials, worker_num, followers=False)

    logging.info(f'Worker {worker_num} COMPLETE, {len(save_names_list)} collected.')

    return None

def chunker_list(seq, size):
    return (seq[i::size] for i in range(size))

@time_function
def main(args):
    intersection = check_existing_downloaded_timelines(args)

    initial_to_collect = [
        save_names(
            user_id = i,
            followers_file=generate_filename(args, i, followers=True),
            following_file=generate_filename(args, i, followers=False)
        ) for i in intersection
    ]

    to_collect = []
    for i in initial_to_collect:
        follower_check=os.path.isfile(i.followers_file)
        following_check=os.path.isfile(i.following_file)
        if follower_check + following_check == 1:
            logging.warning(f'Only one of following/follower pair files exists for {i.user_id} exists.')
            logging.warning(f'Follower file for user {i.user_id}: {os.path.isfile(i.followers_file)}')
            logging.warning(f'Following file for user {i.user_id}: {os.path.isfile(i.following_file)}')
            logging.warning('Continuing with overwriting the existing file.')
            to_collect.append(i)
        elif not follower_check and not following_check:
            to_collect.append(i)
        else:
            pass

    to_collect = [i for i in to_collect if (not os.path.isfile(i.followers_file)) and (not os.path.isfile(i.following_file))]

    logging.info(f'To collect: {len(to_collect)} users.')

    to_collect = list(chunker_list(to_collect, 4))

    logging.debug(to_collect)

    # import twarc_credentials
    with open(args.twarc_credentials, 'r') as f:
        twarc_credentials = json.load(f)

    with ThreadPoolExecutor(max_workers=4) as executor:
        executor.map(process_one_list, repeat(args), to_collect, [0,1,2,3], list(twarc_credentials.values()))

    # for user_num, user_id in enumerate(intersection):
    #     logging.info(f'Collecting user {user_num+1} of {total}: {100*(user_num+1)/total:.2f}%')
    #     twarc_follow(args, user_id, followers = True)
    #     twarc_follow(args, user_id, followers = False)

    return None

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Script to collect the followers and following from an existing list of users.')

    parser.add_argument(
        'user_list_file',
        help='the user list file from which users are to be collected. it will be referenced against existing downloaded timelines so as not to over collect followers.'
    )

    parser.add_argument(
        '--data_dir',
        help='Where the raw data files are stored (to check which timelines are downloaded.',
        default = '../../data/01_raw/'
    )

    parser.add_argument(
        '--output_dir',
        help='Where to place output files. Defaults to data_dir.'
    )

    parser.add_argument(
        '--twarc_credentials',
        help='twarc credentials json'
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
        logging_file  = os.path.join(args.log_dir, f'{today_datetime}_follower_collection.log')
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

        logging.info(f'Start time of follower collection script is {today_datetime}')

    if not args.output_dir:
        args.output_dir = args.data_dir

    logging.info(f'User file is {args.user_list_file}')
    logging.info(f'Data dir is {args.data_dir}')
    logging.info(f'Output dir is {args.output_dir}')

    _ = main(args)