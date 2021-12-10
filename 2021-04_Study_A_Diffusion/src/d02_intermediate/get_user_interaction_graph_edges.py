#!/usr/bin/python3.9

'''
Script to obtain interaction edges from user timelines

'''

import argparse
import datetime
import glob
import logging
import os
import re
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat

import attr
import h5py
import jsonlines
import pandas as pd
# from mpi4py import MPI


@attr.s(frozen=True, slots=True)
class Tweet_Interaction:
    tweet_id: str = attr.ib(validator=attr.validators.instance_of(str))
    author_id: str = attr.ib(validator=attr.validators.instance_of(str))
    tweet_lang: str = attr.ib(validator=attr.validators.instance_of(str))
    likes: int = attr.ib(validator=attr.validators.instance_of(int))
    created_at: datetime.datetime = attr.ib(converter=lambda x: datetime.datetime.fromisoformat(x[:-1]).date())
    in_reply_to: list[str] = attr.ib(factory=list, order=False, hash=False, repr=True)
    mentions: list[str] = attr.ib(factory=list, order=False, hash=False, repr=True)
    quotes: list[str] = attr.ib(factory=list, order=False, hash=False, repr=True)
    replies: list[str] = attr.ib(factory=list, order=False, hash=False, repr=True)
    contains_hashtags: list[str] = attr.ib(factory=list, order=False, hash=False, repr=True)


def process_one_tweet_object(tweet, user_list):

    # check mentions. These are the users that the person @-ed in their tweet text. mentions will have length 0 if either none are mentioned or if none mentioned are in the desired user list. We make this check because this paper is only interested in the interactions between users.
    try:
        mentions = [i['id'] for i in tweet['entities']['mentions'] if i['id'] in user_list]
    except:
        mentions = []

    # referenced tweets are those quoted or replied to.
    contains_referenced_tweets = 'referenced_tweets' in tweet
    if contains_referenced_tweets:

        referenced_tweets = tweet['referenced_tweets']

        # extract quote tweets. do not check for user_list because these are tweet ids not user ids
        quotes = [i['id'] for i in referenced_tweets if i['type'] == 'quoted']

        # extract replies. do not check for user_list because these are tweet ids not user ids
        replies = [i['id'] for i in referenced_tweets if i['type'] == 'replied_to']

        current_tweet_id = tweet['id']
        try:
            in_reply_to = tweet['in_reply_to_user_id']
            if isinstance(in_reply_to,str):
                in_reply_to=[in_reply_to]
            elif isinstance(in_reply_to, list):
                pass
        except KeyError:
            in_reply_to = []
        logging.debug(in_reply_to)

        # if no saveable mentions, quotes, or replies are found then 
        if len(quotes) + len(replies) == 0 and len(mentions) == 0 and (str(in_reply_to) not in user_list):
            return None

        if 'entities' in tweet and 'hashtags' in tweet['entities']:
            contains_hashtags = [i['tag'] for i in tweet['entities']['hashtags']]
        else:
            contains_hashtags = []

        # get tweet language
        tweet_lang = tweet.get('lang')

        public_metrics = tweet.get('public_metrics')
        if public_metrics:
            likes = int(public_metrics.get('like_count'))
        else:
            likes = 0
            logging.warning(
                f'public metrics not found in tweet id {current_tweet_id}'
            )

        tweet_extract = Tweet_Interaction(
            tweet_id=current_tweet_id,
            author_id=tweet['author_id'],
            tweet_lang = tweet_lang,
            likes = likes,
            mentions=mentions,
            quotes=quotes,
            replies=replies,
            created_at=tweet['created_at'],
            in_reply_to=in_reply_to,
            contains_hashtags=contains_hashtags
        )
        return tweet_extract

    else:
        return None

def process_one_file_pair(input_tuple):

    twitter_user_timeline, augmented_file, user_list = input_tuple

    logging.info(f'Processing {twitter_user_timeline} and {augmented_file}')

    results = []
    with jsonlines.open(twitter_user_timeline) as reader:
        for tweet_jsonl in reader:
            # iterate over the tweets
            for tweet in tweet_jsonl['data']:
                results.append(process_one_tweet_object(tweet, user_list))

    # incorporate augmented data too.
    with jsonlines.open(augmented_file) as reader:
        for tweet in reader:
            results.append(process_one_tweet_object(tweet, user_list))

    logging.info(f'End processing {twitter_user_timeline} and {augmented_file}')

    return results

def main(args):

    # collect data files
    timeline_file_list = sorted(glob.glob(os.path.join(args.data_dir, 'timeline*.jsonl')))
    augmented_file_list = sorted(glob.glob(os.path.join(args.data_dir, 'augmented*.jsonl')))

    # sanity checks
    try:
        assert len(timeline_file_list) == len(augmented_file_list)
        assert all([
            re.split('[_.]', timeline_file_list[i])[-2] == re.split('[_.]',augmented_file_list[i])[-2] for i in range(len(timeline_file_list))
        ])
    except AssertionError:
        logging.error(f'Assertion error on timeline and augmented jsonl files', exc_info=True)
        raise

    # get userset
    with open(args.user_list_file, 'r') as f:
        p = f.readlines()
        p = [str(i.replace('\n','')) for i in p]
        user_list = set(p)

    # users_in_sample = [re.split('[_.]', i)[-2] for i in timeline_file_list]

    # assert set(users_in_sample).issubset(user_list)

    assert os.path.isdir(args.output_dir)
    output_filename = os.path.join(args.output_dir, 'interactions.hdf5')
    if args.output_dataset_name:
        output_key = args.output_dataset_name
    else:
        output_key = str(int(re.split('_',args.user_list_file)[-2]) + 1)
        logging.info(f'group number {output_key}')
        output_key = f'interactions_group_{output_key}'

    if args.max_workers is None:
        args.max_workers = os.cpu_count()

    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        logging.info(f'Beginning ProcessPool Executor with {args.max_workers} workers.')
        results = executor.map(process_one_file_pair, zip(timeline_file_list, augmented_file_list, repeat(user_list)))

    # flatten list of lists
    results = [item for sublist in results for item in sublist if item is not None]

    results_pd = pd.DataFrame([attr.asdict(x) for x in results])

    with h5py.File(output_filename, 'a') as f:
        if output_key in f and args.overwrite:
            del f[output_key]
            logging.info(f'Overwrite flag set and {output_key} deleted.')

    results_pd.to_hdf(
        output_filename,
        output_key,
        mode='a'
    )
    logging.info(f'File saved to {output_filename}')

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='extract interaction edges')

    parser.add_argument(
        '--data_dir',
        help='where the timeline files are, in .jsonl format',
        default='../../data/01_raw/'
    )

    parser.add_argument(
        '--user_list_file',
        help='user list file to compare to.',
        default='../../data/02_intermediate/user_list_full_0_nocount.txt'
    )

    parser.add_argument(
        '--output_dir',
        help='desired output directory',
        default='../../data/02_intermediate/'
    )

    parser.add_argument(
        '--output_dataset_name',
        help='custom output dataset name'
    )

    parser.add_argument(
        '--overwrite',
        help='overwrite interactions for a certain group',
        action='store_true'
    )

    parser.add_argument(
        '--max_workers',
        help='Max workers for ProcessPoolExecutor',
        default=None,
        type=int
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
        logging_file  = os.path.join(args.log_dir, f'{today_datetime}_interaction_edges.log')
        logging.basicConfig(
            handlers=[
                logging.FileHandler(filename=logging_file,mode='w'),
                logging.StreamHandler()
            ],
            format=logging_fmt,
            level=logging_level,
            datefmt='%m/%d/%Y %I:%M:%S %p'
        )

        logging.info(f'Start time of script is {today_datetime}')

    try:
        main(args)
    except KeyboardInterrupt:
        logging.info('Keyboard Interrupt from user')
