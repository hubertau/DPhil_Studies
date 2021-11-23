'''
2021-09-14: Script to collect user timelines from a list of users supplied.

A list of users and their respective existing frequencies is to be supplied in a text file. The purpose of this is an estimate of their actual tweeting frequencies.

'''

import subprocess
import os
import argparse
import pandas as pd
import logging
import tqdm
import datetime
import csv


def main(args):

    if args.min_tweets:
        # read in data. N.B. the data should come in the form of a txt file with each row as a user_id, count where count is the number of tweets in the initial FAS collection.
        users = pd.read_csv(
            args.user_list,
            header=None,
            names=['user_id','tweet_count']
        )

        # if continue_from_user_id has been provided, then print this and filter the data
        users = users.iloc[users['user_id'][users['user_id']==args.continue_from_user_id].index[0]:,:]
        logging.info('\nContinuing from user {}. Collecting this user now.'.format(args.continue_from_user_id))

        # convert tweet_count column to numeric
        users['tweet_count'] = pd.to_numeric(users['tweet_count'])

        logging.info('{} users to be collected.'.format(sum(users['tweet_count']>=int(args.min_tweets))))
        logging.info('{} users to be dropped.'.format(sum(users['tweet_count']<int(args.min_tweets))))
        users = users[users['tweet_count']>=int(args.min_tweets)]
    else:
        users = pd.read_csv(
            args.user_list,
            header=None,
            names=['user_id']
        )

        logging.info('No min tweets parameter supplied, assuming txt file is only list of user ids')

    # now iterate through the users
    for user_row in tqdm.tqdm(users.iterrows(), total=len(users)):

        # user_id
        user_id = str(user_row[1]['user_id'])

        # generate save filename
        save_filename = os.path.join(args.output_dir,'timeline_' + user_id + '.jsonl')

        # check if file already exists
        if os.path.isfile(save_filename):
            logging.info('{} user already pulled. Continuing...'.format(user_id))
            continue

        # generate query
        query = 'from:' + user_id

        # omit retweets
        if args.omit_retweets:
            query = query + ' -is:retweet'

            # N.B. Prepend a dash (-) to a keyword (or any operator) to negate it (NOT). For example, cat #meme -grumpy will match Tweets containing the hashtag #meme and the term cat, but only if they do not contain the term grumpy. One common query clause is -is:retweet, which will not match on Retweets, thus matching only on original Tweets, Quote Tweets, and replies. All operators can be negated, but negated operators cannot be used alone.

        if args.omit_hashtags:
            with open('/home/hubert/DPhil_Studies/2021-04_Study_A_Diffusion/search_hashtags.txt', newline='') as f:
                terms = list(csv.reader(f))
            terms = ['-'+i[0] for i in terms]
            terms  = ' '.join(terms)
            query = query + ' ' + terms

        assert len(query) <= 1024, 'Query length too long'

        logging.debug(f'Query is: {query}')

        # generate start and end dates
        end_time = datetime.datetime.fromisoformat(args.end_time)
        start_time = end_time - datetime.timedelta(args.back)
        end_time = datetime.datetime.isoformat(end_time)
        start_time = datetime.datetime.isoformat(start_time)

        if args.start_time:
            start_time = datetime.datetime.fromisoformat(args.start_time)
            start_time = datetime.datetime.isoformat(start_time)


        print('start time: {}'.format(start_time))
        print('end_time: {}'.format(end_time))

        subprocess.run(
            ['twarc2',
            'search',
            '--archive',
            '--limit',
            args.limit,
            '--max-results',
            '100',
            '--end-time',
            end_time,
            '--start-time',
            start_time,
            query,
            save_filename]
        )

        logging.info(f'{user_id} collected. Saved at {save_filename}.')

if __name__ == '__main__':

    # set up parsing arguments
    parser = argparse.ArgumentParser(description='timeline collection')

    parser.add_argument(
        'user_list',
        help='list of users in a txt file. One user per line.'
    )

    parser.add_argument(
        '--output_dir',
        help='where to place the data files.'
    )

    parser.add_argument(
        '--back',
        help='The maximum period back for each user to extract. Give integer in days. Default = 14',
        type = int,
        default = 14
    )

    parser.add_argument(
        '--end_time',
        help='end_time argument in format YYYY-MM-DDTHH:mm:ss (ISO format)',
        default=datetime.datetime.isoformat(datetime.datetime.now().replace(microsecond=0))
    )

    parser.add_argument(
        '--start_time',
        help='end_time argument in format YYYY-MM-DDTHH:mm:ss (ISO format). If present, --back option is ignored.'
    )

    parser.add_argument(
        '--min_tweets',
        help='The minimum frequency for user history to be collected.',
        default=50
    )

    parser.add_argument(
        '--limit',
        help='The maximum number of tweets to collect for each user.',
        default=2000
    )

    parser.add_argument(
        '--omit_hashtags',
        help='Include in query to not include tweets that contain the original search hashtags. These are already collected in the FAS.',
        default = False,
        action = "store_true"
    )

    parser.add_argument(
        '--omit_retweets',
        help = 'omit retweets in the query',
        default = False,
        action = 'store_true'
    )

    parser.add_argument(
        '--continue_from_user_id',
        help = 'continue from user_id in list',
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
    if args.omit_hashtags:
        print("N.B. Original hashtag search queries have been omitted from collected tweets.")

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

    # argument checks
    try:
        datetime.datetime.fromisoformat(args.end_time)
    except:
        logging.exception('invalid end_time string')
        raise ValueError('invalid end_time string')

    assert os.path.isdir(args.output_dir)
    assert os.path.isdir(args.log_dir)

    main(args)