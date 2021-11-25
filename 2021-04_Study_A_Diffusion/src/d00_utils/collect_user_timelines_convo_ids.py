'''
Script to collect conversation ids from user timelines to be fleshed out

'''

import subprocess
import os
import argparse
import pandas as pd
import logging
import tqdm
import datetime
import glob
import jsonlines
from collections import defaultdict

def main():

    # read in data
    file_list = glob.glob(args.data_dir + '/timeline*.jsonl')

    # collect conversation ids
    print('collecting conversation ids from user timelines scraped:')
    conversation_ids = defaultdict(list)
    for file in tqdm.tqdm(file_list):
        with jsonlines.open(file) as reader:
            for tweet_jsonl in reader:
                tweet_list_in_file = tweet_jsonl['data']
                for tweet_obj in tweet_list_in_file:
                    if 'conversation_id' in tweet_obj:
                        conversation_ids[tweet_obj['conversation_id']].append(tweet_obj['author_id'])


    logging.info('{} conversation ids to be collected.'.format(sum(users['tweet_count']>=int(args.min_tweets))))
    print('{} users to be collected.'.format(sum(users['tweet_count']>=int(args.min_tweets))))

    for user_row in tqdm.tqdm(users.iterrows(), total=len(users)):

        # user_id
        user_id = str(user_row[1]['user_id'])

        # generate save filename
        save_filename = os.path.join(args.output_dir,'data/timeline_' + user_id + '.jsonl')

        # check if file already exists
        if os.path.isfile(save_filename):
            logging.info('{} user already pulled. Continuing...'.format(user_id))
            continue

        # generate query
        query = 'conversation_id:' + c_id

        # generate start and end dates
        end_time = datetime.datetime.fromisoformat(args.end_time)
        start_time = end_time - datetime.timedelta(args.back)
        end_time = datetime.datetime.isoformat(end_time)
        start_time = datetime.datetime.isoformat(start_time)

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

        logging.info(user_id)

if __name__ == '__main__':

    # set up parsing arguments
    parser = argparse.ArgumentParser(description='timeline collection')

    parser.add_argument(
        'data_dir',
        help='directory containing data'
    )

    parser.add_argument(
        '--limit',
        help='The maximum number of tweets to collect for each user.',
        default=2000
    )

    # parse arguments
    args = parser.parse_args()

    # set up logging
    logging.basicConfig(filename=os.path.join(os.path.split(args.data_dir)[0], 'conversation_ids.log'),
                        encoding='utf-8',
                        format='%(levelname)s:%(message)s',
                        level=logging.DEBUG)

    # argument checks
    assert os.path.isdir(args.data_dir)

    main()