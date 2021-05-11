import subprocess
import os
import argparse
import pandas as pd
import logging
import tqdm
import datetime

def main():

    users = pd.read_csv(
        args.user_list,
        header=None,
        names=['user_id','tweet_count']
    )

    logging.info('{} users to be collected.'.format(sum(users['tweet_count']>=args.min_tweets)))
    print('{} users to be collected.'.format(sum(users['tweet_count']>=args.min_tweets)))
    logging.info('{} users to be dropped.'.format(sum(users['tweet_count']<args.min_tweets)))
    print('{} users to be dropped.'.format(sum(users['tweet_count']<args.min_tweets)))    
    users = users[users['tweet_count']>=args.min_tweets]

    for user_row in users.iterrows():

        # user_id
        user_id = user_row[1]['user_id']

        # generate save filename
        save_filename = os.path.join(args.output_dir,'data/timeline_' + user_id + '.jsonl')

        # check if file already exists
        if os.path.isfile(save_filename):
            logging.info('{} user already pulled. Continuing...'.format(user_id))
            continue

        # generate query
        query = 'from:' + user_id

        # generate start and end dates
        end_time = datetime.datetime.fromisoformat(args.end_time)
        start_time = end_time - datetime.timedelta(args.back)
        end_time = datetime.datetime.isoformat(end_time)
        start_time = datetime.datetime.isoformat(start_time)

        subprocess.run(
            ['twarc2',
            'search',
            '--archive',
            '--max-results',
            args.max_tweets,
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
        'user_list',
        help='list of users in a txt file. One user per line.'
    )

    parser.add_argument(
        '--output_dir',
        help='directory to place outputs. defaults to current working directory',
        default = os.getcwd()
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
        '--min_tweets',
        help='The minimum frequency for user history to be collected.',
        type=int,
        default=50
    )

    parser.add_argument(
        '--max_tweets',
        help='The maximum number of tweets to collect for each user.',
        type=int,
        default=2000
    )

    # parse arguments
    args = parser.parse_args()

    # set up logging
    logging.basicConfig(filename=os.path.join(args.output_dir, 'user_timelines.log'),
                        encoding='utf-8',
                        format='%(levelname)s:%(message)s',
                        level=logging.DEBUG)

    # argument checks
    try:
        datetime.datetime.fromisoformat(args.end_time)
    except:
        logging.exception('invalid end_time string')
        raise ValueError('invalid end_time string')

    assert os.path.isdir(args.output_dir)

    main()