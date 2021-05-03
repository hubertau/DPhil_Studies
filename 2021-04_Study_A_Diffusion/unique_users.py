'''
This script is to define a function that takes a results file and obtain the unique user ids from it.
'''

import pandas as pd
import argparse
from collections import Counter

parser = argparse.ArgumentParser(description='obtain unique users of results tweets collected')

parser.add_argument(
    'file',
    help='results file from full archive search'
)

# parse 
args = parser.parse_args()

def get_unique_users(input_tweets_csv):

    # read in file
    DATA = pd.read_csv(input_tweets_csv)

    # convert ISO 8601 time from Twitter to DateTime objects
    DATA['created_at'] = pd.to_datetime(DATA['created_at'])

    users_and_counts = Counter(DATA['author_id'])

    return users_and_counts

if __name__ == '__main__':

    print(get_unique_users(args.file).most_common(3))
