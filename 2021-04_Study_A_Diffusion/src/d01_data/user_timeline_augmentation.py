#!/usr/bin/python3.9

'''
2021-11-05
This script should be run after timeline data collection. The search parameter of timeline collection is not to include the hashtags. The structure of this solution:

* get tweet ids from FAS files associated with each user 
'''

import argparse
import datetime
import glob
from collections import defaultdict
import re
import os
import tqdm

import jsonlines

def timeline_file_span(timeline_file):

    '''
    Obtain the date span of a sinlge timeline file.
    '''

    assert os.path.isfile(timeline_file), 'Check file paths.'

    first = True
    with jsonlines.open(timeline_file) as reader:
        for tweet_jsonl in reader:
            for tweet_data in tweet_jsonl['data']:
                if first:
                    current_min_date = datetime.datetime.fromisoformat(tweet_data['created_at'][:-1])
                    current_max_date = datetime.datetime.fromisoformat(tweet_data['created_at'][:-1])
                    first = False
                tweet_created_at = datetime.datetime.fromisoformat(tweet_data['created_at'][:-1])
                if tweet_created_at > current_max_date:
                    current_max_date = tweet_created_at
                if tweet_created_at < current_min_date:
                    current_min_date = tweet_created_at

    return (current_min_date, current_max_date)

def datetime_from_string(input_string):
    return datetime.datetime.strptime(input_string, "%Y-%m-%d")

def FAS_range_from_filename(FAS_filename):

    '''
    Obtain date range of FAS files from a file name.
    '''
    FAS_split = re.split('[_.]',FAS_filename)
    FAS_min = FAS_split[-3]
    FAS_max = FAS_split[-2]

    # convert to datetime objects
    FAS_min = datetime_from_string(FAS_min)
    FAS_max = datetime_from_string(FAS_max)

    return (FAS_min, FAS_max)

def sort_FAS_by_daterange(FAS_filelist):

    # get FAS files dates
    FAS_dates = [(i, FAS_range_from_filename(i)[0], FAS_range_from_filename(i)[1]) for i in FAS_filelist]
    sorted_FAS_list_with_dates = sorted(FAS_dates, key = lambda x: x[2])

    return sorted_FAS_list_with_dates

def filter_FAS(date_range, sorted_FAS_filelist):

    '''
    From the date range collected in a tuple (min, max), find the list of files required to scan in FAS.

    N.B. the dates of the FAS files are not inclusive, whereas because both min an max dates of date range are from tweet objects they are inclusive.

    This function returns
    '''

    required_min = len(sorted_FAS_filelist)
    required_max = 0

    for i, e in enumerate(sorted_FAS_filelist):
        _, FAS_date_min, FAS_date_max = e
        if date_range[0] > FAS_date_max:
            continue
        if date_range[1] < FAS_date_min:
            continue
        if date_range[0] < FAS_date_max:
            required_min = min(required_min, i)
        if date_range[1] > FAS_date_min:
            required_max = max(required_max, i)
    return (required_min, required_max)

def main(args):

    # get filelist in strings
    FAS_filelist = glob.glob(os.path.join(args.data_dir, 'FAS*.jsonl'))
    timeline_filelist = glob.glob(os.path.join(args.data_dir, 'timeline*.jsonl'))

    # sort and extract date ranges
    sorted_FAS_list_with_dates = sort_FAS_by_daterange(FAS_filelist)
    if args.verbose:
        print(sorted_FAS_list_with_dates[:10])

    # get user_ids
    user_ids = [re.split('[_.]',timeline)[-2] for timeline in timeline_filelist]
    # if args.verbose:
    #     print(user_ids)

    # get user_timespans
    # timeline_date_range = [timeline_file_span(timeline) for timeline in timeline_filelist]

    user_scan_ranges = defaultdict(set)

    # for timeline in tqdm.tqdm(timeline_filelist, desc='User Files'):
        # 

        # get user date range
        

        # # determine the FAS files to scan
        # FAS_files_to_scan = filter_FAS(timeline_date_range, sorted_FAS_list_with_dates)

    user_tweets_to_append = defaultdict(list)
    for i in tqdm.tqdm(sorted_FAS_list_with_dates, desc='FAS scan', leave=False):
        with jsonlines.open(i[0]) as reader:
            for tweet_json in reader:
                for tweet in tweet_json['data']:
                    if tweet['author_id'] in user_ids:
                        user_tweets_to_append[tweet['author_id']].append(tweet['id'])
                        user_scan_ranges[tweet['author_id']].add(i)

    for id in tqdm.tqdm(user_ids, desc='writing to files'):
        # write out results
        save_filename = 'augmented_timeline_ids_' + id + '.txt'
        save_filename = os.path.join(args.data_dir, save_filename)

        # scan_ranges = sort_FAS_by_daterange(list(user_scan_ranges[id]))

        with open(save_filename, 'w') as f:
            # f.write(scan_ranges[0]+ '\n')
            # f.write(scan_ranges[1] + '\n')
            for j in user_tweets_to_append[id]:
                f.write(j)
                f.write('\n')

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='')

    parser.add_argument(
        'data_dir',
        help='data directory'
    )

    parser.add_argument(
        '--verbose',
        help='verbosity',
        default=False,
        action='store_true'
    )

    args = parser.parse_args()

    main(args)
