#!/usr/bin/python3.9

'''
For sampling of users to do second data collection.
'''

import argparse
import datetime
import glob
import math
import os
import pickle
import pprint

import attr
import h5py
import jsonlines
import pandas as pd
import tqdm

import user_timeline_augmentation as augmentation


# for reference, see https://www.youtube.com/watch?v=vBH6GRJ1REM
@attr.s(frozen=True, slots=True)
class Tweet_Entry:
    author_id: str = attr.ib(validator=attr.validators.instance_of(str))
    created_at: datetime.datetime = attr.ib(converter=lambda x: datetime.datetime.fromisoformat(x[:-1]).date())
    all_hashtags: list[str] = attr.ib(factory=list, order=False, hash=False, repr=True)
    search_hashtags: list[str] = attr.ib(factory=list, order=False, hash=False, repr=True)

def activity_df(search_hashtags, file_list, user_list, verbose = False):

    if verbose:
        print('\nObtaining empirical activity distribution from FAS files.\n')

    # create output
    results = []

    # Also need to do stratified sampling with each hashtag. How to deal with overlap?

    if verbose:
        print('First processing the FAS files to collect the relevant information from the raw data files:\n')

    for user_jsonl_file in tqdm.tqdm(file_list, desc='processing FAS jsonl files'):
        with jsonlines.open(user_jsonl_file) as reader:
            for tweet_jsonl in reader:
                tweet_list_in_file = tweet_jsonl['data']
                for tweet_data in tweet_list_in_file:

                    if tweet_data['author_id'] not in user_list:
                        continue

                    if 'entities' in tweet_data:
                        if 'hashtags' in tweet_data['entities']:

                            # remember that the Twitter objects dont' have '#' in front of the hashtags.
                            hts = ['#' + i['tag'].lower() for i in tweet_data['entities']['hashtags']]

                            if any(item in search_hashtags for item in hts):

                                hts_overlap = set(hts).intersection(set(search_hashtags))

                                results.append(Tweet_Entry(
                                    tweet_data['author_id'],
                                    tweet_data['created_at'],
                                    hts,
                                    hts_overlap,
                                ))

    if verbose: print('converting to df')
    FAS_activity_df = pd.DataFrame([attr.asdict(x) for x in results])
    if verbose: print('Done.')

    if verbose: print('extracting search hashtag counts...')
    for hashtag in search_hashtags:
        FAS_activity_df['vocab:'+hashtag] = FAS_activity_df['search_hashtags'].apply(lambda x: any(hashtag.lower() == item for item in x))
    if verbose: print('Done.')

    return FAS_activity_df

def sample_df(df, max_total = None, weights = False):

    '''
    sample an activity df.
    '''

    columns = list(df.columns)
    columns = [i for i in columns if 'vocab:' in i]

    number_unique_users = len(df['author_id'].unique())
    max_users_per_hashtag = math.ceil(number_unique_users/len(columns))

    if max_total:
        max_users_per_hashtag = math.ceil(max_total/len(columns))
    else:
        max_total=max_users_per_hashtag

    sampled_users_to_return = []
    ids_sampled_already = []
    total_users_collected = []
    while sum(total_users_collected) < max_total:
        for hashtag in columns:

            # set weights
            if weights:
                weights=hashtag
            else:
                weights=None

            temp_df = df[df[hashtag] > 0]

            # remove already sampled users from other hashtags
            temp_df = temp_df[~temp_df['author_id'].isin(ids_sampled_already)]

            temp_df = temp_df.groupby('author_id').sum()
            # print(len(temp_df))
            sample = temp_df.sample(
                n=min(max_users_per_hashtag, len(temp_df)),
                random_state=1,
                weights=weights
            )
            sampled_users_to_return.append((hashtag,sample))
            ids_sampled_already = list(set(ids_sampled_already + list(sample.index)))

        total_users_collected = [len(i[1]) for i in sampled_users_to_return]
        print('currently sampled {} users'.format(sum(total_users_collected)))

    users_collected = [i[1].index for i in sampled_users_to_return]
    users_collected = [val for sublist in users_collected for val in sublist]

    assert len(users_collected) == len(list(set(users_collected)))

    return users_collected, sampled_users_to_return

def main(args):

    # get daterange from group number
    # with open(args.group_daterange_file, 'rb') as f:
    #     group_daterange = pickle.load(f)
    with h5py.File(args.group_daterange_file, 'r') as f:
        group_daterange = f['segments']['selected_ranges']
        group_daterange = group_daterange[()]
        group_daterange = group_daterange.astype('U13')

    # get appropriate file list
    FAS_filelist = augmentation.sort_FAS_by_daterange(glob.glob(os.path.join(args.data_dir, 'FAS*.jsonl')))

    # convert to dates
    x = group_daterange[args.group-1]
    x = (datetime.datetime.strptime(x[0],'%Y-%m-%d'), datetime.datetime.strptime(x[1],'%Y-%m-%d'))
    FAS_filelist = augmentation.filter_FAS(x,FAS_filelist)
    FAS_filelist = [i[0] for i in FAS_filelist]

    if args.verbose:
        pp = pprint.PrettyPrinter(indent=4)
        print('\n For group {} on daterange {}'.format(args.group, group_daterange[args.group-1]))
        pp.pprint(FAS_filelist)

    # read in users to care about
    with open(args.group_userlist, 'r') as f:
        p = f.readlines()
        p = [i.replace('\n','') for i in p]
        user_list = p

    # read in hashtags to plot:
    with open(args.search_hashtags, 'r') as f:
        search_hashtags = f.readlines()
        search_hashtags = [i.rstrip('\n') for i in search_hashtags]
        search_hashtags = [i.lower() for i in search_hashtags]

    save_filename = 'group_' + str(args.group) + '_sampling_df.obj'
    save_filename = os.path.join(args.output_dir, save_filename)

    if os.path.isfile(save_filename):
        with open(save_filename, 'rb') as f:
            df = pickle.load(f)
        print('loaded file in')
    else:
        # collect activity df for each user in the windows selected.
        df = activity_df(
                search_hashtags,
                FAS_filelist,
                user_list,
                verbose = True
            )
        with open(save_filename, 'wb') as f:
            pickle.dump(df, f)

    if args.verbose: print('saved to {}'.format(save_filename))

    sampled_df = sample_df(df, max_total = args.max_total, weights=args.weighted)

    sampled_save_filename = 'group_' + str(args.group) + '_sampled_weight_' + str(args.weighted) + '_users.obj'
    sampled_save_filename = os.path.join(args.output_dir, sampled_save_filename)

    with open(sampled_save_filename, 'wb') as f:
        pickle.dump(sampled_df, f)

    if args.verbose: print('saved to {}'.format(sampled_save_filename))

    user_list_save_filename = 'group_' + str(args.group) + '_sampled_weight_' + str(args.weighted) + '_users.txt'
    user_list_save_filename = os.path.join(args.output_dir, user_list_save_filename)

    with open(user_list_save_filename, 'w') as f:
        for j in sampled_df[0]:
            f.write(j)
            f.write('\n')

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Sample users based on the empirical distribution of user activity from full archive search (FAS) jsonl files.')

    parser.add_argument(
        'data_dir',
        help='data directory'
    )

    parser.add_argument(
        'output_dir',
        help='output directory'
    )

    parser.add_argument(
        'search_hashtags',
        help='txt file containing search hashtags, one per line.',
        default='../../references/search_hashtags.txt'
    )

    parser.add_argument(
        'group',
        help='Group of users, 1-3, that this applies to.',
        type=int
    )

    parser.add_argument(
        'group_userlist',
        help='txt file of user ids, no counts'
    )

    parser.add_argument(
        '--max_total',
        help='max number of users in total to sample',
        default=5000,
        type=int
    )

    parser.add_argument(
        '--group_daterange_file',
        help='group selected dates object',
        default='../../data/02_intermediate/FAS_selected_date_ranges.obj'
    )

    parser.add_argument(
        '--weighted',
        help='whether to sample weighted by activity',
        default=False,
        action='store_true'
    )

    parser.add_argument(
        '--verbose',
        help='verbosity parameter',
        default=False,
        action='store_true'
    )

    args = parser.parse_args()

    main(args)
