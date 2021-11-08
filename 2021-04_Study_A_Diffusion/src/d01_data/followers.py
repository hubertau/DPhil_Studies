#!/usr/bin/python3.9

'''
Script to collect followers and following with a supplied list of user ids.
'''

import argparse
import datetime
import glob
import os
import re
import subprocess
import tqdm

def time_function(func):

        """
        Wrapper function to time execution.
        """

        def inner(*args, **kwargs):
            timefunc_start_time = datetime.datetime.now()
            print('\nStart Time: {}'.format(timefunc_start_time))
            result = func(*args, **kwargs)
            print('Total Time Taken: {}'.format(datetime.datetime.now()-timefunc_start_time))
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

    if args.verbose:
        print('running intersection check')
    # intersection = [item for item in user_list if item in all_timeline_file_ids]

    # intersection = []
    # for item in tqdm.tqdm(all_timeline_file_ids, leave = False):
        # if item in tqdm.tqdm(user_list, leave=False):
            # intersection.append(item)

    # Use of hybrid method
    temp = set(all_timeline_file_ids)
    intersection = [value for value in user_list if value in temp]

    return intersection

def generate_filename(args, user_id, followers = True):

    if followers:
        return os.path.join(args.data_dir, 'followers_' + str(user_id) + '.jsonl')
    else:
        return os.path.join(args.data_dir, 'following_' + str(user_id) + '.jsonl')

def twarc_follow(args, user_id, followers = True):

    if followers:
        save_filename = generate_filename(args, user_id, followers)
        # this is to collect the followers of a user
        subprocess.run([
            'twarc2',
            'followers',
            str(user_id),
            str(save_filename)
        ])
        if args.verbose:
            print('saving at {}'.format(save_filename))

    else:
        save_filename = generate_filename(args, user_id, followers)
        # this is to collect the users that the one in question is following
        subprocess.run([
            'twarc2',
            'following',
            str(user_id),
            str(save_filename)
        ])
        if args.verbose:
            print('saving at {}'.format(save_filename))

@time_function
def main(args):
    intersection = check_existing_downloaded_timelines(args)
    if args.verbose:
        print('To collect: {} users.'.format(len(intersection)))

    for user in tqdm.tqdm(intersection):
        twarc_follow(args, user, followers = True)
        twarc_follow(args, user, followers = False)

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
        '--verbose',
        help='verbosity parameter',
        default = False,
        action= 'store_true'
    )

    args = parser.parse_args()
    if not args.output_dir:
        args.output_dir = args.data_dir

    main(args)