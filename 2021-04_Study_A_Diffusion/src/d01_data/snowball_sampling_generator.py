#!/usr/bin/python3.9

'''
2021-11-24 Snowball sampling. Empirical Distibution Sampling generates a list of users per group, and then this script

(a) augments the timelines for each one using the agumentation script
(b) collects interactions between users in this group
(c) samples a new list of users excluding existing ones
'''

import argparse
import glob
import logging
import os
import pickle
import random
import subprocess
import re
import datetime

import h5py
import numpy as np
import pandas as pd


def main(args):

    ############################################################################
    # (a) augment timelines
    ############################################################################
    random.seed(1)

    # set group_num
    group_num = str(int(re.split('_',args.user_list_file)[-2]) + 1)

    # collect user timelines and existing augmentations
    timeline_filelist = sorted(glob.glob(os.path.join(args.timeline_data_dir, 'timeline*.jsonl')))
    augmented_filelist = sorted(glob.glob(os.path.join(args.timeline_data_dir, 'augmented*.jsonl')))

    # set directories
    dirname = os.path.dirname(os.path.dirname(os.path.abspath((__file__))))
    logging.debug(f'this should show src/ folder: {dirname}')

    # extract start and stop times from FAS peak analysis and group number
    with h5py.File(args.FAS_peak_analysis_file, 'r') as f:
        x = f['segments']['selected_ranges'][int(group_num)-1]
        args.min_date = x[0].decode()
        args.max_date = x[1].decode()

    if len(timeline_filelist) == len(augmented_filelist):
        logging.info('No timelines need to be augmented. Continuing...')
    else:
        assert len(timeline_filelist) > len(augmented_filelist)
        logging.info('Timelines need augmenting.')

        # extract user ids from augmenting file ids
        augmented_ids = [re.split('[_.]',i)[-2] for i in augmented_filelist]

        # filter out timeline files that aren't already paired with an augmentation file
        timeline_filelist = [tf for tf in timeline_filelist if re.split('[_.]',tf)[-2] not in augmented_ids]

        logging.info(f'Number of timelines to augment: {len(timeline_filelist)}')

        # list of users to augment, passed to temporary pickle file.
        augmentation_obj_file = os.path.join(args.output_dir,'timelines_to_augment_TEMP.obj')

        # write to pickle file
        with open(augmentation_obj_file, 'wb') as f:
            pickle.dump(timeline_filelist, f)

        user_timeline_augmentation_py_file = os.path.join(dirname, 'd01_data','user_timeline_augmentation.py')
        subprocess.run([
            'python3.9',
            user_timeline_augmentation_py_file,
            args.FAS_dir,
            args.timeline_data_dir,
            args.min_date,
            args.max_date,
            '--custom_list',
            augmentation_obj_file,
            '--log_dir',
            args.log_dir,
            '--log_level',
            args.log_level
        ], cwd=os.path.join(dirname, 'd01_data'),check=True)

        os.remove(augmentation_obj_file)

    ############################################################################
    # (b) Collect interactions
    ############################################################################

    # check if interactions already exist for this group
    interactions_file = os.path.join(args.output_dir, 'interactions.hdf5')
    with h5py.File(interactions_file, 'r') as f:
        x = list(f.keys())
        x = [i for i in x if f'group_{group_num}_snowball_num' in x]
        if x:
            snowball_nums = [int(f[i].attrs[f'snowball_num']) for i in x]
            max_snowball = max(snowball_nums)
            snowball_num = max_snowball + 1
        else:
            snowball_num = 1

    custom_interactions_dataset_name = f'interactions_group_{group_num}_snowball_{snowball_num}'

    logging.info(f'Snowball number is {snowball_num}')
    logging.info(f'Custom interactions dataset name is: {custom_interactions_dataset_name}')
    logging.info(f'Detected group number: {group_num}')

    interaction_py = os.path.join(dirname, 'd02_intermediate','get_user_interaction_graph_edges.py')
    def run_interactions():
        subprocess.run([
            'python3.9',
            interaction_py,
            '--data_dir',
            args.timeline_data_dir,
            '--user_list_file',
            args.user_list_file,
            '--output_dir',
            args.interactions_dir,
            '--output_dataset_name',
            custom_interactions_dataset_name,
            '--log_dir',
            args.log_dir,
            '--log_level',
            args.log_level
        ],cwd=os.path.join(dirname, 'd02_intermediate'),check=True)


    # check if interactions for this already exist
    if os.path.isfile(interactions_file):

        with h5py.File(interactions_file, 'r') as f:
            x = list(f.keys())
        if custom_interactions_dataset_name in x:
            logging.warning(f'Name {custom_interactions_dataset_name} already exists in hdf5 file. Moving on.')
            pass
        else:
            run_interactions()
    else:
        run_interactions()

    # update attrs for sampling interaction dfs
    with h5py.File(interactions_file, 'a') as f:
        f[custom_interactions_dataset_name].attrs['snowball_num'] = snowball_num
        logging.info(f'Snowball number for this sample set to {snowball_num}')


    ############################################################################
    # (c) sample interactions df for user list and save to txt
    ############################################################################

    # recollect augmented list for ids
    augmented_filelist = sorted(glob.glob(os.path.join(args.timeline_data_dir, 'augmented*.jsonl')))
    augmented_filelist = [re.split('[_.]',i)[-2] for i in augmented_filelist]

    # read in interactions dataset using pandas
    df = pd.read_hdf(interactions_file, custom_interactions_dataset_name)

    def convert_in_reply_to(x):
        if x:
            return x[0]
        else:
            return None

    logging.debug('Converting "in_reply_to" column to all strings')
    df['in_reply_to'] = df['in_reply_to'].apply(convert_in_reply_to)
    unique_users = list(df['in_reply_to'].unique())

    # filter out users already collected (i.e. in augmented) and entries in df that had no reply to.
    users_to_sample = [i for i in unique_users if i not in augmented_filelist and i is not None]

    with open(args.user_list_file, 'r') as f:
        p = f.readlines()
        p = [str(i.replace('\n','')) for i in p]
        user_list = set(p)

    users_to_sample = [i for i in users_to_sample if i in user_list]

    sample_size = min(args.num_to_sample, len(users_to_sample))
    sampled_list = random.sample(users_to_sample, sample_size)

    logging.info(f'Number of sampled users returned: {sample_size}')

    sampled_users_txt_file = os.path.join(args.output_dir, f'group_{group_num}_snowball_{snowball_num}.txt')

    with open(sampled_users_txt_file, 'w') as f:
        for i in sampled_list:
            f.write(i)
            f.write('\n')

    logging.info(f'File saved to {sampled_users_txt_file}')
    logging.info(f'SCRIPT ENDED AT {datetime.datetime.now()}')

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='')

    parser.add_argument(
        '--FAS_dir',
        help='FAS directory'
    )

    parser.add_argument(
        '--timeline_data_dir',
        help='Where existing timeline files are. This will also be where the augmented files will be placed.'
    )

    parser.add_argument(
        '--interactions_dir',
        help='directory where interactions.hdf5 is or will be placed. File will be written to in append mode.'
    )

    parser.add_argument(
        '--output_dir',
        help='Directory to place sampling output txt.'
    )

    parser.add_argument(
        '--FAS_peak_analysis_file',
        help='FAS peak analysis hdf5'
    )

    parser.add_argument(
        '--user_list_file',
        help='user list file for this group'
    )

    parser.add_argument(
        '--num_to_sample',
        help='number of users to sample',
        default=2500,
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
        logging_file  = os.path.join(args.log_dir, f'{today_datetime}_snowball_sampling.log')
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

    args.today_datetime = today_datetime

    assert os.path.isdir(args.timeline_data_dir)
    assert os.path.isdir(args.interactions_dir)
    assert os.path.isdir(args.output_dir)
    assert os.path.isfile(args.user_list_file)

    # convert all to absolute paths
    current_file_absdir = os.path.dirname(os.path.abspath(__file__))
    args.timeline_data_dir = os.path.normpath(os.path.join(current_file_absdir, args.timeline_data_dir))
    args.interactions_dir = os.path.normpath(os.path.join(current_file_absdir, args.interactions_dir))
    args.output_dir = os.path.normpath(os.path.join(current_file_absdir, args.output_dir))
    args.user_list_file = os.path.normpath(os.path.join(current_file_absdir, args.user_list_file))

    logging.debug(f'Timeline dir is {args.timeline_data_dir}')
    logging.debug(f'Interactions dir is {args.interactions_dir}')
    logging.debug(f'Output dir is {args.output_dir}')
    logging.debug(f'User list file {args.user_list_file}')

    main(args)
