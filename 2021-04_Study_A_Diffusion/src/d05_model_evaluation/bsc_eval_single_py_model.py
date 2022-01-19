#!/usr/bin/python3.9

'''
This script is to evaluate the clustering ON A SINGLE PYTHON MODEL.
'''

import argparse
import datetime
import glob
import logging
import os
import re
import pickle
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from itertools import repeat
from time import sleep

import h5py
import numpy as np
from typing import NamedTuple

class ncut_result(NamedTuple):
    cluster_number: int
    ncut: float

# Function handling the intelligent hdf5 file opening
def open_hdf5(filename, *args, **kwargs):
    time_waited = 0
    time_to_wait = 10
    while True:
        try:
            hdf5_file = h5py.File(filename, *args, **kwargs)
            break  # Success!
        except OSError:
            logging.info('Waiting required.')
            sleep(time_to_wait)  # Wait a bit
            time_waited+=time_to_wait
            if time_waited % 60 == 0:
                logging.info(f'waited {time_waited} seconds to write')
    return hdf5_file

def ncut_cluster(cocluster, csr, i):

    logging.info(f'Total clusters {cocluster.n_clusters}. Processing no. {i}')

    rows, cols = cocluster.get_indices(i)
    if not (np.any(rows) and np.any(cols)):
        # return sys.float_info.max
        logging.info(f'Empty cluster so no result. Ending Processing no. {i}')
        return ncut_result(int(i), -1)
    row_complement = np.nonzero(np.logical_not(cocluster.rows_[i]))[0]
    col_complement = np.nonzero(np.logical_not(cocluster.columns_[i]))[0]
    # Note: the following is identical to X[rows[:, np.newaxis],
    # cols].sum() but much faster in scipy <= 0.16
    weight = csr[rows][:, cols].sum()
    if weight == 0:
        return ncut_result(int(i), -1)
    # weight = csr[rows[:, np.newaxis],cols].sum()
    cut = csr[row_complement][:, cols].sum() + csr[rows][:, col_complement].sum()

    logging.info(f'Total clusters {cocluster.n_clusters}. END Processing no. {i}')

    result = cut / weight

    return ncut_result(int(i), result)

def main(args):

    args.results_file = f'{args.results_file_base}{args.hashtag_str}{args.before_str}.obj'
    logging.debug(f'Results file is {args.results_file}')

    try:
        with open(args.results_file, 'rb') as f:
            model = pickle.load(f)
    except FileNotFoundError:
        logging.warning(f'File {args.results_file} not found. Ending...')
        return None

    args.csr_filename = os.path.join(args.csr_dir, f'user_count_mat_ngram_{args.ngram_range}{args.hashtag_str}{args.before_str}.obj')
    logging.debug(f'CSR file detected is {args.csr_filename}')

    with open(args.csr_filename, 'rb') as f:
        csr = pickle.load(f)

    if args.max_workers is None:
        args.max_workers = os.cpu_count()

    logging.info(f'Beginning ProcessPool Executor with {args.max_workers} workers.')
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        results = executor.map(ncut_cluster, repeat(model), repeat(csr), range(model.n_clusters))
    logging.info(f'End Processing')

    results = list(results)
    results = np.array(sorted(results, key=lambda x: x.cluster_number))

    ngram_range = re.split('[_.]', os.path.split(args.results_file)[-1])[5]
    logging.debug(f'DETECTED NGRAM RANGE: {ngram_range}')
    min_user    = re.split('[_.]', os.path.split(args.results_file)[-1])[7]
    logging.debug(f'DETECTED MIN_USER: {min_user}')

    save_file = os.path.join(args.output_dir, 'bispec_cluster_eval.hdf5')
    logging.info(f'Writing to file {save_file}')
    # How to use the function
    with open_hdf5(save_file, mode = 'a') as f:

        g = f.require_group(f'group_{args.group_num}')
        n = g.require_group(f'ngram_{ngram_range}')
        x = n.require_group(f'min_{min_user}')

        # no hashtag specified, this is evaluation on all timelines in group date range.
        if args.hashtag_str == '':

            if f'{model.n_clusters}' in x.keys():
                if args.overwrite:
                    del x[f'{model.n_clusters}']
                    d = x.create_dataset(f'{model.n_clusters}', data = results)
                else:
                    logging.warning(f'{model.n_clusters} key already exists and overwrite is {args.overwrite}. Not writing.')
            else:
                d = x.create_dataset(f'{model.n_clusters}', data = results)

        # if doing for hashtags
        else:

            y = x.require_group(args.hashtag)
            z = y.require_group(args.before_str[1:])

            if f'{model.n_clusters}' in z.keys():
                if args.overwrite:
                    del z[f'{model.n_clusters}']
                    d = z.create_dataset(f'{model.n_clusters}', data = results)
                else:
                    logging.warning(f'{model.n_clusters} key already exists and overwrite is {args.overwrite}. Not writing.')
            else:
                d = z.create_dataset(f'{model.n_clusters}', data = results)

    logging.info('Writing complete. Ending.')

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Evaluate the best bispectral clustering result')

    parser.add_argument(
        'results_file_base',
        help='particular .obj model file to load in'
    )

    parser.add_argument(
        'output_dir',
        help='output directory'
    )

    parser.add_argument(
        'csr_dir',
        help='csr matrix after vectorizing'
    )

    parser.add_argument(
        'ngram_range'
    )


    parser.add_argument(
        'group_num'
    )

    parser.add_argument(
        'hashtag_num',
        help='index of hashtag',
        type=int
    )

    parser.add_argument(
        'search_hashtags'
    )

    parser.add_argument(
        '--before',
        type=int,
        choices=[0,1]
    )

    parser.add_argument(
        '--overwrite',
        help='whether to overwrite',
        default=False,
        action='store_true'
    )

    parser.add_argument(
        '--max_workers',
        help='For multiprocessing. Default to None.',
        default=None,
        type = int
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
        logging_file  = os.path.join(args.log_dir, f'{today_datetime}__bispec_eval_py_pid_{os.getpid()}.log')
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

        logging.info(f'Start time of script is {today_datetime}')


    with open(args.search_hashtags, 'r') as f:
        search_hashtags = f.readlines()
        search_hashtags = [i.replace('\n', '') for i in search_hashtags]
        search_hashtags = [i.replace('#', '') for i in search_hashtags]
        args.search_hashtags = [i.lower() for i in search_hashtags]
        args.search_hashtags.remove('وأناكمان')

    # check that the chosen hashtag is in the keys
    if args.hashtag_num is not None:
        args.hashtag = args.search_hashtags[args.hashtag_num]
        args.hashtag = args.hashtag.lower()
        # assert args.hashtag in list(args.most_prominent_peaks.keys())
    else:
        args.hashtag = ''

    if args.before==1:
        args.before_str='_before'
    elif args.before==0:
        args.before_str='_after'
    else:
        args.before_str=''

    if args.hashtag != '':
        args.hashtag_str = f'_{args.hashtag}'
    else:
        args.hashtag_str = ''

    logging.info(f'overwrite flag is {args.overwrite}')

    try:
        main(args)
    except KeyboardInterrupt:
        logging.info('Keyboard Interrupt from user')

