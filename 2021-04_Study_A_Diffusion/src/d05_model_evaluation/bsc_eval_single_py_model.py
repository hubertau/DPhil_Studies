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
    while True:
        try:
            hdf5_file = h5py.File(filename, *args, **kwargs)
            break  # Success!
        except OSError:
            sleep(5)  # Wait a bit
    return hdf5_file

def ncut_cluster(cocluster, csr, i):

    logging.info(f'Total clusters {cocluster.n_clusters}. Processing no. {i}')

    rows, cols = cocluster.get_indices(i)
    if not (np.any(rows) and np.any(cols)):
        # return sys.float_info.max
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

    with open(args.results_file, 'rb') as f:
        model = pickle.load(f)

    with open(args.csr, 'rb') as f:
        csr = pickle.load(f)

    if args.max_workers is None:
        args.max_workers = os.cpu_count()

    logging.info(f'Beginning ProcessPool Executor with {args.max_workers} workers.')
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        results = executor.map(ncut_cluster, repeat(model), repeat(csr), range(model.n_clusters))
    logging.info(f'End Processing')

    results = list(results)
    results = np.array(sorted(results, key=lambda x: x.cluster_number))

    ngram_range = re.split('[_.]', args.results_file)[-4]
    min_user    = re.split('[_.]', args.results_file)[-2]

    save_file = os.path.join(args.output_dir, 'bispec_cluster_eval.hdf5')
    logging.info(f'Writing to file {save_file}')
    # How to use the function
    with open_hdf5(save_file, mode = 'a') as f:

        g = f.require_group(f'group_{args.group_num}')
        n = g.require_group(f'ngram_{ngram_range}')
        x = n.require_group(f'min_{min_user}')
        d = x.create_dataset(f'{model.n_clusters}', data = results)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Evaluate the best bispectral clustering result')

    parser.add_argument(
        'results_file',
        help='particular .obj model file to load in'
    )

    parser.add_argument(
        'output_dir',
        help='output directory'
    )

    parser.add_argument(
        'csr',
        help='csr matrix after vectorizing'
    )

    parser.add_argument(
        'group_num'
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

    try:
        main(args)
    except KeyboardInterrupt:
        logging.info('Keyboard Interrupt from user')

