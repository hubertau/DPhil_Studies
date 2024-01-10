'''
Script to collect all the abm hdf5 files together
'''

import argparse
import datetime
import logging
import os
import re
import glob

import h5py
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor


def consolidate_one_file(file_tuple):

    file_index = file_tuple[0]
    logging.info(f'Processing {file_index}')
    file = file_tuple[1]

    with h5py.File(file, 'r') as f:
        params = f['params_array'][:]
        results = f['batch_result'][:,:,:,-1].sum(axis=1)

    return (params, results)

def main(args):

    batch_size = 48

    # create output array. Just needs to be number run * num_hashtags
    with h5py.File(args.hdf5_files[0]) as subfile:
        # assign key order
        attrs_batch_result_shape = subfile['batch_result'].shape
        attrs_key_order = subfile['batch_result'].attrs['key_order']
        attrs_param_order = subfile['params_array'].attrs['param_order']
        # agent_order = subfile['agent_order'][:]
    result_array = np.zeros(shape=(len(args.hdf5_files*batch_size),attrs_batch_result_shape[-2]))
    params_array = np.zeros(shape=(len(args.hdf5_files*batch_size), len(attrs_param_order.strip('][').split(', '))))


    logging.info('Beginning ProcessPoolExecutor')
    with ProcessPoolExecutor(max_workers=48) as executor:
        output = executor.map(consolidate_one_file, enumerate(args.hdf5_files))

    logging.debug(f'Processing output')
    index_tracker = 0
    for params, results in output:
        assert params.shape[0] == results.shape[0]
        length = params.shape[0]
        params_array[index_tracker:index_tracker+length] = params
        result_array[index_tracker:index_tracker+length] = results
        index_tracker += length

    logging.info('Writing to consolidation file')
    with h5py.File(args.consolidated_savepath, 'w') as consolidated_file:

        consolidated_file.create_dataset('result', data=result_array)
        consolidated_file['result'].attrs['key_order'] = attrs_key_order
        consolidated_file.create_dataset('params', data=params_array)
        consolidated_file['params'].attrs['param_order'] = attrs_param_order

    logging.info('Complete.')

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Script to collect hdf5 results together for agent-based model')

    parser.add_argument(
        '--hdf5_dir',
        help='hdf5 file directory'
    )

    parser.add_argument(
        '--output_dir',
        help='where to place results. Defaults to same directory'
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
        logging_file  = os.path.join(args.log_dir, f'{today_datetime}_abm_collect.log')
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


    # check file dir is indeed a file dir
    assert os.path.isdir(args.hdf5_dir)

    args.hdf5_files = glob.glob(os.path.join(args.hdf5_dir, 'ABM_output*batch*.hdf5'))
    assert len(args.hdf5_files) > 0

    group_num = int(re.split('[_.]', args.hdf5_files[0])[-4])

    savepath_end = f'ABM_output_consolidated_group_{group_num}.hdf5'
    if args.output_dir:
        assert os.path.isdir(args.output_dir)
        args.consolidated_savepath = os.path.join(args.output_dir, savepath_end)
    else:
        args.consolidated_savepath = os.path.join(args.hdf5_dir, savepath_end)
    logging.info(f'Output file is: {args.consolidated_savepath}')

    main(args)