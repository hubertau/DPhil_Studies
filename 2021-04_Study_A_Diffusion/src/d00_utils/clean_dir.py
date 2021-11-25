#!/usr/bin/python3.9

import logging
import argparse
import glob
import os
import datetime
import re

def main(args):

    timeline_filelist = sorted(glob.glob(os.path.join(args.data_dir, 'timeline*.jsonl')))
    augmented_filelist = sorted(glob.glob(os.path.join(args.data_dir, 'augmented*.jsonl')))

    with open(args.user_list_file, 'r') as f:
        p = f.readlines()
        p = [str(i.replace('\n','')) for i in p]
        user_list = set(p)

    timelines_to_remove = [i for i in timeline_filelist if re.split('[_.]', i)[-2] not in user_list]
    augmented_to_remove = [i for i in augmented_filelist if re.split('[_.]', i)[-2] not in user_list]

    assert (len(timelines_to_remove) != len(timeline_filelist)) or len(timeline_filelist) == 0
    assert (len(augmented_to_remove) != len(augmented_filelist)) or len(augmented_filelist) == 0
    logging.info(f'removing {len(augmented_to_remove)} augmented of {len(augmented_filelist)} original')
    logging.info(f'removing {len(timelines_to_remove)} timelines of {len(timeline_filelist)} original')

    for i in timelines_to_remove:
        os.remove(i)
        logging.info(f'Removing {i}')
    for j in augmented_to_remove:
        os.remove(j)
        logging.info(f'Removing {j}')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='')

    parser.add_argument(
        '--data_dir',
        help='timeline and aug files'
    )

    parser.add_argument(
        '--user_list_file'
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
        logging_file  = os.path.join(args.log_dir, f'{today_datetime}_clean_dir.log')
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

    main(args)