'''
2022-02-08: Script to collect tweets from a set of given tweet ids. Hydration.

A list of tweet ids is required as input

'''

import subprocess
import os
import argparse
import logging
import datetime

def main(args):

    # generate save filename
    save_filename = os.path.join(args.output_dir,'hydration.jsonl')

    # check if file already exists
    if args.overwrite:
        pass
        if os.path.isfile(save_filename):
            logging.warning(f'Already hydrated and overwrite flag is {args.overwrite}. Continuing...')
            return None

    subprocess.run(
        ['twarc2',
        'hydrate',
        args.user_list,
        save_filename]
    )

    logging.info(f'Hydration collected. Saved at {save_filename}.')

if __name__ == '__main__':

    # set up parsing arguments
    parser = argparse.ArgumentParser(description='Hydration')

    parser.add_argument(
        'user_list',
        help='list of users in a txt file. One user per line.'
    )

    parser.add_argument(
        '--output_dir',
        help='where to place the data files.'
    )

    parser.add_argument(
        '--overwrite',
        default=False,
        action='store_true'
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
        logging_file  = os.path.join(args.log_dir, f'{today_datetime}_hydration.log')
        logging.basicConfig(
            handlers=[
                logging.FileHandler(filename=logging_file,mode='w'),
                logging.StreamHandler()
            ],
            format=logging_fmt,
            level=logging_level,
            datefmt='%m/%d/%Y %I:%M:%S %p'
        )

        logging.info(f'Start time of twarc script is {today_datetime}')

    assert os.path.isdir(args.output_dir)
    assert os.path.isdir(args.log_dir)

    main(args)