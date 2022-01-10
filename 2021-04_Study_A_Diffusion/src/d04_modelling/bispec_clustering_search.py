#!/usr/bin/python3.9

'''
Script to search through different bispectral clustering parameters. Allows for both Python and R implementations.
'''

import datetime
import glob
import logging
import os
import pickle
import pprint
import re
import subprocess
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import scipy.sparse
import tqdm
from numpy.core.fromnumeric import argsort, nonzero, searchsorted
from sklearn.cluster import SpectralCoclustering
from sklearn.metrics import consensus_score


class bispec_search(object):

    def __init__(self,
        ngram_range,
        data_dir,
        output_dir,
        bsc_range,
        interval,
        min_user,
        before,
        hashtag,
        overwrite,
        implementation = 'Python',
        max_workers = None
    ):
        self.ngram_range = ngram_range
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.bsc_range = bsc_range
        self.interval = interval
        self.min_user = min_user
        self.before = before
        self.hashtag = hashtag
        self.overwrite = overwrite
        self.type = implementation.lower()
        self.max_workers = max_workers

        logging.info('Data/Input directory is: {}'.format(self.data_dir))
        logging.info('Output directory is: {}'.format(self.output_dir))
        logging.info('ngram range is {}'.format(self.ngram_range))
        logging.info('Searching over bsc cluster sizes {} with interval {} and min_user parameter {}'.format(self.bsc_range, self.interval, self.min_user))

        # Obtain csv file for r implementation
        if self.type == 'r':

            # get all available csv values
            self.csv_file = glob.glob(os.path.join(self.data_dir, 'bispec_ready_counts*.csv'))

            # extract ngram ranges
            self.csv_file = [(re.split('[_.]', i)[-2], i) for i in self.csv_file]

            # select the one that matches
            self.csv_file = [i[1] for i in self.csv_file if i[0] == self.ngram_range]

            # check there is only one
            assert len(self.csv_file) == 1

            # set to the string
            self.csv_file = self.csv_file[0]

            logging.info('Detected csv file: {}'.format(self.csv_file))

    def time_function(func):

        """
        Wrapper function to time execution.
        """

        def inner(*args, **kwargs):
            timefunc_start_time = datetime.datetime.now()
            logging.info('\nStart Time: {}'.format(timefunc_start_time))
            result = func(*args, **kwargs)
            logging.info('Total Time Taken: {}'.format(datetime.datetime.now()-timefunc_start_time))
            return result
        return inner

    def model_python(self, cluster):

        model = SpectralCoclustering(n_clusters=cluster, random_state=0)

        logging.info(f'Modelling cluster size {cluster}.')

        results = model.fit(self.new_csr)

        save_filename = os.path.join(self.output_dir, f'bsc_python_cluster_{cluster}_ngram_{self.ngram_range}_min_{self.min_user}{self.hash_str}{self.before_str}.obj')
        logging.info(f'Saving to {save_filename}')

        with open(save_filename, 'wb') as f:
            pickle.dump(results, f)

        msg = 'saved at {}'.format(save_filename)

        logging.info(f'End modelling cluster size {cluster}.')
        return msg

    def cluster(self):

        if self.hashtag is None:
            self.hash_str = ''
        else:
            self.hash_str = f'_{self.hashtag}'

        if self.before is not None:
            self.before_str = f'_{self.before}'
        else:
            self.before_str = ''

        # Check if user wants to continue. First glob output directory
        existing_files = glob.glob(os.path.join(self.output_dir, f'bsc_python_cluster*_ngram_{self.ngram_range}_min_{self.min_user}{self.hash_str}{self.before_str}.obj'))
        logging.debug(f'Existing files: {existing_files}')

        # User may be runing bsc on a new range, so only reduce to ones that would be overwritten with this script
        existing_files = [(re.split('[_.]',i),i) for i in existing_files]
        existing_files = [i[1] for i in existing_files if
            self.bsc_range[0] <= int(i[0][-6]) <= self.bsc_range[1] and
            i[0][-4] == str(self.ngram_range[0])+str(self.ngram_range[1]) and
            i[0][-2] == str(self.min_user)
        ]

        # prompt for input
        if len(existing_files) > 0:
            if self.overwrite:
                pass
            else:
                logging.debug('Clustering Aborted.')
                return None

        if self.type == 'r':
            for cluster in tqdm.tqdm(range(self.bsc_range[0],self.bsc_range[1]+1,self.interval)):
                subprocess.run(
                    [
                        'Rscript',
                        'bispectral_clustering.R',
                        str(self.csv_file),
                        str(self.output_dir),
                        '--min_user',
                        str(self.min_user),
                        '--ncluster',
                        str(cluster)
                    ]
                )

        elif self.type == 'python':

            logging.info('Python implementation selected.')

            self.csr_path = os.path.join(self.data_dir, f'user_count_mat_ngram_{self.ngram_range}{self.hash_str}{self.before_str}.obj')

            logging.info('Attempting to load in file {}'.format(self.csr_path))

            # open the saved files
            with open(self.csr_path, 'rb') as f:
                self.csr = pickle.load(f)

            logging.info('CSR loaded in.')
            logging.info('Filtering for min user value...')

            self.new_csr = self.csr[:, np.array(self.csr.sum(axis=0) >= self.min_user).flatten()]

            logging.info('Done.')

            # 2021-07-14 Check NaN
            for row in range(self.new_csr.shape[0]):
                assert np.sum(np.isnan(np.array(self.new_csr[row,:].todense()))) == 0

            logging.info('NaN check complete.')
            logging.info('Beginning modelling')
            if self.max_workers is not None:
                workers = self.max_workers
            else:
                workers = os.cpu_count()
            logging.info(f'Running with {workers} workers')
            with ProcessPoolExecutor(max_workers=workers) as executor:
                result_messages = executor.map(
                    self.model_python,
                    range(self.bsc_range[0],self.bsc_range[1]+1,self.interval)
                )

            for i in result_messages:
                logging.info(i)

def main(args):

    clusterer = bispec_search(
        args.ngram_range,
        args.data_dir,
        args.output_dir,
        args.range,
        args.interval,
        args.min_user,
        args.before,
        args.hashtag,
        args.overwrite,
        implementation = args.implementation
    )

    clusterer.cluster()

if __name__ == '__main__':

    parser = ArgumentParser(description='Bispectral Cluster Search')

    parser.add_argument(
        'ngram_range',
        help='ngram range to consider. Enter in format e.g. 23 for range (2,3)'
    )

    parser.add_argument(
        'data_dir',
        help='Data directory'
    )

    parser.add_argument(
        '--output_dir',
        help='output directory. defaults to data_dir location.'
    )

    parser.add_argument(
        '--range',
        help='range of clusters to work scan through. In format [low]-[high]',
        default='10-20',
        type=str
    )

    parser.add_argument(
        '--interval',
        help='step value within range of search.',
        default=1,
        type=int
    )

    parser.add_argument(
        '--min_user',
        help='count for which a user must have to participate in clustering',
        default='10'
    )

    parser.add_argument(
        '--implementation',
        help='implementation to use. either python or R',
        default='R'
    )

    parser.add_argument(
        '--overwrite',
        help='overwrite flag',
        default=False,
        action='store_true'
    )

    parser.add_argument(
        '--before',
        help='whether input is split before and after peaks',
        type=int,
        choices=[0,1]
    )

    parser.add_argument(
        '--hashtag_num'
    )

    parser.add_argument(
        '--search_hashtags'
    )

    parser.add_argument(
        '--max_workers',
        help='Max CPU nodes to use',
        default=None
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
        logging_file  = os.path.join(args.log_dir, f'{today_datetime}_bispectral_clustering.log')
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


    #load in search hashtags
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
        assert args.hashtag in list(args.most_prominent_peaks.keys())
    else:
        args.hashtag = None

    args.range = args.range.split('-')
    args.range = (int(args.range[0]),int(args.range[1]))
    args.interval = int(args.interval)
    args.min_user = int(args.min_user)

    if not args.output_dir:
        args.output_dir = args.data_dir

    main(args)
