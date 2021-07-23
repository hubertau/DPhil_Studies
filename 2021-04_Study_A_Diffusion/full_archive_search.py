import argparse
import csv
import datetime
import glob
import json
import logging
import os
import subprocess
import time

import re
import jsonlines
import requests
from dateutil.relativedelta import *
class FAS_Collector(object):

    def __init__(self, args):

        self.end_time                   = args.end_time
        self.start_time                 = args.start_time
        self.output_dir                 = args.output_dir
        self.search_query_txt           = args.search_query_txt
        self.existing_collection_folder = args.existing_collection_folder

        # obtain search terms
        with open(self.search_query_txt, newline='') as f:
            self.terms = list(csv.reader(f))

        # unroll list
        self.terms = [i[0] for i in self.terms]
        self.search_query = ' OR '.join(self.terms)
        assert len(self.search_query) <= 1024, 'Search query is above 1024 characters in length'

        print('Collector Instance Created. Terms: {}'.format(self.terms))

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

    def check_existing_folder(self):

        if self.existing_collection_folder != None:
            self.OUTPUT_PATH = self.existing_collection_folder
            self.DATA_PATH   = os.path.join(self.OUTPUT_PATH, 'data')
        else:

            # obtain current run time for reuslts
            self.CURRENT_RUN_TIME = datetime.datetime.today()
            self.CURRENT_RUN_TIME = self.CURRENT_RUN_TIME.strftime("%Y_%m_%d_%H_%M")

            # creating new path
            self.OUTPUT_PATH = os.path.join('collection_results_' + self.CURRENT_RUN_TIME)
            self.DATA_PATH   = os.path.join(self.OUTPUT_PATH, 'data')

            # simply point OUTPUT_PATH and DATA_PATH to the correct places
            if os.path.isdir(self.OUTPUT_PATH) and os.path.isdir(self.DATA_PATH):
                pass
            else:
                # create folders
                os.makedirs(self.OUTPUT_PATH, exist_ok=True)
                os.makedirs(self.DATA_PATH, exist_ok=True)

    def set_up_logging(self):

        # set up logging file
        logging.basicConfig(filename=os.path.join(self.OUTPUT_PATH, 'full_archive_search.log'),
                            encoding='utf-8',
                            format='%(levelname)s:%(message)s',
                            level=logging.DEBUG)

        self.collection_start_time = datetime.datetime.now()

        logging.info('Full Archive Search Collection Start: {}'.format(self.collection_start_time))


    def get_next_filename(self):

        def datetime_from_string(input_string):
            return datetime.datetime.strptime(input_string, "%Y-%m-%d")

        self.first_run = True

        first_start = self.start_time
        first_end   = datetime_from_string(self.start_time) + relativedelta(months=+1)
        first_end   = first_end.strftime('%Y-%m-%d')

        self.current_start = self.start_time
        self.current_end  = first_end

        self.save_filename = os.path.join(self.DATA_PATH, 'FAS_' + first_start + '_' + first_end + '.jsonl') 

        existing_FAS_results = glob.glob(os.path.join(self.DATA_PATH, 'FAS*.jsonl'))

        if len(existing_FAS_results)>0:
            self.first_run = False
            existing_FAS_results = [re.split('[_.]',i)[-2] for i in existing_FAS_results]
            latest_file = max(existing_FAS_results, key=datetime_from_string)
            print('Latest file is: {}'.format(latest_file))
            new_start = datetime_from_string(latest_file).strftime('%Y-%m-%d')
            new_end   = datetime_from_string(latest_file) + relativedelta(months=+1)
            new_end = new_end.strftime('%Y-%m-%d')

            self.save_filename = os.path.join(self.DATA_PATH, 'FAS_' + new_start + '_' + new_end + '.jsonl')

            self.current_start = new_start
            self.current_end   = new_end

            # if new_end>specified end time
            if datetime_from_string(self.current_end) > datetime_from_string(self.end_time):
                print('ending collection. Current start: {}'.format(self.current_start))
                print('ending collection. Current end: {}'.format(self.current_end))
                self.save_filename = None

    @time_function
    def run_twarc(self):

        print('Collecting {}'.format(self.save_filename))

        subprocess.run(
                ['twarc2',
                'search',
                '--archive',
                '--max-results',
                '100',
                '--end-time',
                self.current_end,
                '--start-time',
                self.current_start,
                self.search_query,
                self.save_filename]
            )

    def collect(self):

        self.check_existing_folder()
        self.set_up_logging()
        logging.info(self.terms)

        self.get_next_filename()
        while self.save_filename != None:
            self.run_twarc()
            self.get_next_filename()

        end_time = datetime.datetime.now()
        print(end_time)
        logging.info('End Time: {}'.format(end_time))
        logging.info('Time Elapsed: {}'.format(end_time-self.collection_start_time))

def main():

    parser = argparse.ArgumentParser(description='Full Archive Search on Twitter. Supply search hashtags with search_hashtags.csv in the same directory.')

    parser.add_argument(
        '--search_query_txt',
        help='hashtag search queries in a txt file.'
    )

    parser.add_argument(
        '--start_time',
        help='end_time argument in format YYYY-MM-DD format',
    )

    parser.add_argument(
        '--end_time',
        help='end_time argument in format YYYY-MM-DD',
    )

    parser.add_argument(
        '--output_dir',
        help='directory to place outputs. defaults to current working directory',
        default = os.getcwd()
    )

    parser.add_argument(
        '--existing_collection_folder',
        help='existing collections folder',
        default = None
    )

    args = parser.parse_args()

    Collector = FAS_Collector(args)

    Collector.collect()

if __name__ == "__main__":
    main()
