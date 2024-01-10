'''
Script to generate primary hashtags of each user
'''
import os
import argparse
import glob
import re
import pickle
import jsonlines
from collections import defaultdict
import numpy as np
import datetime
from sklearn.feature_extraction.text import CountVectorizer
import logging
import unidecode

def iterator(jsonl_file):
    hts = []
    with jsonlines.open(jsonl_file,'r') as reader:
        for line in reader:
            tweets = line['data']
            for tweet in tweets:
                if 'entities' in tweet and 'hashtags' in tweet['entities']:
                    hts.append([unidecode.unidecode(i['tag'].lower()) for i in tweet['entities']['hashtags']])
                    # logging.info(hts)
    # hts = ' '.join(hts)
    hts = [item for sublist in hts for item in sublist]

    return hts

def main(args):

    with open(args.search_hashtags, 'r') as f:
        search_hashtags = f.readlines()
        search_hashtags = [i.replace('\n', '') for i in search_hashtags]
        search_hashtags = [i.replace('#', '') for i in search_hashtags]
        search_hashtags = [i.lower() for i in search_hashtags]
        search_hashtags.remove('وأناكمان')
    logging.debug(f'vocab is {search_hashtags}')

    assert os.path.isdir(args.input_dir)

    # collect files from directory:
    file_list = glob.glob(f'{args.input_dir}/*/timeline*.jsonl')
    unique_users = set([re.split('[_.]',i)[-2] for i in file_list])

    file_list = glob.glob(f'{args.input_dir}/FAS*.jsonl')

    assert len(file_list)>0

    #setup
    # output = {}
    # for user_id in unique_users:
    #     output[user_id] = {}
    #     for ht in search_hashtags:
    #         output[user_id][ht] = 0


    content_input = defaultdict(list)
    for index, FAS_file in enumerate(file_list):
        print(f'Processing {index} of {len(file_list)}: {FAS_file}')
        with jsonlines.open(FAS_file,'r') as reader:
            for line in reader:
                tweets = line['data']
                for tweet in tweets:
                    if tweet['author_id'] in unique_users:
                        if 'entities' in tweet and 'hashtags' in tweet['entities']:
                            hts = [unidecode.unidecode(i['tag'].lower()) for i in tweet['entities']['hashtags']]

                            content_input[tweet['author_id']] += hts

    print('Sorting user keys')
    user_order = sorted(content_input, key=int)
    final_content_input = []
    print('Assembling input')
    for user in user_order:
        final_content_input.append(' '.join(content_input[user]))

    vectorizer = CountVectorizer(
        input='content',
        vocabulary=search_hashtags
    )
    print('Vectorizing')
    res = vectorizer.fit_transform(final_content_input)

    assert res.shape[1] == len(search_hashtags)

    # for row in range(res.shape[0]):
    #     for i, e in enumerate(search_hashtags):
    #         output[user_order[row]][e] += res[row,i]


    # for file in file_list:
    #     # logging.info(f'processing {file}')
    #     user_id = re.split('[_.]', file)[-2]
    #     vectorizer = CountVectorizer(
    #         input='content',
    #         vocabulary=search_hashtags
    #     )
    #     res = vectorizer.fit_transform(iterator(file))
    #     res = np.sum(res, axis=0)
    #     # logging.debug(f'Shape of resulting matrix: {res.shape}')
    #     # logging.debug(f'number of columns should be: {len(search_hashtags)}')
    #     assert res.shape[1] == len(search_hashtags)
    #     # logging.debug(res.shape)
    #     for i, e in enumerate(search_hashtags):
    #         output[user_id][e] += res[0,i]

    #     #check that output is not zero
    #     checksum = 0
    #     for k, v in output[user_id].items():
    #         checksum+=v
    #     if checksum < np.sum(res):
    #         logging.debug(f'Checksum: {checksum}')
    #         logging.debug(f'Should be: {np.sum(res)}')
    #     if checksum == 0:
    #         logging.warning(f'WARNING ZERO VALUE {user_id}')
    #         logging.warning(f'Detected input was {list(iterator(file))}')

    #     # logging.info(f'end processing {file}')

    output_filename = os.path.join(args.output_dir, f'primary_ht_global.obj')

    with open(output_filename, 'wb') as f:
        pickle.dump((user_order,res),f)
    logging.info(f'Saved to {output_filename}.')

if __name__=='__main__':

    parser = argparse.ArgumentParser(description='Collect primary hashtags for each user')

    parser.add_argument(
        'input_dir',
        help='directory where timeline and augmentation files are. We just need the augmentation files'
    )

    parser.add_argument(
        'output_dir',
        help='Output directory'
    )

    parser.add_argument(
        'search_hashtags',
        help='reference file of hashtags'
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
        logging_file  = os.path.join(args.log_dir, f'{today_datetime}_primary_hashtags.log')
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

    main(args)