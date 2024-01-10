'''
Given a dictionary of 
{
    'before':{
        'ht':int
    },
    'after':{
        'ht':int
    }
}
return for those clusters users and clusters they are in, for given users.

input: int_df and dictionary like above

output: int_df modified with cluster ids

'''

import re
import pickle
import pandas as pd
import numpy as np
from typing import NamedTuple
import glob
import argparse
import logging
import datetime
import os

# class single_cluster(NamedTuple):
#     n_clusters: int
#     n_invalid: int
#     total_ncut: float
#     norm_ncut: float
#     hashtag: str
#     before: str

def model_params_from_file(filename):
    return re.split('[_.]',filename)

def main(args):

    logging.debug(f'Dict file is {args.dictionary_file}')
    with open(args.dictionary_file, 'rb') as f:
        ht_best_cluster_dict = pickle.load(f)
    logging.debug(f'int df is {args.int_df}')
    with open(args.int_df, 'rb') as f:
        int_df = pickle.load(f)

    int_df['res_before'] = None
    int_df['res_after'] = None

    for ht in int_df['ht'].unique():
        try:
            # open bsc model files:
            best_cluster_before = ht_best_cluster_dict['before'][ht]
            best_cluster_after  = ht_best_cluster_dict['after'][ht]

            bsc_before = os.path.join(args.bsc_model_dir, f'bsc_python_cluster_{best_cluster_before}_ngram_{args.ngram_range}_min_{args.min_user}_{ht}_before.obj')
            assert os.path.isfile(bsc_before)
            logging.debug(f'BSC before is {bsc_before}')
            bsc_after = os.path.join(args.bsc_model_dir, f'bsc_python_cluster_{best_cluster_after}_ngram_{args.ngram_range}_min_{args.min_user}_{ht}_after.obj')
            assert os.path.isfile(bsc_before)
            logging.debug(f'BSC after is {bsc_after}')

            logging.debug('Attempting to load in bsc before')
            with open(bsc_before, 'rb') as f:
                model_before = pickle.load(f)
            logging.debug('Done. Attempting to load in bsc after')
            with open(bsc_after, 'rb') as f:
                model_after = pickle.load(f)

            features_before = os.path.join(args.vectorizer_dir,f'mapping_ngram_{args.ngram_range}_{ht}_before.obj')
            logging.debug(f'Features before is {features_before}')
            features_after = os.path.join(args.vectorizer_dir,f'mapping_ngram_{args.ngram_range}_{ht}_after.obj')
            logging.debug(f'Features after is {features_after}')

            logging.debug('Attempting to load in features before')
            with open(features_before, 'rb') as f:
                features_before = pickle.load(f)
            logging.debug('Attemting to load in features after')
            with open(features_after, 'rb') as f:
                features_after = pickle.load(f)


            user_doc_ids = sorted(glob.glob(os.path.join(f'/home/hubert/DPhil_Studies/2021-04_Study_A_Diffusion/data/01_raw/0{group_num}_group/','timeline*.jsonl')))
            user_doc_ids = [re.split('[_.]',i)[-2] for i in user_doc_ids]
            assert len(user_doc_ids)>0

            # csr_before = os.path.join(args.vectorizer_dir,f'mapping_ngram_{args.ngram_range}_{ht}_before.obj')
            # csr_after = os.path.join(args.vectorizer_dir,f'mapping_ngram_{args.ngram_range}_{ht}_after.obj') 

            # with open(csr_before, 'rb') as f:
            #     csr_before = pickle.load(f)
            # with open(csr_after, 'rb') as f:
            #     csr_after = pickle.load(f)

            logging.debug('Getting features names out')
            features_before = features_before.get_feature_names()
            logging.debug('Feature names out for ')
            features_after = features_after.get_feature_names()

            logging.debug(f'Filter only to this ht: {ht}')
            int_df_sub = int_df[int_df['ht']==ht]



            for row in int_df_sub.itertuples():

                logging.debug('gathering indices of where author is in features')
                author_index_before = np.where(features_before == row.author_id)
                indices_interacted_before = [np.where(features_before == i) for i in row.interacted_total_users]
                interacted_before = zip(row.interacted_total_users, indices_interacted_before)

                author_index_after = np.where(features_after == row.author_id)
                indices_interacted_after = [np.where(features_before == i) for i in row.interacted_total_users]
                interacted_after = zip(row.interacted_total_users, indices_interacted_after)

                author_interacted_before = {}
                author_interacted_after = {}
                for i in range(int(best_cluster_before)):
                    model_before_rows = model_before.get_indices(i)[0]
                    model_after_rows = model_after.get_indices(i)[0]
                    if author_index_before not in model_before_rows:
                        continue
                    else:
                        for user, index in interacted_before:
                            if index in model_before_rows:
                                author_interacted_before[user] = True
                            else:
                                author_interacted_before[user] = False
                    if author_index_after not in model_after_rows:
                        continue
                    else:
                        for user, index in interacted_after:
                            if index in model_after_rows:
                                author_interacted_after[user] = True
                            else:
                                author_interacted_after[user] = False

                int_df.loc[row.author_id,'res_before']=author_interacted_before
                int_df.loc[row.author_id,'res_after']=author_interacted_after

        except:
            logging.warning(f'{ht} failed, continuing.')
            continue

    with open(os.path.join(args.output_dir, f'int_df_{args.group_num}'), 'wb') as f:
        pickle.dump(int_df, f)
    # # open mapping also get relevent features names and user ids
    # with open(feature_mapping_file, 'rb') as f:
    #     feature_names = pickle.load(f)

    # # also load in csr
    # with open(csr_file,'rb') as f:
    #     csr = pickle.load(f)

    # # get also the userlist that was used to generate this for 'document' names
    # user_doc_ids = sorted(glob.glob(os.path.join(f'/home/hubert/DPhil_Studies/2021-04_Study_A_Diffusion/data/01_raw/0{args.group_num}_group/','timeline*.jsonl')))
    # user_doc_ids = [re.split('[_.]',i)[-2] for i in user_doc_ids]
    # assert len(user_doc_ids)>0

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='user pairs')

    parser.add_argument(
        'dictionary_file',
        help='dict file'
    )

    parser.add_argument(
        'int_df',
        help='int_df'
    )

    parser.add_argument(
        'output_dir'
    )

    parser.add_argument(
        'vectorizer_dir'
    )

    parser.add_argument(
        'timeline_dir'
    )

    parser.add_argument(
        'bsc_model_dir'
    )

    parser.add_argument(
        'ngram_range',
        type=str
    )

    parser.add_argument(
        'min_user'
    )

    parser.add_argument(
        'group_num',
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
        logging_file  = os.path.join(args.log_dir, f'{today_datetime}_user_pairs.log')
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

    logging.info(f'Group num: {args.group_num}')
    logging.info(f'Ngram range: {args.ngram_range}')
    logging.info(f'min user: {args.min_user}')

    main(args)