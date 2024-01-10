#!/usr/bin/python3.9

'''

Script to write results of vectorizer objects etc. to csv ready for bispectarl clustering.

From the bi-spectral comm readme
# User-to-Hashtag File Format

The file can be either a ```.tsv``` file or a ```.csv``` file. It can also be gzipped if desired.
The file should have three columns, ```Source```, representing the User, ```Target```,
which specified the hashtags, and ```weight```, which gives the number of times this user used
the given hashtag. Note that zero-entries are not required,
i.e. values should only be specified where a user used a hashtag at least once.  For example:

```
Source, Target, weight
USER_ID_1, #cat, 4
USER_ID_1, #dog, 1
USER_ID_2, #cat, 10
...
'''

import argparse
import csv
from ctypes import ArgumentError
import glob
import os
import pickle
import re

import numpy as np
import tqdm
from numpy.core.fromnumeric import nonzero, searchsorted

def main(args):

    user_count_mat_file = args.user_count_mat_file
    # vectorizer_file = args.user_count_mat_file
    mapping_file = args.mapping_file
    file_list = glob.glob('../../data/01_raw/timeline*.jsonl')
    ngram_range=args.mapping_file[-6:-4]
    bispec_count_csv_file = '../../data/02_intermediate/bispec_ready_counts_' + ngram_range + '.csv'


    # open the saved files
    print('loading in files...')
    with open(user_count_mat_file, 'rb') as f:
        csr = pickle.load(f)
    # with open(vectorizer_file, 'rb') as f:
        # vectorizer = pickle.load(f)
    with open(mapping_file, 'rb') as f:
        mapping = pickle.load(f)

    # obtain file list
    file_list = sorted(file_list)

    # mapping = vectorizer.get_feature_names()
    # mapping = np.array(mapping)

    # generate csv in the right format
    with open(bispec_count_csv_file, 'w', newline='') as csvfile:

        file_writer = csv.writer(csvfile)
        file_writer.writerow(['Source', 'Target', 'weight'])

        nonzero_row_index_array, nonzero_col_index_array = csr.nonzero()

        # drop indices with eottokens
        nonzero_col_index_sort_indices = nonzero_col_index_array.argsort()
        nonzero_col_index_array = nonzero_col_index_array[nonzero_col_index_sort_indices]
        nonzero_row_index_array = nonzero_row_index_array[nonzero_col_index_sort_indices]

        # define tokens to drop
        tokens_to_drop = []

        # mask = np.ones(len(mapping), dtype=bool)

        for token in tokens_to_drop:

            # N.B. a number of options were tried here. The rationale for the final design is:
            # mapping contains indices and token string values. Searching this will allow finding the INDICES
            # of the relevant tokens to ignore. Also because of mapping containing the index information in its
            # own indices we cannot remove items from the array itself, this would mess up the referencing.
            #
            # The nonzero arrays, on the other hand, have no significance to the indices. The elements are the indices of tokens.
            # Therefore once the relevant token indices are found from mapping, they must be located (multiple instances thereof)
            # in the nonzero arrays.

            print('getting locations of {} token'.format(token))
            mapping_token_locs = np.flatnonzero(np.core.defchararray.find(mapping,token)!=-1)
            print('done')

            # build a mask for each token to drop
            # N.B. using the np.isin function to search for multiple instances is orders of magnitude
            # faster than any parallelisation at Python speed.
            mask = np.logical_not(np.isin(nonzero_col_index_array,mapping_token_locs))

            if len(nonzero_col_index_array) == np.sum(mask):
                print('WARNING: no instances of token [{}] were found. Is this intended?'.format(token))

            # we don't want to shorten or modify mapping, but can we happily truncate nonzero arrays.
            nonzero_col_index_array = nonzero_col_index_array[mask]
            nonzero_row_index_array = nonzero_row_index_array[mask]

        for row_index, input_file in enumerate(tqdm.tqdm(file_list, desc='writing to final csv file at {}'.format(bispec_count_csv_file))):

            # obtain user id from file path name.
            user_id = os.path.split(input_file)[1]
            user_id = re.sub('\D','',user_id)

            for i,j in tqdm.tqdm(zip(nonzero_row_index_array[nonzero_row_index_array==row_index],
                                    nonzero_col_index_array[nonzero_row_index_array==row_index]),
                                    total=np.sum(nonzero_row_index_array==row_index),
                                    leave=False,
                                    desc='writing user {}. ({} out of {})'.format(user_id, row_index+1, len(file_list))):
                if csr[i,j] > 5:
                    file_writer.writerow([user_id, mapping[j], csr[i,j]])

if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='Write csv file for bispectral clustering.')


    parser.add_argument(
        'user_count_mat_file',
        help='user count matrix file'
    )

    parser.add_argument(
        'mapping_file',
        help='mapping file'
    )

    args = parser.parse_args()

    main(args)