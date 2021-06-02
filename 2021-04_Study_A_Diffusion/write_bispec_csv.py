import csv
import glob
import os
import pickle
import re

import numpy as np
import scipy
import scipy.sparse
import tqdm
from numpy.core.fromnumeric import nonzero, searchsorted

# open the saved files
with open('collection_results_2021_05_04_16_22/user_count_mat.obj', 'rb') as f:
    csr = pickle.load(f)
with open('collection_results_2021_05_04_16_22/vectorizer.obj', 'rb') as f:
    vectorizer = pickle.load(f)
with open('collection_results_2021_05_04_16_22/mapping.obj', 'rb') as f:
    mapping = pickle.load(f)

# obtain file list
file_list = glob.glob('collection_results_2021_05_04_16_22/data/timeline*.jsonl')
file_list = sorted(file_list)

# FOR UNIT TEST PURPOSES
file_list = file_list

mapping = vectorizer.get_feature_names()
mapping = np.array(mapping)

# generate csv in the right format
with open('collection_results_2021_05_04_16_22/bispec_ready_counts.csv', 'w', newline='') as csvfile:

    file_writer = csv.writer(csvfile)
    file_writer.writerow(['Source', 'Target', 'weight'])

    nonzero_row_index_array, nonzero_col_index_array = csr.nonzero()

    # drop indices with eot_tokens
    nonzero_col_index_sort_indices = nonzero_col_index_array.argsort()
    nonzero_col_index_array = nonzero_col_index_array[nonzero_col_index_sort_indices]
    nonzero_row_index_array = nonzero_row_index_array[nonzero_col_index_sort_indices]

    # define tokens to drop
    tokens_to_drop = ['eot_token','rt']

    # mask = np.ones(len(mapping), dtype=bool)

    for token in tokens_to_drop:
        print('getting locations of {} token'.format(token))
        mapping_token_locs = np.flatnonzero(np.core.defchararray.find(mapping,token)!=-1)
        print('done')

        # searchsorted doesn't find multiple instances, just matching for each in the mapping_token_locs.
        # coord_token_locs = np.searchsorted(nonzero_col_index_array, mapping_token_locs)

        # mask = np.ones(len(nonzero_col_index_array),dtype=bool)


        mask = np.logical_not(np.isin(nonzero_col_index_array,mapping_token_locs))

        # # mask, don't delete
        # mask[token_locs] = False
        nonzero_col_index_array = nonzero_col_index_array[mask]
        nonzero_row_index_array = nonzero_row_index_array[mask]
        # mapping                 = np.delete(mapping, token_locs)

    for row_index, input_file in enumerate(tqdm.tqdm(file_list, desc='writing to final csv file')):
        user_id = os.path.split(input_file)[1]
        user_id = re.sub('\D','',user_id)

        # mask_for_this_row = np.logical_and(mask,nonzero_row_index_array==row_index)

        for i,j in tqdm.tqdm(zip(nonzero_row_index_array[nonzero_row_index_array==row_index],
                                 nonzero_col_index_array[nonzero_row_index_array==row_index]),
                                 total=np.sum(nonzero_row_index_array==row_index),
                                 leave=False,
                                 desc='writing user {}. ({} out of {})'.format(user_id, row_index+1, len(file_list))):
            # if j in eot_token_locs:
            #     continue
            if csr[i,j] > 10:
                file_writer.writerow([user_id, mapping[j], csr[i,j]])
