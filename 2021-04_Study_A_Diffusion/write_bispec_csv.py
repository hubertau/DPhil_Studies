import csv
import glob
import os
import pickle
import re

import scipy
import scipy.sparse
import tqdm
import numpy as np

csr = scipy.sparse.load_npz('collection_results_2021_05_04_16_22/sparse_bispec_mat.npz')
with open('collection_results_2021_05_04_16_22/vocab.obj', 'rb') as f:
    vocab = pickle.load(f)

# obtain file list
file_list = glob.glob('collection_results_2021_05_04_16_22/data/timeline*.jsonl')
file_list = sorted(file_list)

# FOR UNIT TEST PURPOSES
file_list = file_list[:150]

# generate csv in the right format
with open('collection_results_2021_05_04_16_22/bispec_ready_counts.csv', 'w', newline='') as csvfile:

    file_writer = csv.writer(csvfile)
    file_writer.writerow(['Source', 'Target', 'weight'])

    nonzero_row_index_array, nonzero_col_index_array = csr.nonzero()

    for row_index, input_file in enumerate(tqdm.tqdm(file_list, desc='writing to final csv file')):
        user_id = os.path.split(input_file)[1]
        user_id = re.sub('\D','',user_id)

        # TODO: subset the nonzero_row_index_array
        for i,j in tqdm.tqdm(zip(nonzero_row_index_array[nonzero_row_index_array==row_index],
                                 nonzero_col_index_array[nonzero_row_index_array==row_index]),
                                 total=np.sum(nonzero_row_index_array==row_index),
                                 leave=False,
                                 desc='writing user {}. ({} out of {})'.format(user_id, row_index+1, len(file_list))):
            if csr[i,j] > 10:
                file_writer.writerow([user_id, vocab[j], csr[i,j]])
        # for col_index in tqdm.tqdm(range(len(vocab)), leave=False):
        #     if csr[row_index, col_index] > 10:
        #         file_writer.writerow([user_id, vocab[col_index], csr[row_index, col_index]])
