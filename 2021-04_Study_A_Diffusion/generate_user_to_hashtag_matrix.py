'''
Script to generate the user-to-hashtag file format to 

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
import array
import concurrent.futures
import csv
import glob
import os
import pickle
import re
from datetime import datetime
from itertools import repeat
from os.path import isfile

import jsonlines
import numpy as np
import pandas as pd
import scipy
import scipy.sparse
import tqdm
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer


def iterator_jsonl(input_file_list):

    for input_file in tqdm.tqdm(input_file_list, desc='Collecting Vocabulary'):

        with jsonlines.open(input_file) as reader:
            for tweet_jsonl in reader:
                tweet_list_in_file = tweet_jsonl['data']
                for tweet_data in tweet_list_in_file:
                    if 'text' in tweet_data:
                        yield tweet_data['text']

class IncrementalCOOMatrix(object):

    def __init__(self, shape, dtype):

        if dtype is np.int32:
            type_flag = 'i'
        elif dtype is np.int64:
            type_flag = 'l'
        elif dtype is np.float32:
            type_flag = 'f'
        elif dtype is np.float64:
            type_flag = 'd'
        else:
            raise Exception('Dtype not supported.')

        self.dtype = dtype
        self.shape = shape

        self.rows = array.array('i')
        self.cols = array.array('i')
        self.data = array.array(type_flag)

    def append(self, i, j, v):

        m, n = self.shape

        if (i >= m or j >= n):
            raise Exception('Index out of bounds')

        self.rows.append(i)
        self.cols.append(j)
        self.data.append(v)

    def tocoo(self):

        rows = np.frombuffer(self.rows, dtype=np.int32)
        cols = np.frombuffer(self.cols, dtype=np.int32)
        data = np.frombuffer(self.data, dtype=self.dtype)

        return scipy.sparse.coo_matrix((data, (rows, cols)),
                             shape=self.shape)

    def __len__(self):

        return len(self.data)

def process_file(input_file, file_list, vocab):

    row_index = file_list.index(input_file)
    print('{} start at {}'.format(row_index, datetime.now()))

    res = np.zeros(len(vocab),dtype=np.int32)

    # count lines
    line_count = 0
    with jsonlines.open(input_file) as reader:
        for _ in reader:
            line_count += 1

    with jsonlines.open(input_file) as reader:
        for tweet_jsonl in reader:
            tweet_list_in_file = tweet_jsonl['data']
            for tweet_data in tweet_list_in_file:
                if 'text' in tweet_data:
                    for column_index, phrase in enumerate(vocab):
                        if phrase in tweet_data['text']:
                            res[column_index]+=1
                            # vocab_matrix[row_index, column_index] += 1

    print('{} end at {}'.format(row_index, datetime.now()))

    return (row_index, res)

def main():

    # instantiate vectorizer
    vectorizer = CountVectorizer(
        input='content',
        ngram_range=(2,3),
        stop_words=stopwords.words().append('rt')
    )

    # obtain file list
    file_list = glob.glob(args.data_dir + '/timeline*.jsonl')
    file_list = sorted(file_list)

    # FOR UNIT TEST PURPOSES
    file_list = file_list[:150]

    # obtain vocabulary
    vectorizer.fit(iterator_jsonl(file_list))

    # obtain vocab
    vocab = sorted(list(vectorizer.vocabulary_.keys()))

    # generate vocab count matrix
    # choosing coo matrix instead of lil matrix because backend of lil is Python lists which don't scale well
    # and can't be easily parallelised
    vocab_matrix = IncrementalCOOMatrix((len(file_list), len(vocab)), np.int32)

    with concurrent.futures.ProcessPoolExecutor() as executor:

        # N.B. you cannot use lambda in the ProcessPoolExecutor because python uses pickle to
        # pass information to the children(?) processes
        #
        # Therefore OLD CODE: results = executor.map(lambda p: process_file(*p), multi_args)

        results = executor.map(process_file, file_list, repeat(file_list), repeat(vocab))

    for row_index, res in tqdm.tqdm(results, desc='writing counts to matrix', total=len(file_list)):
        for col_index, col_value in enumerate(res):
            vocab_matrix.append(row_index, col_index, col_value)

    # convert to COO matrix
    print('Converting to CSR Sparse Format')
    coo = vocab_matrix.tocoo()
    csr = coo.tocsr()

    # save file
    print('Saving to .npz file')
    scipy.sparse.save_npz('2021-04_Study_A_Diffusion/collection_results_2021_05_04_16_22/sparse_bispec_mat.npz', csr)
    with open('2021-04_Study_A_Diffusion/collection_results_2021_05_04_16_22/vocab.obj', 'wb') as f:
        pickle.dump(vocab, f)

    # generate csv in the right format
    with open('2021-04_Study_A_Diffusion/collection_results_2021_05_04_16_22/bispec_ready_counts.csv', 'w', newline='') as csvfile:

        file_writer = csv.writer(csvfile)
        file_writer.writerow(['Source', 'Target', 'weight'])

        nonzero_row_index_array, nonzero_col_index_array = csr.nonzero()

        for row_index, input_file in enumerate(tqdm.tqdm(file_list, desc='writing to final csv file')):
            user_id = os.path.split(input_file)[1]
            user_id = re.sub('\D','',user_id)

            # TODO: subset the nonzero_row_index_array
            for i,j in zip(nonzero_row_index_array, nonzero_col_index_array):
                if csr[i,j] > 10:
                    file_writer.writerow([user_id, vocab[j], csr[i,j]])

    print('done. Sum of matrix: {}'.format(np.sum(csr)))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='generate file in the first format fo bi-spectral clustering')

    parser.add_argument(
        'data_dir',
        help='data directory'
    )

    # parse args
    args = parser.parse_args()

    main()
