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
import glob
import pickle
from datetime import datetime
from os.path import isfile

import jsonlines
import numpy as np
import scipy
import scipy.sparse
import tqdm
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer


def iterator_jsonl(input_file_list):

    for input_file in tqdm.tqdm(input_file_list, desc='CountVectorizer over collected users:'):

        user_joined_tweet_body = []

        with jsonlines.open(input_file) as reader:
            for tweet_jsonl in reader:
                tweet_list_in_file = tweet_jsonl['data']
                for tweet_data in tweet_list_in_file:
                    if 'text' in tweet_data:
                        user_joined_tweet_body.append(tweet_data['text'])

        # for the final yield, there needs to be an in-between character I can easily discard
        # so tokens spanning multiple documents can be discarded
        yield ' <EOT_TOKEN> '.join(user_joined_tweet_body)

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

class TweetVocabVectorizer(object):

    def __init__(self,ngram_range=(2,3),stopwords_to_append=['rt']):
        pass

def main():

    start_time = datetime.now()
    print('start time: {}'.format(start_time))

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
    file_list = file_list

    # obtain vocabulary
    user_vocab_matrix = vectorizer.fit_transform(iterator_jsonl(file_list))

    # set count of any token including the end_of_tweet_token to zero.
    print('getting mapping between feature names and indices...')
    mapping = vectorizer.get_feature_names()
    print('done')

    # csc = scipy.sparse.lil_matrix(user_vocab_matrix)
    # # N.B. eot_token is lowercased because the CountVectorizer does the same
    # print('removing end of tweet tokens from vocabulary by setting frequency counts to 0...')
    # eot_token_locs = np.flatnonzero(np.core.defchararray.find(mapping,'eot_token')!=-1)
    # if len(eot_token_locs) > 0:
    #     for col_index in tqdm.tqdm(eot_token_locs, desc='setting to 0:'):
    #         csc[:,col_index] = np.zeros(csc.shape[0], dtype=np.int64)

    # save vectorizer
    # user_vocab_matrix = scipy.sparse.csr_matrix(user_vocab_matrix)
    # scipy.sparse.save_npz('2021-04_Study_A_Diffusion/collection_results_2021_05_04_16_22/sparse_bispec_mat.npz', csc)
    with open('2021-04_Study_A_Diffusion/collection_results_2021_05_04_16_22/user_count_mat.obj', 'wb') as f:
        pickle.dump(user_vocab_matrix,f)
    with open('2021-04_Study_A_Diffusion/collection_results_2021_05_04_16_22/vectorizer.obj', 'wb') as f:
        pickle.dump(vectorizer,f)
    with open('2021-04_Study_A_Diffusion/collection_results_2021_05_04_16_22/mapping.obj', 'wb') as f:
        pickle.dump(mapping,f)

    print('done. Sum of matrix: {}'.format(np.sum(user_vocab_matrix)))
    print('Total time taken: {}'.format(datetime.now()-start_time))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='generate file in the first format fo bi-spectral clustering')

    parser.add_argument(
        'data_dir',
        help='data directory'
    )

    # parse args
    args = parser.parse_args()

    main()
