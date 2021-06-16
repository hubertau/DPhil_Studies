'''
Script to tokenise and count the instances of hashtags and designated phrases.

Saves to files to be read by subsequent scripts to write to csv, which will finally be used in clustering.
'''

import argparse
import glob
import os
import pickle
import re
import string
import unicodedata
from datetime import datetime
from os.path import isfile

import jsonlines
import numpy as np
import scipy
import scipy.sparse
import tqdm
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer


class TweetVocabVectorizer(object):

    def __init__(self,data_dir,ngram_range=(2,3),stopwords_to_append=['rt']):

        self.stopwords_to_append = stopwords_to_append
        self.data_dir = data_dir
        self.file_list = sorted(glob.glob(self.data_dir + '/timeline*.jsonl'))
        self.ngram_range = ngram_range

    def custom_preprocessor(self, doc):

        # lowercase
        doc = doc.lower()

        # remove urls. from 
        doc = re.sub('\shttps?:\/\/\S*', '', doc, flags=re.MULTILINE)

        # now this is also an opportunity to do some light cleaning. here i remove ellipses
        doc = re.sub('\.{2,}', '', doc, flags=re.MULTILINE)

        # remove mentions
        doc = re.sub('(@\w+:|@\w+)', '', doc, flags=re.MULTILINE)

        # drop the retweet token
        doc = re.sub('rt','', doc, flags=re.MULTILINE)

        # strip accents
        # NOTE TO SELF: the r prefix to a string in python means it is a literal string
        doc = re.sub(r'[!"$%&\'…()*+,-.\/:;<=>?@\[\\\]\^_`{\|}~]', '', doc, flags=re.MULTILINE)

        # remove newline characters
        doc = re.sub(r'\n', ' ', doc, flags=re.MULTILINE)

        # replace multiple whitespaces with a single whitespace
        doc = re.sub("\s+", ' ', doc, flags=re.MULTILINE)

        # now remove leading and trailing whitespace
        doc = re.sub('^\s+', '', doc, flags=re.MULTILINE)
        doc = re.sub('\s+$', '', doc, flags=re.MULTILINE)

        return doc

    def iterator_jsonl(self, limit=None):

        if limit==None:
            iter_list = self.file_list
        else:
            iter_list = self.file_list[:limit]

        for input_file in tqdm.tqdm(iter_list, desc='CountVectorizer over collected users:'):

            user_joined_tweet_body = []

            with jsonlines.open(input_file) as reader:
                for tweet_jsonl in reader:
                    tweet_list_in_file = tweet_jsonl['data']
                    for tweet_data in tweet_list_in_file:
                        if 'text' in tweet_data:
                            user_joined_tweet_body.append(tweet_data['text'])

            # for the final yield, there needs to be an in-between character i can easily discard
            # so tokens spanning multiple documents can be discarded
            output =  ' eottoken '.join(user_joined_tweet_body)

            yield output

    def get_hashtag_vocab(self):

        print('collecting hashtag list')
        hashtag_set = set()

        for file_name in tqdm.tqdm(self.file_list):
            with jsonlines.open(file_name) as reader:
                for tweet_jsonl in reader:
                    tweet_list_in_file = tweet_jsonl['data']
                    for tweet_data in tweet_list_in_file:
                        if 'entities' in tweet_data:
                            if 'hashtags' in tweet_data['entities']:
                                hts = [i['tag'].lower() for i in tweet_data['entities']['hashtags']]
                                hashtag_set.update(hts)

        self.hashtag_set = hashtag_set

        # add hashes to the beginning of each hashtag! This is so the CountVectorizer can only pick out instances of words used as hashtags and not the content of the hashtag used elsewhere.
        self.hashtag_set = ['#' + hashtag for hashtag in self.hashtag_set] 
        print('done')

        return hashtag_set

    def fit(self):

        start_time = datetime.now()
        print('start time: {}'.format(start_time))

        # 2021-06-15: introduction of token_pattern argument: the tokenizer itself within the default CountVectorizer class always ignores punctuation surrounding a word even if it has 
        self.vectorizer_draft = CountVectorizer(
            input='content',
            ngram_range=self.ngram_range,
            stop_words=stopwords.words().extend(self.stopwords_to_append),
            preprocessor=self.custom_preprocessor,
            token_pattern=r"(?u)#?\b\w\w+\b"
        )

        # obtain vocabulary of ngrams
        print('collecting ngram vocab list')
        user_vocab_matrix = self.vectorizer_draft.fit_transform(self.iterator_jsonl())

        # obtain ngram vocab
        ngram_vocab = self.vectorizer_draft.get_feature_names()

        # append the hashtags to this ngram vocab, the hashtags are the only
        # unigrams desired.
        total_vocab = list(np.append(ngram_vocab,(list(self.hashtag_set))))

        # 2021-06-07: lowercase everything in vocab, otherwise it doesn't get picked up in the CountVectorizer.
        # see issue: https://github.com/scikit-learn/scikit-learn/issues/19311
        total_vocab = [vocab_item.lower() for vocab_item in total_vocab]

        # 2021-06-08: see https://tinyurl.com/y2sgrsh6 (StackOverflow)
        # when supplying a vocab you still need to supply ngram_range, this is not
        # inferred from the vocabulary provided as I suppose that adds a significant
        # overhead to information the user probably knows.
        self.vectorizer = CountVectorizer(
            input='content',
            stop_words=stopwords.words().extend(self.stopwords_to_append),
            vocabulary=total_vocab,
            ngram_range=(1,3),
            preprocessor=self.custom_preprocessor,
            token_pattern=r"(?u)#?\b\w\w+\b"
        )

        self.user_vocab_matrix = self.vectorizer.fit_transform(self.iterator_jsonl())

        # set count of any token including the end_of_tweet_token to zero.
        print('getting mapping between feature names and indices...')
        self.mapping = self.vectorizer.get_feature_names()
        print('done')

        print('done. Sum of matrix: {}'.format(np.sum(user_vocab_matrix)))
        print('Total time taken: {}'.format(datetime.now()-start_time))

    def save_files(self):

        collection_results_folder = os.path.split(self.data_dir)[0]

        with open(os.path.join(collection_results_folder,'user_count_mat.obj'), 'wb') as f:
            pickle.dump(self.user_vocab_matrix,f)
        with open(os.path.join(collection_results_folder,'vectorizer.obj'), 'wb') as f:
            pickle.dump(self.vectorizer,f)
        with open(os.path.join(collection_results_folder,'mapping.obj'), 'wb') as f:
            pickle.dump(self.mapping,f)

        print('files saved')


def main():

    vocab_vectorizer = TweetVocabVectorizer(args.data_dir)

    _ = vocab_vectorizer.get_hashtag_vocab()
    vocab_vectorizer.fit()
    vocab_vectorizer.save_files()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='vocab extraction from json files')

    parser.add_argument(
        'data_dir',
        help='data directory'
    )

    # parse args
    args = parser.parse_args()

    main()
