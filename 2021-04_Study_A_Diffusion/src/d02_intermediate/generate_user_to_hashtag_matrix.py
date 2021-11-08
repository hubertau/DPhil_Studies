#!/usr/bin/python3.9

'''
Script to tokenise and count the instances of hashtags and designated phrases.

Saves to files to be read by subsequent scripts to write to csv, which will finally be used in clustering.
'''

import argparse
import glob
import os
import pickle
import re
from datetime import datetime
from os.path import isfile

import jsonlines
import numpy as np
import tqdm
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer

class TweetVocabVectorizer(object):

    def __init__(
        self,
        data_dir,
        output_dir,
        subset,
        low_memory,
        ngram_range=(2,3),
        remove_stop_words=True,
        eot_token='eottoken'
    ):

        # set attributes
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.low_memory = low_memory
        self.file_list = sorted(glob.glob(self.data_dir + '/timeline*.jsonl'))
        self.ngram_range = ngram_range
        self.eot_token = eot_token
        self.remove_stop_words = remove_stop_words
        self.token_pattern = r"(?u)#?\b\w\w+\b"
        self.subset = subset

    def time_function(func):

        """
        Wrapper function to time execution.
        """

        def inner(*args, **kwargs):
            start_time = datetime.now()
            print('\nStart Time: {}'.format(start_time))
            result = func(*args, **kwargs)
            print('Total Time Taken: {}'.format(datetime.now()-start_time))
            return result
        return inner

    def custom_preprocessor(self, doc):

        """Preprocess a collection of tweets.
        The following is done:
        * lowercase
        * remove URLs
        * remove ellipses
        * remove mentions
        * drop rt token
        * strip accents
        * remove newline characters
        * replace multiple whitespaces with a single whitespace.
        * remove trailing and leading shitespaces

        Returns:
            doc: preprocessed string.
        """

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

    def custom_tokenizer(self):

        """Build a function to identify tokens. The default token pattern identifies words AND hashtags.

        Returns:
            func: regex findall function compiled on token pattern supplied to class instance
        """

        token_pattern = re.compile(self.token_pattern, flags=re.MULTILINE)

        return token_pattern.findall

    def custom_analyzer(self, doc):

        """
        Custom analyzer for tweet docs.

        Arguments:
            doc: unpreprocessed string, output from self.iterator_jsonl().

        Returns:
            doc: sequentially preprocessed, then tokenized, and ngram generated to return to CountVectorizer the actual features to be counted.
        """

        # parts of this code adapted from https://github.com/scikit-learn/scikit-learn/blob/15a949460dbf19e5e196b8ef48f9712b72a3b3c3/sklearn/feature_extraction/text.py#L222

        # preprocess
        doc = self.custom_preprocessor(doc)

        # get tokens
        tokens = self.custom_tokenizer()(doc)

        # handle stop words
        # 2021-06-16 N.B. not removing stop words may be desired because phrases may need to contain them.
        if self.remove_stop_words:
            self.stop_words = stopwords.words()
            tokens = [w for w in tokens if (w not in self.stop_words)]

        # handle token n-grams
        min_n, max_n = self.ngram_range
        if max_n != 1:
            original_tokens = tokens
            # 2021-06-16 Hubert note: the original sklearn tokenizer here just iterates through all the unigrams. But I don't want that so only preserve the unigrams that are hashtags.
            tokens = [w for w in tokens if w[0]=='#']

            n_original_tokens = len(original_tokens)

            # bind method outside of loop to reduce overhead
            tokens_append = tokens.append
            space_join = " ".join

            for n in range(min_n,
                           min(max_n + 1, n_original_tokens + 1)):
                for i in range(n_original_tokens - n + 1):
                    current_token = original_tokens[i: i + n] 
                    if self.eot_token not in current_token:
                        tokens_append(space_join(current_token))

        return tokens

    def iterator_jsonl(self):

        """
        Iterator to yield a raw input string from a user file.

        Yields:
            [type]: [description]
        """

        if self.subset == None:
            self.iter_list = self.file_list
        else:
            assert type(self.subset) == int
            self.iter_list = self.file_list[:self.subset]


        # set eot_token join string
        eot_join_str = ' ' + self.eot_token + ' '

        for input_file in tqdm.tqdm(self.iter_list, desc='CountVectorizer over collected users:'):

            user_joined_tweet_body = []

            with jsonlines.open(input_file) as reader:
                for tweet_jsonl in reader:
                    tweet_list_in_file = tweet_jsonl['data']
                    for tweet_data in tweet_list_in_file:
                        if 'text' in tweet_data:
                            user_joined_tweet_body.append(tweet_data['text'])

            # for the final yield, there needs to be an in-between character i can easily discard
            # so tokens spanning multiple documents can be discarded
            output =  eot_join_str.join(user_joined_tweet_body)

            yield output

    def get_hashtag_vocab(self):

        """
        Retrieve all hashtags used as a set from the user files in the data directory.

        As of 2021-06-17, possibly obsolete since hastags can be collected directly from the text.

        Returns:
            hashtag_set: list of hashtags used, with hashtags attached to each string.
        """

        print('collecting hashtag list')
        hashtag_set = set()

        if self.subset == None:
            self.iter_list = self.file_list
        else:
            assert type(self.subset) == int
            self.iter_list = self.file_list[:self.subset]

        for file_name in tqdm.tqdm(self.iter_list):
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

    @time_function
    def fit(self):

        # 2021-06-16: draft new CountVectorizer with proper custom analyzer
        # logic: with the right analzyer that first gets all n>1 n-grams and then drops all non-hashtag unigrams only a single pass should be required.
        self.vectorizer = CountVectorizer(
            input='content',
            analyzer=self.custom_analyzer
        )

        self.user_vocab_matrix = self.vectorizer.fit_transform(self.iterator_jsonl())

        # set count of any token including the end_of_tweet_token to zero.
        print('\ngetting mapping between feature names and indices...')
        self.mapping = self.vectorizer.get_feature_names()
        print('done')

        if self.low_memory:
            print('\n Not converting to numpy array for memory reasons. Continuing...')
        else:
            # convert mapping to array form
            print('\nconverting feature mapping to array form...')
            self.mapping = np.array(self.mapping)
            print('done')



    def _check_vectorizer_output(self):

        def assert_helper(condition, true_msg='Condition Fulfilled', false_msg='Error'):
            assert(condition), false_msg
            print(true_msg)

        assert_helper(len(self.mapping)>0, 'Vocabulary is non-zero. Check OK.')
        user_vocab_matrix_sum = np.sum(self.user_vocab_matrix)
        assert_helper(user_vocab_matrix_sum>0, 'Count Matrix has non-zero sum, and is {}. Check OK.'.format(user_vocab_matrix_sum))
        metoosum = np.sum(['#metoo' in i for i in self.mapping])
        assert_helper(metoosum>0, 'At least #MeToo hashtag is in vocab and has count {}. Check OK.'.format(metoosum))
        # elements_with_eot_token = np.sum(np.core.defchararray.find(self.mapping,self.eot_token)!=-1)
        elements_with_eot_token = np.sum(['eottoken' in i for i in self.mapping])
        assert_helper(elements_with_eot_token==0, 'No eot_tokens found in vocabulary. Check OK.', '{} counts of eot_token found in voabulary! Check NOT OK.'.format(elements_with_eot_token))

        print('All Checks OK.')

    @time_function
    def save_files(self):

        with open(os.path.join(self.output_dir,'user_count_mat_ngram_' + str(self.ngram_range[0]) + str(self.ngram_range[1]) + '.obj'), 'wb') as f:
            pickle.dump(self.user_vocab_matrix,f)
        with open(os.path.join(self.output_dir,'vectorizer_ngram_' + str(self.ngram_range[0]) + str(self.ngram_range[1]) + '.obj'), 'wb') as f:
            pickle.dump(self.vectorizer,f)
        with open(os.path.join(self.output_dir,'mapping_ngram_' + str(self.ngram_range[0]) + str(self.ngram_range[1]) + '.obj'), 'wb') as f:
            pickle.dump(self.mapping,f)

        print('files saved')


def main():

    vocab_vectorizer = TweetVocabVectorizer(
        args.data_dir,
        args.output_dir,
        args.subset,
        args.low_memory,
        ngram_range=args.ngram_range,
        remove_stop_words=False
    )

    # _ = vocab_vectorizer.get_hashtag_vocab()
    vocab_vectorizer.fit()
    vocab_vectorizer.save_files()
    print('\nFitting complete. Running Checks:')
    vocab_vectorizer._check_vectorizer_output()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='vocab extraction from json files')

    parser.add_argument(
        'data_dir',
        help='data directory'
    )

    parser.add_argument(
        'output_dir',
        help='output directory'
    )

    parser.add_argument(
        '--ngram_range',
        help='specify ngram_range. place two integers contiguously for max and min, inclusive',
        default = (3,4)
    )

    parser.add_argument(
        '--subset',
        help='debugging purposes. Index to iterate timeline files over.',
        default = None,
        type = int
    )

    parser.add_argument(
        '--low_memory',
        help='Low Memory option. In particular it seems allocating memory for np.array(mapping) is large. Turn this off to not convert mapping with numpy.',
        default = False,
        action='store_true'
    )

    # parse args
    args = parser.parse_args()

    args.ngram_range = (int(args.ngram_range[0]), int(args.ngram_range[1]))

    main()
