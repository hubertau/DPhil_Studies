#!/usr/bin/python3.9

'''
Script to tokenise and count the instances of hashtags and designated phrases.

Saves to files to be read by subsequent scripts to write to csv, which will finally be used in clustering.
'''

import argparse
import glob
import logging
import os
import pickle
import re
from datetime import datetime, timedelta
from os.path import isfile
from typing import NamedTuple

import h5py
import jsonlines
import numpy as np
import tqdm
from nltk.corpus import stopwords
from sklearn.exceptions import NotFittedError
from sklearn.feature_extraction.text import CountVectorizer


def unit_conv(val):
    return datetime.strptime('2017-10-16', '%Y-%m-%d') + timedelta(days=int(val))

def reverse_unit_conv(date):
    return (datetime.strptime(date, '%Y-%m-%d') - datetime.strptime('2017-10-16', '%Y-%m-%d')).days

class daterange(NamedTuple):
    start: str
    end: str
class TweetVocabVectorizer(object):

    def __init__(
        self,
        data_dir,
        output_dir,
        subset,
        ngram_range=(2,3),
        remove_stop_words=True,
        eot_token='eottoken',
        max_prominence_dates=None,
        hashtag=None,
        overwrite=False
    ):

        # set attributes
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.file_list = sorted(glob.glob(os.path.join(self.data_dir,'timeline*.jsonl')))
        self.augmented_file_list = sorted(glob.glob(os.path.join(self.data_dir,'augmented*.jsonl')))
        self.ngram_range = ngram_range
        self.eot_token = eot_token
        self.remove_stop_words = remove_stop_words
        self.token_pattern = r"(?u)#?\b\w\w+\b"
        self.subset = subset
        self.max_prominence_dates = max_prominence_dates
        self.hashtag=hashtag
        self.overwrite=overwrite

        # sanity check for user timelines and agumented tweets
        assert len(self.file_list) == len(self.augmented_file_list)
        assert all([
            re.split('[_.]', self.file_list[i])[-2] == re.split('[_.]',self.augmented_file_list[i])[-2] for i in range(len(self.file_list))
        ])

        # check if input directories are directories
        assert os.path.isdir(self.data_dir)
        assert os.path.isdir(self.output_dir)

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

    def _token_digit_hashtag(self,x):
        if x[0]=='#':
            return x
        if x[0].isdigit():
            return x
        return None

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
            tokens = [v for v in [self._token_digit_hashtag(x) for x in tokens] if v is not None]

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

    def _filter_tweet_date(self, tweet_data, before=True):
        tweet_created_at = datetime.fromisoformat(tweet_data['created_at'][:-1])
        if before:
            if tweet_created_at <= self.max_prominence_dates[self.hashtag]:
                return True
            else:
                return False
        elif before == False:
            if tweet_created_at > self.max_prominence_dates[self.hashtag]:
                return True
            else:
                return False
        if before is None:
            # TODO ADD CASE FOR NONE FOR FULL VECTORIZING
            return True

    def iterator_jsonl(self, before=True):

        """
        Iterator to yield a raw input string from a user file.

        Yields:
            [type]: [description]
        """

        if self.subset:
            self.iter_list = self.file_list[:self.subset]
            self.augmented_iter_list = self.augmented_file_list[:self.subset]
        else:
            self.iter_list = self.file_list
            self.augmented_iter_list = self.augmented_file_list

        # set eot_token join string
        eot_join_str = ' ' + self.eot_token + ' '

        total_files=len(self.iter_list)

        logging.debug(f'Before flag for iterator is {before}')

        for index, input_file in enumerate(self.iter_list):

            if index%100 == 0:
                logging.info(f'Completed {index-1} files of {total_files} for ngram range {self.ngram_range}')

            user_joined_tweet_body = []

            with jsonlines.open(input_file) as reader:
                for tweet_jsonl in reader:
                    tweet_list_in_file = tweet_jsonl['data']
                    for tweet_data in tweet_list_in_file:
                        if self._filter_tweet_date(tweet_data, before) and 'text' in tweet_data:
                            user_joined_tweet_body.append(tweet_data['text'])

            # incorporate augmented data too.
            with jsonlines.open(self.augmented_iter_list[index]) as reader:
                for tweet in reader:
                    if self._filter_tweet_date(tweet, before) and 'text' in tweet:
                        user_joined_tweet_body.append(tweet['text'])

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

        if self.subset:
            self.iter_list = self.file_list[:self.subset]
            self.augmented_iter_list = self.augmented_file_list[:self.subset]

        for file_name in tqdm.tqdm(self.iter_list):
            with jsonlines.open(file_name) as reader:
                for tweet_jsonl in reader:
                    tweet_list_in_file = tweet_jsonl['data']
                    for tweet_data in tweet_list_in_file:
                        if 'entities' in tweet_data:
                            if 'hashtags' in tweet_data['entities']:
                                hts = [i['tag'].lower() for i in tweet_data['entities']['hashtags']]
                                hashtag_set.add(hts)

        for file_name in tqdm.tqdm(self.augmented_iter_list):
            with jsonlines.open(file_name) as reader:
                for tweet in reader:
                    if 'entities' in tweet:
                        if 'hashtags' in tweet_data['entities']:
                            hts = [i['tag'].lower() for i in tweet_data['entities']['hashtags']]
                            hashtag_set.add(hts)

        self.hashtag_set = hashtag_set

        # add hashes to the beginning of each hashtag! This is so the CountVectorizer can only pick out instances of words used as hashtags and not the content of the hashtag used elsewhere.
        self.hashtag_set = ['#' + hashtag for hashtag in self.hashtag_set] 
        print('done')

        return hashtag_set

    def _no_overwrite(self, before=True):

        before_str = '_after'
        if before:
            before_str = '_before'
        elif before is None:
            before_str = ''

        self.mat_filename = os.path.join(self.output_dir,f'user_count_mat_ngram_{self.ngram_range[0]}{self.ngram_range[1]}_{self.hashtag}{before_str}.obj')

        self.vectorizer_filename=os.path.join(self.output_dir,f'vectorizer_ngram_{self.ngram_range[0]}{self.ngram_range[1]}_{self.hashtag}{before_str}.obj')

        self.mapping_filename = os.path.join(self.output_dir,f'mapping_ngram_{self.ngram_range[0]}{self.ngram_range[1]}_{self.hashtag}{before_str}.obj')

        # return False to overwrite
        if os.path.isfile(self.mat_filename) and os.path.isfile(self.vectorizer_filename) and os.path.isfile(self.mapping_filename) and not self.overwrite:
            return True
        else:
            return False

    @time_function
    def fit(self, before=True):

        self.before=before
        self.fitted=False

        if self._no_overwrite(before):
            logging.info(f'No files written, overwrite flag is {self.overwrite} and files exist')
            return None

        # 2021-06-16: draft new CountVectorizer with proper custom analyzer
        # logic: with the right analzyer that first gets all n>1 n-grams and then drops all non-hashtag unigrams only a single pass should be required.
        self.vectorizer = CountVectorizer(
            input='content',
            analyzer=self.custom_analyzer
        )

        try:
            self.user_vocab_matrix = self.vectorizer.fit_transform(self.iterator_jsonl(before=before))
            self.fitted=True
        except ValueError:
            logging.warning('Empty vocabulary detected.')

        # set count of any token including the end_of_tweet_token to zero.
        logging.info('getting mapping between feature names and indices')
        try:
            self.mapping = self.vectorizer.get_feature_names_out()
        except NotFittedError:
            logging.warning('Not fitted because empty vocabulary.')
        logging.info('getting mapping between feature names and indices... done')



    def _check_vectorizer_output(self):

        if not self.fitted:
            logging.warning('Fitting not complete. No checking is conducted')
            return None

        if self._no_overwrite(self.before):
            logging.info(f'No vectorizer output checked because file exists.')
            return None

        def assert_helper(condition, true_msg='Condition Fulfilled', false_msg='Error'):
            assert(condition), false_msg
            logging.info(true_msg)

        assert_helper(len(self.mapping)>0, 'Vocabulary is non-zero. Check OK.')
        user_vocab_matrix_sum = np.sum(self.user_vocab_matrix)
        assert_helper(user_vocab_matrix_sum>0, 'Count Matrix has non-zero sum, and is {}. Check OK.'.format(user_vocab_matrix_sum))

        metoosum = np.sum(['#metoo' in i for i in self.mapping])
        assert_helper(metoosum>0, 'At least #MeToo hashtag is in vocab and has count {}. Check OK.'.format(metoosum))

        elements_with_eot_token = np.sum(['eottoken' in i for i in self.mapping])
        assert_helper(elements_with_eot_token==0, 'No eot_tokens found in vocabulary. Check OK.', '{} counts of eot_token found in voabulary! Check NOT OK.'.format(elements_with_eot_token))

        logging.info('All Checks OK.')

    @time_function
    def save_files(self):

        if not self.fitted:
            logging.warning('No files saved. No fitting was done.')
            return None

        if self._no_overwrite(self.before):
            logging.info(f'No files saved.')
            return None

        with open(self.mat_filename, 'wb') as f:
            pickle.dump(self.user_vocab_matrix,f)

        with open(self.vectorizer_filename, 'wb') as f:
            pickle.dump(self.vectorizer,f)

        with open(self.mapping_filename, 'wb') as f:
            pickle.dump(self.mapping,f)

        logging.info('Files Saved.')


def main(args):

    vocab_vectorizer = TweetVocabVectorizer(
        args.data_dir,
        args.output_dir,
        args.subset,
        ngram_range=args.ngram_range,
        remove_stop_words=False,
        max_prominence_dates=args.most_prominent_peaks,
        hashtag=args.hashtag,
        overwrite=args.overwrite
    )

    if args.hashtag is not None:
        # _ = vocab_vectorizer.get_hashtag_vocab()
        vocab_vectorizer.fit(before=True)
        vocab_vectorizer._check_vectorizer_output()
        vocab_vectorizer.save_files()
        logging.info('Fitting complete for before. Running Checks')

        vocab_vectorizer.fit(before=False)
        vocab_vectorizer._check_vectorizer_output()
        vocab_vectorizer.save_files()
        logging.info('Fitting complete for before. Running Checks')

    elif args.hashtag is None:
        vocab_vectorizer.fit(before=None)
        vocab_vectorizer._check_vectorizer_output()
        vocab_vectorizer.save_files()
        logging.info('Fitting complete for before. Running Checks')


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
        '--FAS_peak_analysis_file',
        help='FAS peak analysis file for peak locations'
    )

    parser.add_argument(
        '--search_hashtags',
        help='search hashtags file'
    )

    parser.add_argument(
        '--group_num'
    )

    parser.add_argument(
        '--hashtag_num',
        help='hashtag index for which peaks to collect',
        type=int
    )

    parser.add_argument(
        '--subset',
        help='debugging purposes. Index to iterate timeline files over.',
        default = None,
        type = int
    )
 
    parser.add_argument(
        '--overwrite',
        help='whether to overwrite existing files',
        default=False,
        action='store_true'
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
        today_datetime = str(datetime.now().strftime('%Y_%m_%d_%H_%M_%S'))
        logging_file  = os.path.join(args.log_dir, f'{today_datetime}_vectorizer.log')
        logging.basicConfig(
            handlers=[
                logging.FileHandler(filename=logging_file,mode='w'),
                logging.StreamHandler()
            ],
            format=logging_fmt,
            level=logging_level,
            datefmt='%m/%d/%Y %I:%M:%S %p'
        )

        logging.info(f'Start time of script is {today_datetime}')

    # parse args
    args = parser.parse_args()

    args.ngram_range = (int(args.ngram_range[0]), int(args.ngram_range[1]))

    #obtain peak times again
    with h5py.File(args.FAS_peak_analysis_file, 'r') as f:
        FAS_peaks = f['peak_detections']
        x = f['segments']['selected_ranges'][int(args.group_num)-1]
        group_date_range = daterange(
            start = x[0].decode(),
            end = x[1].decode()
        )

        args.most_prominent_peaks = {}
        for name, h5obj in FAS_peaks.items():

            peak_locations = h5obj['peak_locations']
            peak_locations = [(i,e) for i,e in enumerate(h5obj['peak_locations']) if (unit_conv(e) > datetime.strptime(group_date_range.start, '%Y-%m-%d')) and (unit_conv(e) < datetime.strptime(group_date_range.end, '%Y-%m-%d'))]
            peak_indices = [i[0] for i in peak_locations]
            prominences = [element for index, element in enumerate(h5obj['prominences']) if index in peak_indices]
            if len(prominences) == 0:
                continue
            max_prominence = np.argmax(prominences)
            args.most_prominent_peaks[name] = unit_conv(peak_locations[max_prominence][1])

    #load in search hashtags
    with open(args.search_hashtags, 'r') as f:
        search_hashtags = f.readlines()
        search_hashtags = [i.replace('\n', '') for i in search_hashtags]
        search_hashtags = [i.replace('#', '') for i in search_hashtags]
        args.search_hashtags = [i.lower() for i in search_hashtags]
        args.search_hashtags.remove('وأناكمان')

    # check that the chosen hashtag is in the keys
    if args.hashtag_num is not None:
        args.hashtag = args.search_hashtags[args.hashtag_num]
        args.hashtag = args.hashtag.lower()
    else:
        args.hashtag = None

    logging.info(f'Hashtag is {args.hashtag}')
    logging.info(f'Overwrite flag is{args.overwrite}')

    if args.hashtag not in list(args.most_prominent_peaks.keys()):
        logging.warning(f'{args.hashtag} is not in keys for this group. Ending.')
    else:
        main(args)


