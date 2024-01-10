#!/usr/bin/python3.9

'''
Script to generate the user list files required for the second data collection script to get timelines.
'''

import numpy as np
import plotnine
import os
import pandas as pd
import datetime
import pickle
import re
import scipy.signal
import matplotlib.pyplot as plt
import glob
import tqdm
import jsonlines
from collections import Counter

def main():

    FAS_object_folder = '../../data/02_intermediate/'
    data_flist = glob.glob('../../data/01_raw/FAS*.jsonl')
    existing_lists = glob.glob('../../data/02_intermediate/user_list_full_*_.txt')
    user_sets_path = '../../data/02_intermediate/user_sets.obj'
    selected_blocks = [1,4,7]
    verbose = False
    overwrite = False

    with open(os.path.join(FAS_object_folder, 'FAS_prominences.obj'), 'rb') as f:
        width_normalised_prominences = pickle.load(f)
    with open(os.path.join(FAS_object_folder, 'FAS_ranges.obj'), 'rb') as f:
        dated_reviewed_ranges = pickle.load(f)

    # select blocks

    # print(pd.DataFrame(width_normalised_prominences/max(width_normalised_prominences)))

    # get date range for selected blocks:
    selected_date_ranges = [dated_reviewed_ranges[i] for i in selected_blocks]
    # print(selected_date_ranges)

    if len(existing_lists) and not overwrite:
        # user_sets = []
        # for file in existing_lists:
        #     with open(file, 'r') as f:
        #         p = f.readlines()
        #     p = [i.replace('\n','') for i in p]
        #     p = [(int(i.split(',')[0]),int(i.split(',')[1])) for i in p]
        #     user_sets.append(p)

        print('Overwrite flag not set and already existing lists. Nothing done.')

    else:

        user_sets = [[] for i in range(len(selected_blocks))]

        # collect users for saving
        for file in tqdm.tqdm(data_flist):
            times=[]
            with jsonlines.open(file) as reader:
                for tweet_jsonl in reader:
                    tweet_list_in_file = tweet_jsonl['data']
                    for tweet_data in tweet_list_in_file:
                        tweet_created_at = datetime.datetime.fromisoformat(tweet_data['created_at'][:-1])
                        for i,e in enumerate(selected_date_ranges):
                            if e[0] <= tweet_created_at <= e[1]:
                                user_sets[i].append(tweet_data['author_id'])

        user_sets = [Counter(i) for i in user_sets]
        user_sets = [i.most_common() for i in user_sets]

        if verbose:
            for i,e in enumerate(user_sets):
                print(selected_date_ranges[i], len(e))

        with open(user_sets_path, 'wb') as f:
            pickle.dump(user_sets,f)

        for i,e in enumerate(user_sets):

            user_list_full_fn = '../../data/02_intermediate/user_list_full_' + str(i) + '_.txt'

            with open(user_list_full_fn, 'w') as f:
                for j in e:
                    f.write(','.join([str(s) for s in j]))
                    f.write('\n')

            print('{} saved.'.format(os.path.split(user_list_full_fn)[1]))

if __name__ == '__main__':
    main()