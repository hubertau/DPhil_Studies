#!/usr/bin/python3.9

import jsonlines
import os
import tqdm
import glob
import pickle

def main():

    timeline_flist = glob.glob('../../data/01_raw/timeline*.jsonl')
    user_pecentage_results_file = '../../data/02_intermediate/user_percentage_results.obj'
    search_hashtags_file = '../../references/search_hashtags.txt'

    with open(search_hashtags_file, 'r') as f:
        search_hashtags = f.readlines()
        search_hashtags = [i.replace('\n', '') for i in search_hashtags]
        search_hashtags = [i.replace('#', '') for i in search_hashtags]

    user_percentage_results = []
    for twitter_user_timeline in tqdm.tqdm(timeline_flist):

        with jsonlines.open(twitter_user_timeline) as reader:
            user_count = 0
            user_hash_count = 0
            for line in reader:
                tweets = line['data']
                for tweet in tweets:
                    user_count += 1
                    if 'entities' in tweet:
                        if 'hashtags' in tweet['entities']:
                            for i in tweet['entities']['hashtags']:
                                if i in search_hashtags:
                                    user_hash_count += 1
        user_percentage_results.append((
            os.path.split(twitter_user_timeline)[1],
            user_hash_count,
            user_count,
            user_hash_count/user_count
        ))

    with open(user_pecentage_results_file, 'wb') as f:
        pickle.dump(user_percentage_results, f)

if __name__ == '__main__':
    main()