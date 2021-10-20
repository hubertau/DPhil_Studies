#!/usr/bin/python3.9

'''
Script to obtain retweet edges from user timelines
'''


import pickle
import glob
import tqdm
import jsonlines
from collections import defaultdict

def main():

    timeline_flist = glob.glob('../../data/01_raw/timeline*.jsonl')

    # generate interaction graph

    # get userset
    with open('../../data/02_intermediate/user_list_full_0_nocount.txt', 'r') as f:
        p = f.readlines()
        p = [int(i.replace('\n','')) for i in p]
        user_list = p

    user_edges_fname = '../../data/d02intermediate/user_sets.obj'


    # if os.path.isfile(user_edges_fname):
    #     with open(user_edges_fname, 'rb') as f:
    #         user_edges = pickle.load(f)
    #     print('user_edges file found and loaded')
    # else:
    user_edges = defaultdict(list)

    # count = 0
    # internal_ref = 0
    for twitter_user_timeline in tqdm.tqdm(timeline_flist):
        with jsonlines.open(twitter_user_timeline) as reader:
            for tweet_jsonl in reader:
                tweet_includes = tweet_jsonl['includes']
                current_user_id = tweet_jsonl['data'][0]['author_id']
                for reffed_users in tweet_includes['users']:
                    # count += 1
                    if int(reffed_users['id']) in user_list:
                        # internal_ref+=1
                        user_edges[str(current_user_id)].append(str(reffed_users['id']))

    with open(user_edges_fname, 'wb') as f:
        pickle.dump(user_edges, f)

if __name__ == '__main__':

    main()