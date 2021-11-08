#!/usr/bin/python3.9

'''
Script to obtain retweet edges from user timelines

'''

import argparse
import pickle
import glob
import os
import tqdm
import jsonlines
import json
from collections import defaultdict

def main(args):

    # process
    timeline_flist = glob.glob(os.path.join(args.data_dir, 'timeline*.jsonl'))

    # get userset
    with open(args.user_list_file, 'r') as f:
        p = f.readlines()
        p = [str(i.replace('\n','')) for i in p]
        user_list = set(p)

    # initiate defaultdict object to contain the users and their interactions
    interaction_edges = defaultdict(list)

    # 2021-10-26: creating new structure to filter out the interactions between users.
    for twitter_user_timeline in tqdm.tqdm(timeline_flist):
        with jsonlines.open(twitter_user_timeline) as reader:
            for tweet_jsonl in reader:

                # obtain current user id of the timeline file. To be used as the key in the interaction edges dict
                current_user_id = tweet_jsonl['data'][0]['author_id']

                # iterate over the tweets
                for tweet in tweet_jsonl['data']:

                    # check mentions. These are the users that the person @-ed in their tweet text. mentions will have length 0 if either none are mentioned or if none mentioned are in the desired user list. We make this check because this paper is only interested in the interactions between users.
                    try:
                        mentions = [i['id'] for i in tweet['entities']['mentions'] if i['id'] in user_list]
                    except:
                        mentions = []

                    # referenced tweets are those quoted or replied to.
                    contains_referenced_tweets = 'referenced_tweets' in tweet
                    if contains_referenced_tweets:

                        referenced_tweets = tweet['referenced_tweets']

                        # extract quote tweets. do not check for user_list because these are tweet ids not user ids
                        quotes = [i['id'] for i in referenced_tweets if i['type'] == 'quoted']

                        # extract replies. do not check for user_list because these are tweet ids not user ids
                        replies = [i['id'] for i in referenced_tweets if i['type'] == 'replied_to']

                        current_tweet_id = tweet['id']
                        current_tweet_created_at = tweet['created_at']
                        if len(quotes)==1 and len(replies) == 0:
                            in_reply_to=[]
                        else:
                            in_reply_to = tweet['in_reply_to_user_id']

                        # if no saveable mentions, quotes, or replies are found then 
                        if len(quotes) + len(replies) == 0 and len(mentions) == 0 and (str(in_reply_to) not in user_list):
                            continue

                        if 'entities' in tweet and 'hashtags' in tweet['entities']:
                            contains_hashtags = [i['tag'] for i in tweet['entities']['hashtags']]
                        else:
                            contains_hashtags = []

                        appending_dict = {
                            'id': current_tweet_id,
                            'mentions': mentions,
                            'quotes': quotes,
                            'replies': replies,
                            'created_at': current_tweet_created_at,
                            'in_reply_to': in_reply_to,
                            'contains_hashtags': contains_hashtags
                        }

                        interaction_edges[str(current_user_id)].append(appending_dict)

    with open(args.output_filename, 'wb') as f:
        pickle.dump(interaction_edges, f)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='extract interaction edges')

    parser.add_argument(
        '--data_dir',
        help='where the timeline files are, in .jsonl format',
        default='../../data/01_raw/'
    )

    parser.add_argument(
        '--user_list_file',
        help='user list file to compare to.',
        default='../../data/02_intermediate/user_list_full_0_nocount.txt'
    )

    parser.add_argument(
        '--output_filename',
        help='desired output filename',
        default='../../data/02_intermediate/interaction_edges.obj'
    )

    parser.add_argument(
        '--verbose',
        help='verbosity parameter',
        default=False,
        action='store_true'
    )

    args = parser.parse_args()

    main(args)
