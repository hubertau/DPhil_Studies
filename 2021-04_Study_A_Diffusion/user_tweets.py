import datetime
import json
import logging
import argparse
import time
import os
import tqdm
import glob
import unique_users
import full_archive_search
# import pandas as pd

parser = argparse.ArgumentParser(description='collecting user timelines')

parser.add_argument(
    'indir',
    help='director above data/ that should also contain the full archive log for example'
)
parser.add_argument(
    '--collect_from',
    help='collect from this user onwards. INCLUSIVE and OVERWRITES any existing file for this user'
)

# parse
args = parser.parse_args()

# obtain credentials from file
with open('Twitter_API_credentials.json', 'r') as f:
    creds = json.loads(f.read())

bearer_token = creds['bearertoken'] 

def create_url(user_id):
    # Replace with user ID below
    return "https://api.twitter.com/2/users/{}/tweets".format(user_id)

def get_params():
    # Tweet fields are adjustable.
    # Options include:
    # attachments, author_id, context_annotations,
    # conversation_id, created_at, entities, geo, id,
    # in_reply_to_user_id, lang, non_public_metrics, organic_metrics,
    # possibly_sensitive, promoted_metrics, public_metrics, referenced_tweets,
    # source, text, and withheld
    return {
        'tweet.fields':'author_id,conversation_id,created_at,entities,geo,in_reply_to_user_id,lang,public_metrics,possibly_sensitive,referenced_tweets',
        'media.fields':'media_key,type,preview_image_url',
        'place.fields':'full_name,id,country,country_code,geo,name,place_type',
        'user.fields':'created_at,description,entities,id,location,name,pinned_tweet_id,profile_image_url,protected,public_metrics,url,username,verified',
        'max_results':100
    }

def main():

    # collect path
    DATA_PATH = os.path.join(args.indir, 'data/')

    assert os.path.isdir(DATA_PATH)

    # set up logging file
    logging.basicConfig(filename=os.path.join(args.indir, 'user_timelines.log'),
                        encoding='utf-8',
                        format='%(levelname)s:%(message)s',
                        level=logging.DEBUG)

    # print start time for records
    start_time = datetime.datetime.now()
    print(start_time)
    logging.info('User Timeline Collection Start: {}'.format(start_time))
    
    # create headers with bearer token
    headers = full_archive_search.create_headers(bearer_token)

    # collect unique users and their counts
    # users_and_counts = unique_users.get_unique_users(os.path.join(args.indir,'parsed_FAS.csv')).most_common()
    # users = [str(i[0]) for i in users_and_counts]
    file_list = glob.glob(DATA_PATH+'FAS_*.json')
    users=[]
    for each_file in tqdm.tqdm(file_list):
        with open(each_file, 'r') as f:
            x = json.load(f)
            for tweet_obj in x:
                users.append(tweet_obj['author_id'])

    if args.collect_from != None:
        assert str(args.collect_from) in users
        users = users[users.index(args.collect_from):]

    for user_id in users:

        user_url = create_url(user_id)
        logging.info('COLLECTING USER {}'.format(user_id))
        query_params = get_params()

        # get first response
        count = 1
        file_to_save = []
        json_response = full_archive_search.connect_to_endpoint(user_url, headers, query_params)

        while 'next_token' in json_response['meta']:
            time.sleep(0.6)
            pagination_params = query_params
            # pagination token is the equivalent of next_token for full archive search.
            pagination_params['pagination_token'] = json_response['meta']['next_token']
            json_response = full_archive_search.connect_to_endpoint(user_url, headers, pagination_params)
            if 'data' not in json_response:
                logging.info(json_response)
                continue
            count += 1
            file_to_save.extend(json_response['data'])
            if count % 1000 == 0:
                print(count)
        
        save_filename = DATA_PATH + 'USER_' + str(user_id) + '.json'
        with open(save_filename, 'w') as f:
            json.dump(file_to_save, f)
    
    # data collection end time
    end_time = datetime.datetime.now()
    print(end_time)
    logging.info('End Time: {}'.format(end_time))
    logging.info('Time Elapsed: {}'.format(end_time-start_time))

if __name__ == "__main__":
    main()
