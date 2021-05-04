import requests
import json
import time
import csv
import datetime
import logging
import os

# obtain credentials from file
with open('Twitter_API_credentials.json', 'r') as f:
    creds = json.loads(f.read())

bearer_token = creds['bearertoken'] 

# obtain search terms
with open('search_hashtags.csv', newline='') as f:
    terms = list(csv.reader(f))

# unroll list
terms = [i[0] for i in terms]
search_query = ' OR '.join(terms)

search_url = "https://api.twitter.com/2/tweets/search/all"

# Optional params: start_time,end_time,since_id,until_id,max_results,next_token,
# expansions,tweet.fields,media.fields,poll.fields,place.fields,user.fields
query_params = {
                    'query': search_query,
                    'tweet.fields': 'author_id,created_at',
                    'start_time': '2017-10-17T00:00:00Z',
                    'end_time': '2017-12-31T23:59:59Z',
                    'max_results':100
               }


def create_headers(bearer_token):
    headers = {"Authorization": "Bearer {}".format(bearer_token)}
    return headers


def connect_to_endpoint(url, headers, params):
    response = requests.request("GET", url, headers=headers, params=params)
    #print(response.status_code)
    if response.status_code == 429:
        try_count = 0
        while try_count < 10 and response.status_code == 429: 
            time.sleep(3)
            response = requests.request("GET", url, headers=headers, params=params) 
            try_count += 1
            if response.status_code == 200:
                continue
        logging.info('\nResult after some retries of 429 response code')
        logging.info('response code: {}'.format(response.status_code))
        logging.info('search url: {}'.format(url))
        logging.info('headers: {}'.format(headers))
        logging.info('query parameters: {}\n'.format(params))
    elif response.status_code != 200:
        logging.info('\nresponse code: {}'.format(response.status_code))
        logging.info('search url: {}'.format(url))
        logging.info('headers: {}'.format(headers))
        logging.info('query parameters: {}\n'.format(params))
    return response.json()

def try_field(tweet_result_obj, field_name):
    try:
        return tweet_result_obj[field_name]
    except:
        return 'NA'

def main():

    # print start time for records
    start_time = datetime.datetime.now()
    print(start_time)
    logging.info('Full Archive Search Collection Start: {}'.format(start_time))

    # obtain current run time for reuslts
    CURRENT_RUN_TIME = datetime.datetime.today()
    CURRENT_RUN_TIME = CURRENT_RUN_TIME.strftime("%Y_%m_%d_%H_%M")

    # creating new path
    OUTPUT_PATH = os.path.join('collection_results_' + CURRENT_RUN_TIME)

    # set up logging file
    logging.basicConfig(filename=os.path.join(OUTPUT_PATH, 'full_archive_search.log'),
                        format='%(levelname)s:%(message)s',
                        level=logging.DEBUG)

    # create results file
    results_path = os.path.join(OUTPUT_PATH, 'results.csv')
    results_file = open(results_path, 'w')
    results_writer = csv.writer(results_file)

    # create headers with bearer token
    headers = create_headers(bearer_token)

    # get first response
    count = 1
    json_response = connect_to_endpoint(search_url, headers, query_params)

    # store results in csv format
    results_header = ('author_id', 'text', 'created_at', 'id')
    results_writer.writerow(results_header)
    for tweet_result in json_response['data']:
        result_row = (
                try_field(tweet_result,'author_id'),
                try_field(tweet_result,'text'),
                try_field(tweet_result,'created_at'),
                try_field(tweet_result,'id')
            )       
        results_writer.writerow(result_row)

    while 'next_token' in json_response['meta']:
        time.sleep(3)
        pagination_params = query_params
        pagination_params['next_token'] = json_response['meta']['next_token']
        logging.info(json_response['meta']['next_token'])
        json_response = connect_to_endpoint(search_url, headers, pagination_params)
        if 'data' not in json_response:
            logging.info(json_response)
            continue
        for tweet_result in json_response['data']:
            result_row = (
                try_field(tweet_result,'author_id'),
                try_field(tweet_result,'text'),
                try_field(tweet_result,'created_at'),
                try_field(tweet_result,'id')
            )
            results_writer.writerow(result_row)
        count += 1
        if count % 100 == 0:
            print(count)

    # close writing to results
    results_file.close()

    # data collection end time
    end_time = datetime.datetime.now()
    print(end_time)
    logging.info('End Time: {}'.foramt(end_time))
    logging.info('Time Elapsed: {}'.format(end_time-start_time))

if __name__ == "__main__":
    main()