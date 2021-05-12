# quick and dirty script to find first use of a hashtag

import json
import glob
import datetime
import tqdm
import csv

hashtags = [
    'BalanceTonPorc',
    'MoiAussi'
]

hashtags = [i.lower() for i in hashtags]

data_dir = './collection_results_2021_05_04_16_22/data/'

file_list = glob.glob(data_dir+'FAS*.json')

earliest = [datetime.datetime.today()] * len(hashtags)

times = [[]] * len(hashtags)

tweet_counter=0
for file_name in tqdm.tqdm(file_list):
    with open(file_name) as f:
        f = json.load(f)
        for tweet in f:
            tweet_counter += 1
            if 'entities' in tweet:
                if 'hashtags' in tweet['entities']:
                    hts = [i['tag'].lower() for i in tweet['entities']['hashtags']]
                    for index, h in enumerate(hashtags):
                        if h in hts:
                            if datetime.datetime.fromisoformat(tweet['created_at'][:-1]) < earliest[index]:
                                earliest[index] = datetime.datetime.fromisoformat(tweet['created_at'][:-1])
                            times[index].append(datetime.datetime.fromisoformat(tweet['created_at'][:-1]))

print(earliest)
print('Total tweets processed: {}'.format(tweet_counter))

times[0].sort()
times[1].sort()

with open('tweet_times_moiaussi.csv', 'w') as f:
    writer = csv.writer(f)
    for i in times[0]:
        writer.writerow([datetime.datetime.isoformat(i)])

with open('tweet_times_balancetonporc.csv', 'w') as f:
    writer = csv.writer(f)
    for i in times[1]:
        writer.writerow([datetime.datetime.isoformat(i)])
