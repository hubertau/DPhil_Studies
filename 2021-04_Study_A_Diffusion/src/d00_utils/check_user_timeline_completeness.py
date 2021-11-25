from bispec_clustering_eval import BSCresults
from timeline_analysis import TimelineAnalyzer
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
from collections import defaultdict
import networkx as nx

# read in selected date ranges.
with open('/home/hubert/DPhil_Studies/2021-04_Study_A_Diffusion/collection_results_2021_06_19_16_21/selected_date_ranges.obj', 'rb') as f:
    selected_date_ranges = pickle.load(f)

# identify which date range to consider:
date_range = 0


timeline_flist = glob.glob('/home/hubert/DPhil_Studies/2021-04_Study_A_Diffusion/collection_results_2021_06_19_16_21/data/timeline*.jsonl')

max_date_per_user = []

for twitter_user_timeline in tqdm.tqdm(timeline_flist):
    current_max_date=selected_date_ranges[0][0]
    with jsonlines.open(twitter_user_timeline) as reader:
        for tweet_jsonl in reader:
            tweet_list_in_file = tweet_jsonl['data']
            for tweet_data in tweet_list_in_file:
                tweet_created_at = datetime.datetime.fromisoformat(tweet_data['created_at'][:-1])
                # print(tweet_created_at)
                if tweet_created_at > current_max_date:
                    current_max_date = tweet_created_at
    max_date_per_user.append((os.path.split(twitter_user_timeline)[1],current_max_date))

with open('/home/hubert/DPhil_Studies/2021-04_Study_A_Diffusion/collection_results_2021_06_19_16_21/timeline_completeness.obj', 'wb') as f:
    pickle.dump(max_date_per_user,f)
