from bispec_clustering_eval import BSCresults
from timeline_analysis import TimelineAnalyzer
import numpy as np
import plotnine
import os
import pandas as pd
import datetime
from itertools import repeat
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
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor


def main():

    user_edges_pickle_file = '/home/hubert/DPhil_Studies/2021-04_Study_A_Diffusion/collection_results_2021_06_19_16_21/user_sets.obj'

    if os.path.isfile(user_edges_pickle_file):
        with open(user_edges_pickle_file, 'rb') as f:
            user_edges = pickle.load(f)

    assert type(user_edges) == defaultdict

    with open('/home/hubert/DPhil_Studies/2021-04_Study_A_Diffusion/collection_results_2021_06_19_16_21/user_list_full_0_nocount.txt', 'r') as f:
        p = f.readlines()
        p = [int(i.replace('\n','')) for i in p]
        user_list = p

    with ThreadPoolExecutor(max_workers=5) as executor:
        results = executor.map(task, list(user_edges.keys()), repeat(user_list), repeat(user_edges))

    # get list object from results:
    results = list(results)

    with open('/home/hubert/DPhil_Studies/2021-04_Study_A_Diffusion/collection_results_2021_06_19_16_21/user_sets_sanity_check.obj', 'wb') as f:
        pickle.dump(results, f)

def task(x, user_list, user_edges):
    print('begin processing index {}, user {} at {}'.format(list(user_edges.keys()).index(x), x, datetime.datetime.now()))
    if int(x) not in user_list:
        key_check = x
    else:
        key_check = None
    val_check = []
    for val in user_edges[x]:
        if int(val) not in user_list:
            val_check.append(val)
    print('end processing {} at {}'.format(x, datetime.datetime.now()))
    return (key_check, val_check)

if __name__ == '__main__':

    main()