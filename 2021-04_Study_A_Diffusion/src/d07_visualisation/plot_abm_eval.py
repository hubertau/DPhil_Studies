import seaborn as sns
sns.set('talk')
sns.set_style('ticks')

import argparse
import glob
import numpy as np
import networkx as nx
import datetime
from itertools import count
import tqdm
from typing import NamedTuple
import h5py

import re
import pandas as pd
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
pd.set_option('precision', 2)

import sys

sys.path.append( '/home/hubert/DPhil_Studies/2021-04_Study_A_Diffusion/src/d04_modelling' )

from abm import *
from abm_eval import *
from abm_history_eval import *

#load in search hashtags
with open('../../references/search_hashtags.txt', 'r') as f:
    search_hashtags = f.readlines()
    search_hashtags = [i.replace('\n', '') for i in search_hashtags]
    search_hashtags = [i.replace('#', '') for i in search_hashtags]
    search_hashtags = [i.lower() for i in search_hashtags]
    search_hashtags.remove('وأناكمان')


def group_peaks_and_daterange(peak_analysis_file, group_num):

    #obtain peak times again
    with h5py.File(peak_analysis_file, 'r') as f:
        FAS_peaks = f['peak_detections']
        x = f['segments']['selected_ranges'][int(group_num)-1]
        group_date_range = daterange(
            start = x[0].decode(),
            end = x[1].decode()
        )

        # group_start_index = reverse_unit_conv(group_date_range.start)
        # group_end_index = reverse_unit_conv(group_date_range.end)

        most_prominent_peaks = {}
        for name, h5obj in FAS_peaks.items():

            peak_locations = h5obj['peak_locations']
            peak_locations = [(i,e) for i,e in enumerate(h5obj['peak_locations']) if (unit_conv(e) > datetime.datetime.strptime(group_date_range.start, '%Y-%m-%d')) and (unit_conv(e) < datetime.datetime.strptime(group_date_range.end, '%Y-%m-%d'))]
            peak_indices = [i[0] for i in peak_locations]
            prominences = [element for index, element in enumerate(h5obj['prominences']) if index in peak_indices]
            if len(prominences) == 0:
                continue
            max_prominence = np.argmax(prominences)
            most_prominent_peaks[name] = unit_conv(peak_locations[max_prominence][1])

    daterange_length = (datetime.datetime.strptime(group_date_range.end, '%Y-%m-%d') - datetime.datetime.strptime(group_date_range.start, '%Y-%m-%d')).days

    return most_prominent_peaks, group_date_range, daterange_length

def reference_results(most_prominent_peaks, group_date_range, group_num, agent_list, activity_file = '/home/hubert/DPhil_Studies/2021-04_Study_A_Diffusion/data/03_processed/activity_counts.hdf5'):
    act_val = {}
    with h5py.File(activity_file, 'r') as f:
        activity_base = f[f'group_{group_num}']
        feature_order = f[f'group_{group_num}'][agent_list[0]]['hashtagged'].attrs['feature_order']
        feature_order = feature_order.split(';')

        for user_id in agent_list:
            # obtain user activity
            act_val[user_id] = {}
            activity = activity_base[user_id]['hashtagged'][:]

            # act_val[user_id] = np.sum(activity[:,-int(daterange_length/2):])

            for hashtag_in_period in most_prominent_peaks:
                hashtag_in_period_index = feature_order.index(hashtag_in_period)

                # obtain the index offset from the detected peak of the hashtag to collect initial time window.
                peak_index_index = (datetime.datetime.strptime(group_date_range.end, '%Y-%m-%d')-most_prominent_peaks[hashtag_in_period]).days
                # offset_index -= peak_delta_init
                # offset_index = max(0,offset_index)+1
                # print(f'Offset for {hashtag_in_period} is {offset_index}')

                act_val[user_id][hashtag_in_period_index]= np.sum(activity[hashtag_in_period_index,-peak_index_index-1:])

    act_val = pd.DataFrame.from_dict(act_val, orient='index').reset_index()
    act_val.columns = ['user_id'] + list(most_prominent_peaks.keys())

    return act_val

def main(args):

    # set group number
    group_num = args.group_num
    abm_results_path = '/home/hubert/DPhil_Studies/2021-04_Study_A_Diffusion/data/06_reporting'

    # collect group information
    most_prominent_peaks, group_date_range, daterange_length = group_peaks_and_daterange('/home/hubert/DPhil_Studies/2021-04_Study_A_Diffusion/data/02_intermediate/FAS_peak_analysis.hdf5', group_num)

    # compressed hdf5 file consolidation:
    hdf5_consolidated_file = os.path.join(abm_results_path, f'ABM_output_consolidated_group_{group_num}.hdf5')

    # extract params and results and key order, converting it back into a list
    with h5py.File(hdf5_consolidated_file, 'r') as f:
        results = f['result'][:]
        key_order = f['result'].attrs['key_order'].strip('[]').replace("'",'').split(', ')
        params = f['params'][:]

    # dummy file ref. This is just for collecting agent order.
    hdf5_res_file_dummy = os.path.join(abm_results_path, f'0{group_num}_group/ABM_output_group_{group_num}_batch_0.hdf5')

    # collect agent order
    with h5py.File(hdf5_res_file_dummy, 'r') as f:
        agents = f['agent_order'][:]

    # extract reference values
    reference_values = reference_results(
        most_prominent_peaks,
        group_date_range,
        group_num,
        agents
    )

    # for graph visualisations later, load in respective graph object.
    graph_savepath = os.path.join(abm_results_path, f'ABM_graph_group_{group_num}.obj')

    if os.path.isfile(graph_savepath):
        with open(graph_savepath, 'rb') as f:
            G = pickle.load(f)

    # load in history evaluation objects:
    with open(f'/home/hubert/DPhil_Studies/2021-04_Study_A_Diffusion/data/06_reporting/ABM_history_eval_group_{group_num}.obj', 'rb') as f:
        history_eval_obj = pickle.load(f)

    print(len(history_eval_obj))

    # flatten history_eval_list
    history_eval_flattened = [item for sublist in history_eval_obj for item in sublist]

    def convert_item(item_tuple):
        output_dict = item_tuple[0]
        output_dict['awareness_count'] = item_tuple[1].awareness_count
        output_dict['awareness_total'] = item_tuple[1].awareness_total
        output_dict['experimentation_count'] = item_tuple[1].experimentation_count
        output_dict['application_count'] = item_tuple[1].application_count

        return output_dict

    # collect into list to create into DF
    history_eval_collated_list = [convert_item(i) for i in history_eval_flattened]

    # convert into DF
    history_eval_df = pd.DataFrame.from_dict(history_eval_collated_list)
    print(len(history_eval_df))

    param_names = ['experimentation_chance',
        'initial_activity_threshold',
        'interact_prob',
        'interact_prob_multiplier',
        'interact_threshold',
        'model_num',
        'peak_delta_init',
        'search_hashtag_propensity']

    measures = ['awareness_count','awareness_total','experimentation_count', 'application_count']

    final_df = history_eval_df.groupby(param_names).max().reset_index()

    # plot variation of each output measure by param:
    for param_name in param_names:
        for measure in measures:
            print(f'Plotting {measure} vs. {param_name}')
            plt.figure(figsize=(15,8))
            sns.scatterplot(x=param_name,y=measure,data=final_df)
            plt.savefig(f'/home/hubert/DPhil_Studies/2021-04_Study_A_Diffusion/results/0{args.group_num}_group/abm_{param_name}+{measure}.png', bbox_inches='tight')
            plt.close()

    # Search hashtag propensity is obviously crap.
    # Turns out peak delta init has no effect on the measures
    # Turns out the same for initial activity threshold
    # model num can be split into separate graphs.
    # so need to represent the other 4:
    # x = interact_threshold
    # y = experimentation_chance
    # colour = interact_prob
    # size = output measure
    # marker type = interact_prob_multiplier

    # plot more sophisticated
    for model_num in final_df['model_num'].unique():
        for measure in measures:
            print(f'Plotting COMPOSITE {measure} for model number {model_num}')
            plt.figure(figsize=(15,8))
            sns.scatterplot(
                x='interact_threshold',
                y='experimentation_chance',
                hue='interact_prob',
                style='interact_prob_multiplier',
                size=measure,
                data=final_df
            )
            plt.savefig(f'/home/hubert/DPhil_Studies/2021-04_Study_A_Diffusion/results/0{args.group_num}_group/abm_composite_{measure}_model_{model_num}.png', bbox_inches='tight')
            plt.close()

    # plot just the histogram of the measure:
    for measure in measures:
        plt.figure(figsize=(15,8))
        sns.histplot(
            x=measure,
            data=final_df
        )
        plt.savefig(f'/home/hubert/DPhil_Studies/2021-04_Study_A_Diffusion/results/0{args.group_num}_group/abm_histogram_{measure}.png', bbox_inches='tight')
        plt.close()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument(
        'group_num',
        help='Group number'
    )

    args = parser.parse_args()

    main(args)