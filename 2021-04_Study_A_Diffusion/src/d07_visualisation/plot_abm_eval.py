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

def score(result_np_array, key_order, params, reference_values, kind = 'percentage', avg=False):


    '''
    Function: process an np array of consolidated results to a score for each parameter combination. This can be by the average results of each combination, or the best one.
    '''

    # TO CHANGE LATER: REMOVE 0 VALUES
    reference_df = reference_values[reference_values['actual']>0]

    # sort refernece df
    reference_df = reference_df.sort_values(by='actual', ascending=False)

    # np.unique returns the unique values in an array and can return their indices to reconstruct the original array.
    unq, indices= np.unique(params, axis=0, return_inverse=True)

    # indices of hashtags needed for this group
    x = np.array([np.where(np.array(key_order)==x)[0][0] for x in reference_df['index']])
    # print(reference_df['index'])
    # print(x)

    # use these unique incides to extract the desired processed result
    # result_np_array_processed = np.zeros(shape = (len(unq), len(x)))

    # filter down necessary results to just those found in the line before
    result_np_array_processed = result_np_array[:,x]
    # print(result_np_array_processed.shape)
    # print(reference_df['actual'].shape)
    # print((result_np_array_processed-np.array(reference_df['actual'])).shape)

    # create output array
    evaluated = np.zeros(len(unq))

    if avg:

        if kind == 'percentage':

            # calc percentages
            result_np_array_processed = np.abs((result_np_array_processed-np.array(reference_df['actual']))/np.array(reference_df['actual']))

            # assign appropriate value in output array
            for i in range(len(unq)):
                evaluated[i] = result_np_array_processed[indices==i].sum().mean()

    else:

        if kind == 'percentage':

            # calc percentages: percentage difference between the actual result and the abm predictions at the end of the process.
            result_np_array_processed = np.abs((result_np_array_processed-np.array(reference_df['actual']))/np.array(reference_df['actual']))

            # assign appropriate value in output array
            for i in range(len(unq)):
                evaluated[i] = result_np_array_processed[indices==i].sum().min()

        elif kind == 'rank':

            # assign score to each rank depending on its distance with actual ranking. Ideal (i.e. matching all ranks) is a 0 score.
            rank_array_sorted = np.argsort(result_np_array_processed, axis=1)[:,::-1]
            rank_array_temp = (np.abs(rank_array_sorted - np.arange(len(x)))).sum(axis=1)
            # print(np.argsort(result_np_array_processed, axis=1).shape)
            # print(rank_array_temp.shape)

            for i in range(len(unq)):
                evaluated[i] = rank_array_temp[indices==i].min()

        elif kind == 'top_rank':

            # assign score to each rank depending on its distance with actual ranking. Only keep top rank.
            rank_array_sorted = np.argsort(result_np_array_processed, axis=1)[:,::-1]
            rank_array_top_check = rank_array_sorted[:,0]==0

            # result is the number of times out of 10 repetitions the top rank is correct
            for i in range(len(unq)):
                evaluated[i] = rank_array_top_check[indices==i].sum()

        elif kind == 'by_far':

            # like top rank, check if English #MeToo is the top
            # assign score to each rank depending on its distance with actual ranking. Only keep top rank.
            rank_array_sorted = np.argsort(result_np_array_processed, axis=1)[:,::-1]
            rank_array_top_check = rank_array_sorted[:,0]==0

            # reference threshold
            ref_percent = (reference_df['actual'].iloc[0]-reference_df['actual'].iloc[0])/reference_df['actual'].iloc[1]

            # get data percentage
            top_by_far_check = (result_np_array_processed[:,0]-result_np_array_processed[:,1])/result_np_array_processed[:,1]

            # compare
            compared_top_by_far = (ref_percent - top_by_far_check)>0
            for i in range(len(unq)):
                evaluated[i] = compared_top_by_far[indices==i].sum()


    return reference_df['index'], evaluated, unq, indices


def extract_directions(graph):
    output = nx.DiGraph()
    for i in tqdm.tqdm(graph.edges(data=True)):
        subject_id = graph.nodes(data=True)[i[0]]['primary_ht']
        object_id  = graph.nodes(data=True)[i[1]]['primary_ht']
        if output.has_edge(subject_id, object_id):
            # we added this one before, just increase the weight by one
            output[subject_id][object_id]['weight'] += 1
        else:
            # new edge. add with weight=1
            output.add_edge(subject_id, object_id, weight=1)
        # output.add_edge(G.nodes(data=True)[i[0]]['primary_ht'], G.nodes(data=True)[i[1]]['primary_ht'])
    return output

def main(args):

    # if the clear flag is up clear all the graphs
    if args.clear:
        abm_to_remove = glob.glob(f'/home/hubert/DPhil_Studies/2021-04_Study_A_Diffusion/data/06_reporting/0{args.group_num}_group/abm*.png')
        for abm_plot in abm_to_remove:
            try:
                os.remove(abm_plot)
            except OSError:
                print(f'Error while deleting {abm_plot}')

    ############################################################################
    # Initial opening of files and collecting relevant information like reference results
    ############################################################################

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
        param_order = f['params'].attrs['param_order'].strip('[]').replace("'",'').split(', ')

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

    ############################################################################
    # Process history evaluation - THEORETICAL Tests
    ############################################################################


    # load in history evaluation objects:
    with open(f'/home/hubert/DPhil_Studies/2021-04_Study_A_Diffusion/data/06_reporting/ABM_history_eval_group_{group_num}.obj', 'rb') as f:
        history_eval_obj = pickle.load(f)

    # print(len(history_eval_obj))

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
    # print(len(history_eval_df))

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

    ############################################################################
    # Process DATA evaluation tasks
    ############################################################################

    print('Processing Data Evaluation')

    # reference_values is 
    act = (reference_values.iloc[:,1:]>2).sum(axis=0).to_frame().reset_index()
    act.columns = ['index', 'actual']

    # calculate results
    top_rank_eval   = score(results, key_order, params, act, kind='top_rank')
    top_by_far_eval = score(results, key_order, params, act, kind='by_far')
    rank_eval       = score(results, key_order, params, act, kind='rank')
    percent_eval    = score(results, key_order, params, act, kind='percentage')

    # create dataplot df
    unq_params        = rank_eval[2] # could equally have used the other ranks
    dataplot_np = np.concatenate(
        (unq_params, top_rank_eval[1].reshape(-1,1), top_by_far_eval[1].reshape(-1,1), rank_eval[1].reshape(-1,1), percent_eval[1].reshape(-1,1)),
        axis=1
    )

    # dataplot colnames
    eval_colnames = ['top_rank_eval', 'top_by_far_eval', 'rank_eval', 'percent_eval']
    colnames = param_order + eval_colnames

    dataplot_df = pd.DataFrame(
        data=dataplot_np,
        index=None,
        columns = colnames
    )

    # Test 1: how many combinations have top rank as correct?
    plt.figure(figsize=(15,8))
    sns.boxplot(
        x='top_rank_eval',
        data=dataplot_df
    )
    plt.savefig(f'/home/hubert/DPhil_Studies/2021-04_Study_A_Diffusion/results/0{args.group_num}_group/abm_data_eval_boxplot.png', bbox_inches='tight')
    plt.close() 

    ############################################################################
    # Metric table output
    ############################################################################

    # convert to LaTeX?


    ############################################################################
    # Influence matrix results
    ############################################################################

    # INSERT GRAPH CODE HERE
    print('Processing Graph')

    d_graph = extract_directions(G)
    d_graph.remove_edges_from(nx.selfloop_edges(d_graph))

    degree_dict = dict(d_graph.out_degree)

    # fixing the size of the figure
    plt.figure(figsize =(15, 10))

    node_color = [d_graph.out_degree(v) for v in d_graph]
    # node colour is a list of degrees of nodes

    # node_size = [0.0005 * nx.get_node_attributes(d_graph, 'population')[v] for v in d_graph]
    # size of node is a list of population of cities

    edge_width = [0.0005 * d_graph[u][v]['weight'] for u, v in d_graph.edges()]
    # width of edge is a list of weight of edges

    nx.draw_networkx(d_graph,
                    pos = nx.spring_layout(d_graph, k=20),
                    node_color = node_color,
                    with_labels = True,
                    #  width = edge_width,
                    node_size = [v * 500 for v in degree_dict.values()],
                    cmap = plt.cm.Blues,
                    font_size = 20)

    plt.axis('off')
    plt.tight_layout();
    plt.savefig(f'/home/hubert/DPhil_Studies/2021-04_Study_A_Diffusion/results/0{args.group_num}_group/abm_graph_original.png', transparent=True, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument(
        'group_num',
        help='Group number'
    )

    parser.add_argument(
        '--clear',
        help='Clear previous graphs with abm prefix',
        default=False,
        action='store_true'
    )

    args = parser.parse_args()

    main(args)