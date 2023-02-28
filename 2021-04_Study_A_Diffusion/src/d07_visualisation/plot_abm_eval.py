import seaborn as sns
sns.set('talk')
sns.set_style('white')
sns.despine(offset=10,trim=True)

import argparse
import glob
import numpy as np
import networkx as nx
import datetime
from itertools import count
from collections import Counter
import tqdm
from typing import NamedTuple
import h5py
from scipy.stats import spearmanr

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

            rank_up_to = 5 # actual rank to go up to

            # assign score to each rank depending on its distance with actual ranking. Ideal (i.e. matching all ranks) is a 0 score.
            rank_array_sorted = np.argsort(result_np_array_processed, axis=1)[:,::-1]

            if rank_array_sorted.shape[1] <= rank_up_to-1:
                rank_up_to = rank_array_sorted.shape[1]

            # rank_array_temp = (np.abs(rank_array_sorted - np.arange(len(x)))).sum(axis=1)
            rank_array_temp = np.zeros(rank_array_sorted.shape[0])
            for i in tqdm.tqdm(range(rank_array_sorted.shape[0])):
                rank_array_temp[i] = spearmanr(rank_array_sorted[i,1:rank_up_to], np.arange(1,rank_up_to))[0]
            # rank_array_temp, _ = spearmanr(rank_array_sorted[:,1:5], np.tile(np.array([1,2,3,4]), (rank_array_sorted.shape[0], 1)))

            # print(np.argsort(result_np_array_processed, axis=1).shape)
            # print(rank_array_temp.shape)

            for i in range(len(unq)):
                # N.B. .max() method should be used 
                evaluated[i] = rank_array_temp[indices==i].max()

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
            ref_percent = (reference_df['actual'].iloc[0]-reference_df['actual'].iloc[1])/reference_df['actual'].iloc[1]

            # get data percentage
            top_by_far_check = (result_np_array_processed[:,0]-result_np_array_processed[:,1])/result_np_array_processed[:,1]

            # compare
            compared_top_by_far = ref_percent - top_by_far_check
            for i in range(len(unq)):
                evaluated[i] = compared_top_by_far[indices==i].max()

        elif kind == 'by_far_ref':
            rank_array_sorted = np.argsort(result_np_array_processed, axis=1)[:,::-1]
            rank_array_top_check = rank_array_sorted[:,0]==0
            ref_percent = (reference_df['actual'].iloc[0]-reference_df['actual'].iloc[1])/reference_df['actual'].iloc[1]
            top_by_far_check = (result_np_array_processed[:,0]-result_np_array_processed[:,1])/result_np_array_processed[:,1]

            for i in range(len(unq)):
                evaluated[i] = top_by_far_check[indices==i].min()
            return ref_percent, top_by_far_check, evaluated, unq, indices


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
        print('Clearing previous abm plots and data...')
        abm_to_remove = glob.glob(f'/home/hubert/DPhil_Studies/2021-04_Study_A_Diffusion/data/06_reporting/0{args.group_num}_group/abm*.*')
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
    param_labels = {
        'experimentation_chance'    : 'Experimentation Chance',
        'initial_activity_threshold': 'Initial Activity Threshold',
        'interact_prob'             : 'Interaction Probability',
        'interact_prob_multiplier'  : 'Interaction Probability Multiplier',
        'interact_threshold'        : 'Interaction Threshold',
        'model_num'                 : 'Model Number',
        'peak_delta_init'           : 'Days Before Peak',
        'search_hashtag_propensity' : 'Propensity'
    }

    measures = ['awareness_count','awareness_total','experimentation_count', 'application_count']
    measure_labels = {
        'awareness_count': 'Awareness',
        'awareness_total': 'Potential Awareness',
        'experimentation_count': 'Experimentation',
        'application_count': 'Movement Application'
    }

    final_df = history_eval_df.groupby(param_names).max().reset_index()

    # plot variation of each output measure by param:
    for param_name in param_names:
        for measure in measures:
            print(f'Plotting {measure} vs. {param_name}')
            plt.figure(figsize=(15,8))
            ax = sns.boxplot(x=param_name,y=measure,data=final_df)
            ax.set_ylabel(f'{measure_labels[measure]}')
            ax.set_xlabel(f'{param_labels[param_name]}')
            if measure=='experimentation_count':
                ax.set_yscale('log')
                ax.set_ylabel(f'{measure_labels[measure]} (log scale)')
            plt.savefig(f'/home/hubert/DPhil_Studies/2021-04_Study_A_Diffusion/results/0{args.group_num}_group/abm_{param_name}+{measure}.png', bbox_inches='tight', dpi=300)
            plt.close()

    # Plot grouped boxplot for awareness and interaction probability
    # plt.figure(figsize=(15,8))
    # ax = sns.boxplot(
    #     x=''
    # )

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
            plt.savefig(f'/home/hubert/DPhil_Studies/2021-04_Study_A_Diffusion/results/0{args.group_num}_group/abm_composite_{measure}_model_{model_num}.png', bbox_inches='tight', dpi=300)
            plt.close()

    # plot just the histogram of the measure:
    print('Plotting histograms for the measures themselves ')
    for measure in measures:
        plt.figure(figsize=(15,8))
        sns.histplot(
            x=measure,
            data=final_df
        )
        plt.savefig(f'/home/hubert/DPhil_Studies/2021-04_Study_A_Diffusion/results/0{args.group_num}_group/abm_histogram_{measure}.png', bbox_inches='tight', dpi=300)
        plt.close()

    print('Plotting boxplots for the measures themselves ')
    # plot maybe boxplots is better?
    for measure in measures:
        plt.figure(figsize=(15,8))
        if measure=='application_count':
            ax = sns.boxplot(
                x=measure,
                data=final_df[final_df['model_num']==6]
            )
        else:
            ax = sns.boxplot(
                x=measure,
                data=final_df
            )
        ax.set_xlabel(f'{measure_labels[measure]}')
        plt.savefig(f'/home/hubert/DPhil_Studies/2021-04_Study_A_Diffusion/results/0{args.group_num}_group/abm_boxplot_{measure}.png', bbox_inches='tight', dpi=300)
        plt.close()

    ## SAVE FINALDF FOR COMBINED PLOTS
    print('Saving theory test df')
    final_df_savepath = f'/home/hubert/DPhil_Studies/2021-04_Study_A_Diffusion/results/0{args.group_num}_group/abm_theoretical_eval_df.obj'
    with open(final_df_savepath, 'wb') as f:
        pickle.dump(final_df, f)

    all_final_df_objs = sorted(glob.glob('/home/hubert/DPhil_Studies/2021-04_Study_A_Diffusion/results/*/abm_theoretical_eval_df.obj', recursive=True))
    print(all_final_df_objs)
    all_final_df = []
    if len(all_final_df_objs) == 3:
        print('All Primary HT Dist DFs found. Combinings for ease of LaTeX...')
        for i in all_final_df_objs:
            with open(i, 'rb') as f:
                all_final_df.append(pickle.load(f))
        final_dist_df = all_final_df[0].merge(all_final_df[1], suffixes=('_1', '_2'), on = param_names)
        final_dist_df = final_dist_df.merge(all_final_df[2], on=param_names)
        for measure in measures:
            final_dist_df.loc[:,f'{measure}_3'] = final_dist_df[f'{measure}'] # didn't have suffix
        # final_dist_df.rename(columns={f'{measure}':f'{measure}_3' for measure in measures}, inplace=True)

        for measure in measures:

            measure_temp_df = final_dist_df.copy().iloc[:,final_dist_df.columns.str.startswith(f'{measure}_')]
            measure_temp_df.loc[:,'id'] = final_dist_df.index
            measure_temp_df = pd.wide_to_long(
                measure_temp_df,
                f'{measure}',
                sep="_",
                i='id',
                j='Period').reset_index()
            plt.figure(figsize=(15,8))
            ax = sns.boxplot(
                x='Period',
                y=f'{measure}',
                data=measure_temp_df
                )
            ax.set_ylabel(f'{measure_labels[measure]}')
            plt.savefig(f'/home/hubert/DPhil_Studies/2021-04_Study_A_Diffusion/results/abm_theory_combined_{measure}.png', bbox_inches='tight', dpi=300)
            plt.close()

    # Make grouped boxplot with y axis awareness count, period on x axis with hue as interaction probability
    grouped_df = pd.melt(
        final_dist_df,
        id_vars=['interact_prob'],
        value_vars=['awareness_count','awareness_count_2','awareness_count_3']
    )
    grouped_df.rename(columns = param_labels, inplace=True)
    plt.figure(figsize=(15,8))
    ax = sns.boxplot(x='variable', y='value', hue='Interaction Probability', data=grouped_df)
    ax.set_ylabel(f'Awareness Count')
    ax.set_xlabel('Period')
    ax.set_xticklabels([1,2,3])
    plt.savefig(f'/home/hubert/DPhil_Studies/2021-04_Study_A_Diffusion/results/abm_grouped_boxplot.png', bbox_inches='tight', dpi=300)

    # Heatmap for medians of movement app awareness and theoretical measures
    median_df_dict = [
        {
            'Period': i,
            'Awareness Count': np.median(final_dist_df[f'awareness_count_{i}']),
            'Experimentation Count': np.median(final_dist_df[f'experimentation_count_{i}']),
            'Movement Application': np.median(final_dist_df[final_dist_df['model_num']==6][f'application_count_{i}']),
        } for i in [1,2,3]
    ]
    median_df = pd.DataFrame.from_records(median_df_dict)
    median_df.set_index('Period', inplace=True)
    plt.figure(figsize=(15,8))
    ax = sns.heatmap(median_df, annot=True, fmt=".1f")
    plt.savefig(f'/home/hubert/DPhil_Studies/2021-04_Study_A_Diffusion/results/abm_theory_heatmap.png', bbox_inches='tight', dpi=300)

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
    by_far_ref      = score(results, key_order, params, act, kind='by_far_ref')

    # create dataplot df
    unq_params        = rank_eval[2] # could equally have used the other ranks
    dataplot_np = np.concatenate(
        (unq_params, top_rank_eval[1].reshape(-1,1), top_by_far_eval[1].reshape(-1,1), rank_eval[1].reshape(-1,1), percent_eval[1].reshape(-1,1), by_far_ref[2].reshape(-1,1)),
        axis=1
    )

    # dataplot colnames
    eval_colnames = ['top_rank_eval', 'top_by_far_eval', 'rank_eval', 'percent_eval', 'sim_percent']
    eval_labels = {
        'top_rank_eval'  : 'Largest Protest Network is #MeToo'     ,
        'top_by_far_eval': 'Difference'     ,
        'rank_eval'      : 'Rank correlation'     ,
        'percent_eval'   : 'Percentage Proximity',
        'sim_percent'    : 'Simulated Percentage Difference'
    }
    colnames = param_order + eval_colnames

    dataplot_df = pd.DataFrame(
        data=dataplot_np,
        index=None,
        columns = colnames
    )

    # Test 1: how many combinations have top rank as correct?
    for data_eval in eval_colnames:
        print(f'Plotting DATA EVAL {data_eval}')
        plt.figure(figsize=(15,8))
        if data_eval == 'rank_eval':
            ax = sns.histplot(
                x=data_eval,
                data=dataplot_df
            )
        else:
            ax = sns.boxplot(
                x=data_eval,
                data=dataplot_df
            )
        ax.set_xlabel(f'{eval_labels[data_eval]}')
        plt.savefig(f'/home/hubert/DPhil_Studies/2021-04_Study_A_Diffusion/results/0{args.group_num}_group/abm_data_eval_boxplot_{data_eval}.png', bbox_inches='tight', dpi=300)
        plt.close()

    # PLOT HISTPLOT WITH EXTRA
    plt.figure(figsize=(15,8))
    temp_dat = dataplot_df.rename(columns={'interact_prob':'Interaction Probability'})
    temp_dat = temp_dat.rename(columns=eval_labels)
    temp_dat['Interaction Probability'] = temp_dat['Interaction Probability'].apply(lambda x: np.round(x, 1))
    ax = sns.histplot(
        x='Rank correlation',
        hue='Interaction Probability',
        data=temp_dat,
        palette='rocket'
    )
    plt.savefig(f'/home/hubert/DPhil_Studies/2021-04_Study_A_Diffusion/results/0{args.group_num}_group/abm_data_eval_boxplot_rank_eval_with_interact_prob.png', bbox_inches='tight', dpi=300)

    # save dataplot df
    print('SAVING DATA DF')
    with open(f'/home/hubert/DPhil_Studies/2021-04_Study_A_Diffusion/results/0{args.group_num}_group/abm_data_eval_df.obj', 'wb') as f:
        pickle.dump(dataplot_df, f)

    ref_percent, sim_percent, evaluated, unq, indices = score(results, key_order, params, act, kind='by_far_ref')

    plt.figure(figsize=(15,8))
    ax = sns.boxplot(
        y = sim_percent
    )
    ax.hlines(ref_percent, -1, 1)
    plt.savefig(f'/home/hubert/DPhil_Studies/2021-04_Study_A_Diffusion/results/0{args.group_num}_group/abm_data_eval_boxplot_reference.png',bbox_inches='tight', dpi=300)

    print('SAVING SIM PERCENT TOP EVAL BY FAR')
    with open(f'/home/hubert/DPhil_Studies/2021-04_Study_A_Diffusion/results/0{args.group_num}_group/abm_sim_percent.obj', 'wb') as f:
       pickle.dump((ref_percent, sim_percent), f)

    all_sim_path = sorted(glob.glob('/home/hubert/DPhil_Studies/2021-04_Study_A_Diffusion/results/*/abm_sim_percent.obj', recursive=True))
    print(all_sim_path)
    all_sim = []
    if len(all_sim_path) == 3:
        for i in all_sim_path:
            with open(i, 'rb') as f:
                x = pickle.load(f)
                all_sim.append(x)
        ref_diff = [i[0] for i in all_sim]
        all_sim = [i[1] for i in all_sim]
        # all_sim_df = pd.DataFrame.from_dict({
            # str(i): e[1] for i,e in enumerate(all_sim) 
        # }, orient='columns')
        all_sim_df = pd.DataFrame.from_dict({
            'Period':list(
                np.concatenate((
                    np.ones(len(all_sim[0])),
                    np.ones(len(all_sim[1]))+1,
                    np.ones(len(all_sim[2]))+2
                ))
            ),
            'Simulated Percent': list(np.concatenate(all_sim))},
            orient='columns'
        )
        all_sim_df.Period = all_sim_df.Period.astype(int)
        plt.figure(figsize=(15,8))
        ax = sns.boxplot(
            y='Simulated Percent',
            x='Period',
            data=all_sim_df
        )
        ax.hlines(ref_diff[0], -0.3, 0.3, color='red')
        ax.hlines(ref_diff[1],  0.7, 1.3, color='red')
        ax.hlines(ref_diff[2],  1.7, 2.3, color='red')
        plt.savefig('/home/hubert/DPhil_Studies/2021-04_Study_A_Diffusion/results/abm_ref_sim_grouped_boxplot.png',bbox_inches='tight', dpi=300)

    all_data_evals = sorted(glob.glob('/home/hubert/DPhil_Studies/2021-04_Study_A_Diffusion/results/*/abm_data_eval_df.obj', recursive=True))
    all_data_dfs = []
    if len(all_data_evals) == 3:
        print('All Data Eval DFs found. Loading in...')
        for i, e in enumerate(all_data_evals):
            with open(e, 'rb') as f:
                loaded = pickle.load(f)
                loaded['Period'] = i+1
                all_data_dfs.append(loaded)
        all_data = pd.concat(all_data_dfs, ignore_index=True)

        print('Plotting grouped boxplot for ref sim percent')
        rename_cols = {**eval_labels, **param_labels}
        all_data.rename(columns=rename_cols, inplace=True)
        for _, w in param_labels.items():
            plt.figure(figsize=(15,8))
            all_data[w] = all_data[w].apply(lambda x: round(x, 1))
            sns.boxplot(
                x = 'Period',
                y = 'Simulated Percentage Difference',
                hue = w,
                data=all_data
            )
            plt.savefig(f'/home/hubert/DPhil_Studies/2021-04_Study_A_Diffusion/results/abm_ref_sim_{w.lower()}_grouped_boxplot.png', bbox_inches='tight', dpi=300)
        # all_data = all_data_dfs[0].merge(all_data_dfs[1], suffixes=('_1', '_2'), on = param_names)
        # all_data = all_data.merge(all_data_dfs[2], on=param_names)


    ############################################################################
    # Metric table output
    ############################################################################


    # primary ht distribution for users
    print('generating primary_ht distribution')
    primary_ht_counter = Counter()
    for _, attributes in G.nodes(data=True):
        primary_ht_counter[attributes['primary_ht']] += 1
    primary_ht_df = pd.DataFrame.from_dict(primary_ht_counter, orient='index', columns = ['Percentage'])
    primary_ht_df['Percentage'] = 100*primary_ht_df['Percentage']/primary_ht_df['Percentage'].sum()
    primary_ht_df = primary_ht_df.sort_values('Percentage', ascending=False)

    primary_ht_savepath = f'/home/hubert/DPhil_Studies/2021-04_Study_A_Diffusion/results/0{args.group_num}_group/abm_primary_ht_dist.txt'

    primary_ht_savepath_obj = f'/home/hubert/DPhil_Studies/2021-04_Study_A_Diffusion/results/0{args.group_num}_group/abm_primary_ht_dist.obj'

    primary_ht_df.to_latex(primary_ht_savepath, float_format = "%.2f", index=True)

    with open(primary_ht_savepath_obj, 'wb') as f:
        pickle.dump(primary_ht_df, f)


    all_dist_objs = glob.glob('/home/hubert/DPhil_Studies/2021-04_Study_A_Diffusion/results/*/abm_primary_ht_dist.obj', recursive=True)
    print(all_dist_objs)
    all_dist = []
    if len(all_dist_objs) == 3:
        print('All Primary HT Dist DFs found. Combinings for east of LaTeX...')
        for i in all_dist_objs:
            with open(i, 'rb') as f:
                all_dist.append(pickle.load(f))
        final_dist_df = all_dist[0].merge(all_dist[1], suffixes=('_1', '_2'), left_index=True, right_index=True, how='outer')
        final_dist_df = final_dist_df.merge(all_dist[2], how='outer', left_index=True, right_index=True)
        final_dist_df = final_dist_df.sort_values('Percentage',ascending=False)
        final_dist_df = final_dist_df.sort_values('Percentage_2',ascending=False)
        final_dist_df = final_dist_df.sort_values('Percentage_1',ascending=False)
        final_dist_savepath = f'/home/hubert/DPhil_Studies/2021-04_Study_A_Diffusion/results/abm_total_primary_dist.txt'
        final_dist_df.to_latex(final_dist_savepath, float_format="%.2f", index=True)


    print('TRYING TO SAVE EXAPMLE RES')

    # unq, indices= np.unique(params, axis=0, return_inverse=True)

    # indices of hashtags needed for this group
    x = np.array([np.where(np.array(key_order)==x)[0][0] for x in act['index']])
    # print(reference_df['index'])
    # print(x)

    # use these unique incides to extract the desired processed result
    # result_np_array_processed = np.zeros(shape = (len(unq), len(x)))

    # filter down necessary results to just those found in the line before
    result_np_array_processed = results[:,x]


    # find where the desired print result is:
    where = dataplot_df.sort_values('percent_eval', ascending=True).iloc[0,:8]
    # where = np.where((params==where).all(axis=1))[0]
    where = np.flatnonzero(np.equal(params, np.array(where)).all(1))[0]
    act['ABM'] = result_np_array_processed[where]

    act.to_latex(f'/home/hubert/DPhil_Studies/2021-04_Study_A_Diffusion/results/0{group_num}_group/abm_example_results_df.txt')

    ############################################################################
    # Influence matrix results
    ############################################################################

    # INSERT GRAPH CODE HERE
    print('Processing Graph')

    d_graph = extract_directions(G)
    d_graph.remove_edges_from(nx.selfloop_edges(d_graph))

    print(f'Graph density is {nx.density(d_graph)}')

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
    plt.savefig(f'/home/hubert/DPhil_Studies/2021-04_Study_A_Diffusion/results/0{args.group_num}_group/abm_graph_original.png', transparent=True, bbox_inches='tight', dpi=300)
    plt.close()


    # DO THE SAME FOR HISTORY GRAPH
    # INSERT GRAPH CODE HERE
    # print('Processing History Graph')

    # h_graph = extract_directions()
    # h_graph.remove_edges_from(nx.selfloop_edges(d_graph))

    # degree_dict = dict(d_graph.out_degree)

    # # fixing the size of the figure
    # plt.figure(figsize =(15, 10))

    # node_color = [d_graph.out_degree(v) for v in d_graph]
    # # node colour is a list of degrees of nodes

    # node_size = [0.0005 * nx.get_node_attributes(d_graph, 'population')[v] for v in d_graph]
    # size of node is a list of population of cities

    # edge_width = [0.0005 * d_graph[u][v]['weight'] for u, v in d_graph.edges()]
    # # width of edge is a list of weight of edges

    # nx.draw_networkx(d_graph,
    #                 pos = nx.spring_layout(d_graph, k=20),
    #                 node_color = node_color,
    #                 with_labels = True,
    #                 #  width = edge_width,
    #                 node_size = [v * 500 for v in degree_dict.values()],
    #                 cmap = plt.cm.Blues,
    #                 font_size = 20)

    # plt.axis('off')
    # plt.tight_layout();
    # plt.savefig(f'/home/hubert/DPhil_Studies/2021-04_Study_A_Diffusion/results/0{args.group_num}_group/abm_graph_original.png', transparent=True, bbox_inches='tight')
    # plt.close()


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