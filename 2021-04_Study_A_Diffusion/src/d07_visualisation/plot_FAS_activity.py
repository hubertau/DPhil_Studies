#!/usr/bin/python3.9

'''
Generate the simple FAS activity plot and save the data

'''

import plotnine
import os
import pandas as pd
import datetime
import glob
import pickle
import tqdm
import jsonlines
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import datetime
import glob
import os
import h5py
import pandas as pd

import sys

sys.path.append( '/home/hubert/DPhil_Studies/2021-04_Study_A_Diffusion/src/d04_modelling' )

from abm import *

def main():
    with h5py.File('/home/hubert/DPhil_Studies/2021-04_Study_A_Diffusion/data/02_intermediate/FAS_peak_analysis.hdf5', 'r') as f:
        peak_detections = f['peak_detections']['jotambe']['peak_locations'][:]

    plot_data = pd.read_hdf('/home/hubert/DPhil_Studies/2021-04_Study_A_Diffusion/data/02_intermediate/FAS_peak_analysis.hdf5', 'plot_data')

    #### PLOT OF SINGLE ONE WITH PEAKS ########################################
    sns.set(rc={"figure.figsize":(15, 10)})
    sns.set(font_scale=1.5)
    ht = 'kutoo'
    temp_plot_data = plot_data[plot_data['hashtag']==ht]
    with h5py.File('/home/hubert/DPhil_Studies/2021-04_Study_A_Diffusion/data/02_intermediate/FAS_peak_analysis.hdf5', 'r') as f:
        peak_detections = f['peak_detections'][ht]['peak_locations'][:]
    temp_peak_data = pd.DataFrame({'created_at':[unit_conv(i) for i in peak_detections]}).merge(temp_plot_data, on='created_at', how='left')
    temp_peak_data['hashtag'] = 'peak'
    # temp_plot_data = pd.concat((temp_plot_data,temp_peak_data))
    ax = sns.lineplot(data = temp_plot_data,
                    x='created_at',
                    y='vocab:#')
    sns.scatterplot(data = temp_peak_data,
                    x='created_at',
                    y='vocab:#', color='orange', s=50)

    ax.set(xlabel='Date',
        ylabel='Activity Count')

    plt.savefig(f'../../results/{ht}.png', dpi=300, transparent=True, bbox_inches='tight')

    ############################################################################


    #### PLOT OF ALL HT ACTIVITY ###############################################

    with open('/home/hubert/DPhil_Studies/2021-04_Study_A_Diffusion/data/archive/FAS_ranges.obj', 'rb') as f:
        ranges = pickle.load(f)

    with open('/home/hubert/DPhil_Studies/2021-04_Study_A_Diffusion/data/archive/FAS_prominences.obj', 'rb') as f:
        wnp = pickle.load(f)

    sns.set(rc={"figure.figsize":(15, 10)})
    sns.set(font_scale=1.5)
    import matplotlib
    # Customize matplotlib
    matplotlib.rcParams.update(
        {
            'text.usetex': False,
            'font.family': ['Noto Sans JP','Noto Sans TC', 'Noto Sans KR', 'Open Sans', 'Cairo'],
        }
    )

    # temp_plot_data = plot_data[plot_data['hashtag']==ht]
    # temp_plot_data = pd.concat((temp_plot_data,temp_peak_data))
    totals = plot_data.groupby('hashtag').sum()
    totals = totals.sort_values('vocab:#', ascending=False)
    to_plot=totals.iloc[:10,:].reset_index(0)

    plt.figure(figsize=(15,8))
    ax = sns.lineplot(data = plot_data[plot_data['hashtag'].isin(to_plot['hashtag'])],
                    x='created_at',
                    hue='hashtag',
                    y='vocab:#');

    ax.set(xlabel='Date',
        ylabel='Activity Count');

    plt.savefig(f'../../results/new_FAS_activity.png', dpi=300, transparent=True, bbox_inches='tight');

    plt.vlines([i[0]+0.5*(i[1]-i[0]) for i in ranges],
        ymin = 0,
        ymax = plot_data['vocab:#'].max(),
        linewidth = [(i[1]-i[0]).days for i in ranges],
        # linewidth = 5,
        alpha = wnp/max(wnp)
        );

    plt.savefig(f'../../results/new_FAS_activity_with_peaks.png', dpi=300, transparent=True, bbox_inches='tight');


    ############################################################################





def archive():

    print('\nRunning script to produce full archive search result activity plots for each hashtag searched.\n')

    # set up paths
    search_query_text_file = '../../references/search_hashtags.txt'
    image_folder = '../../results/'
    data_folder = '../../data/01_raw/'
    object_save_folder = '../../data/02_intermediate/'

    # read in hashtags to plot:
    with open(search_query_text_file, 'r') as f:
        FAS_hashtags = f.readlines()
        FAS_hashtags = [i.rstrip('\n') for i in FAS_hashtags]
        FAS_hashtags = [i.lower() for i in FAS_hashtags]

    # create output
    FAS_activity_dict = {
        'created_at': [],
        'hashtags': []
    }

    print('First processing the FAS files to collect the relevant information from the raw data files:\n')

    for user_jsonl_file in tqdm.tqdm(glob.glob(os.path.join(data_folder,'FAS*.jsonl')), desc='processing FAS jsonl files'):
        with jsonlines.open(user_jsonl_file) as reader:
            for tweet_jsonl in reader:
                tweet_list_in_file = tweet_jsonl['data']
                for tweet_data in tweet_list_in_file:
                    if 'entities' in tweet_data:
                        if 'hashtags' in tweet_data['entities']:
                            # remember that the Twitter objects dont' have '#' in front of the hashtags.
                            hts = ['#' + i['tag'].lower() for i in tweet_data['entities']['hashtags']]
                            if any(item in FAS_hashtags for item in hts):
                                FAS_activity_dict['created_at'].append(datetime.datetime.fromisoformat(tweet_data['created_at'][:-1]).date())

                                # get hashtag overlap
                                hts_overlap = set(hts).intersection(set(FAS_hashtags))

                                FAS_activity_dict['hashtags'].append(hts_overlap)

    print('converting to df')
    FAS_activity_df = pd.DataFrame.from_dict(FAS_activity_dict)

    print('getting feature counts')
    for hashtag in FAS_hashtags:
        # FAS_activity_df['vocab:'+hashtag] = any(hashtag.lower() == item for item in FAS_activity_df['hashtags'])

        FAS_activity_df['vocab:'+hashtag] = FAS_activity_df['hashtags'].apply(lambda x: any(hashtag.lower() == item for item in x))

    # groupby to get sums
    FAS_activity_df = FAS_activity_df.groupby('created_at').sum()
    FAS_activity_df = FAS_activity_df.reset_index() 

    FAS_activity_df_long = pd.wide_to_long(FAS_activity_df, stubnames='vocab:#', i='created_at', j='hashtag', suffix = '.+')

    FAS_activity_df_long= FAS_activity_df_long.reset_index()
    FAS_activity_df_long_with_zeros = FAS_activity_df_long.copy(deep=True)

    # drop_zeros
    FAS_activity_df_long = FAS_activity_df_long[FAS_activity_df_long['vocab:#']!=0]

    FAS_activity_plot = plotnine.ggplot(FAS_activity_df_long, plotnine.aes(x = 'created_at', y = 'vocab:#', color = 'hashtag')) + \
        plotnine.geom_line(group=1) + \
        plotnine.scale_x_datetime(date_breaks = '1 month') + \
        plotnine.theme(
            text = plotnine.element_text(family=['Noto Sans KR', 'Noto Serif JP','STIX Two Text', 'Cairo']), 
            axis_text_x =  plotnine.element_text(rotation = 45, hjust=1),
            plot_backgroun = plotnine.element_rect(colour='#f5ebe2')
            ) + \
        plotnine.ggtitle('Activity Plot for Searched #MeToo Hashtags') + \
        plotnine.xlab('Date') + \
        plotnine.ylab('Volume of Activity')

    # save plot
    plot_savename = os.path.join(image_folder, 'FAS_activity.png')
    print('\nSaving to {}'.format(plot_savename))
    FAS_activity_plot.save(
                    plot_savename,
                    width=15,
                    height=10,
                    dpi=600,
                    verbose = False
                )

    FAS_activity_plot_log = plotnine.ggplot(FAS_activity_df_long, plotnine.aes(x = 'created_at', y = 'vocab:#', color = 'hashtag')) + \
        plotnine.geom_line(group=1) + \
        plotnine.scale_x_datetime(date_breaks = '1 month') + \
        plotnine.theme(
            text = plotnine.element_text(family=['Noto Sans KR', 'Noto Serif JP','STIX Two Text', 'Cairo']), 
            axis_text_x =  plotnine.element_text(rotation = 45, hjust=1)) + \
        plotnine.ggtitle('Activity Plot for Searched #MeToo Hashtags') + \
        plotnine.xlab('Date') + \
        plotnine.ylab('Volume of Activity') + \
        plotnine.scale_y_continuous(trans='log10')

    # save plot
    plot_savename = os.path.join(image_folder, 'FAS_activity_log.png')
    print('\nSaving to {}'.format(plot_savename))
    FAS_activity_plot_log.save(
                    plot_savename,
                    width=15,
                    height=10,
                    dpi=600,
                    verbose = False
                )

    # save object
    print('saving collected FAS data for reuse.')
    with open(object_save_folder+'FAS_plot_data.obj','wb') as f:
        pickle.dump(FAS_activity_df_long_with_zeros, f)

if __name__ == '__main__':

    main()