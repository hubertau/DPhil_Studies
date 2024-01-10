#!/usr/bin/python3.9

import matplotlib.pyplot as plt
import pickle
import os
import pandas as pd
import argparse
import h5py
import glob
import jsonlines
import datetime
import matplotlib

def generate_cdf_values(data):

    # Get the frequency, PDF and CDF for each value in the series

    data = pd.Series(data, name = 'value')
    df = pd.DataFrame(data)

    # Frequency
    stats_df = df \
    .groupby('value') \
    ['value'] \
    .agg('count') \
    .pipe(pd.DataFrame) \
    .rename(columns = {'value': 'frequency'})

    # PDF
    stats_df['pdf'] = stats_df['frequency'] / sum(stats_df['frequency'])

    # CDF
    stats_df['cdf'] = stats_df['pdf'].cumsum()
    stats_df = stats_df.reset_index()
    return stats_df

def main(args):

    group_num = args.group_num
    group_index = group_num - 1
    print(group_num)

    FAS_peak_file = '../../data/02_intermediate/FAS_peak_analysis.hdf5'
    tc_file = '../../data/02_intermediate/timeline_completeness.hdf5'
    plot_save_file = f'../../results/0{group_num}_group/plot_timeline_completeness.png'
    cdf_plot_save_file = f'../../results/0{group_num}_group/plot_timeline_completeness_cdf.png'

    # first, how many users are incomplete?

    # read in selected date ranges.
    with h5py.File(FAS_peak_file, 'r') as f:
        selected_date_ranges = f['segments']['selected_ranges'][group_index]
        date_range_start = selected_date_ranges[0].decode()
        print(date_range_start)
        date_range_start = datetime.datetime.strptime(date_range_start, '%Y-%m-%d')
        # date_range_end = selected_date_ranges[1].decode()


    with h5py.File(tc_file, 'a') as f:
        if f'group_{group_num}' in f.keys():
            timeline_completeness = f[f'group_{group_num}'][:]
            timeline_completeness = [datetime.datetime.strptime(i.decode(),'%Y-%m-%d') for i in timeline_completeness]

        else:

            timeline_flist = glob.glob(f'/home/hubert/DPhil_Studies/2021-04_Study_A_Diffusion/data/01_raw/0{group_num}_group/timeline*.jsonl')

            timeline_completeness = []

            for index, twitter_user_timeline in enumerate(timeline_flist):
                print(f'Processing file num {index} of {len(timeline_flist)}')
                current_max_date=date_range_start
                with jsonlines.open(twitter_user_timeline) as reader:
                    for tweet_jsonl in reader:
                        tweet_list_in_file = tweet_jsonl['data']
                        for tweet_data in tweet_list_in_file:
                            tweet_created_at = datetime.datetime.fromisoformat(tweet_data['created_at'][:-1])
                            # print(tweet_created_at)
                            if tweet_created_at > current_max_date:
                                current_max_date = tweet_created_at
                timeline_completeness.append((os.path.split(twitter_user_timeline)[1],current_max_date))
                # print(current_max_date)
                # g.create_dataset(os.path.split(twitter_user_timeline)[1], data=datetime.datetimecurrent_max_date)

            timeline_completeness = [datetime.datetime.strftime(i[1], '%Y-%m-%d') for i in timeline_completeness]

            f.create_dataset(f'group_{group_num}', data = timeline_completeness)

    def set_size(width, fraction=1):
        """ Set aesthetic figure dimensions to avoid scaling in latex.

        Parameters
        ----------
        width: float
                Width in pts
        fraction: float
                Fraction of the width which you wish the figure to occupy

        Returns
        -------
        fig_dim: tuple
                Dimensions of figure in inches
        """
        # Width of figure
        fig_width_pt = width * fraction

        # Convert from pt to inches
        inches_per_pt = 1 / 72.27

        # Golden ratio to set aesthetic figure height
        golden_ratio = (5 ** 0.5 - 1) / 2

        # Figure width in inches
        fig_width_in = fig_width_pt * inches_per_pt
        # Figure height in inches
        fig_height_in = fig_width_in * golden_ratio

        return fig_width_in, fig_height_in

    # read in existing timeline completeness file
    # with h5py.File(tc_file, 'r') as f:
    #     print(f.keys())
    #     timeline_completeness = f[f'group_{group_num}'][:]

    # simple scatter plot to seee which ones are not compelte
    f = plt.figure(figsize=set_size(400))
    plt.style.use("style.mplstyle")
    nice_fonts = {
        "font.family": "serif"
    }
    matplotlib.rcParams.update(nice_fonts)
    plt.scatter(range(len(timeline_completeness)), timeline_completeness)
    plt.savefig(plot_save_file, bbox_inches = 'tight')

    print('Saved at {}'.format(plot_save_file))

    # pd.DataFrame([i[1] for i in timeline_completeness]).describe(datetime_is_numeric=True)


    # plot cdf
    stats_df = generate_cdf_values(timeline_completeness)

    f = plt.figure(figsize=(25,15))
    stats_df.plot(x = 'value', y = 'cdf', grid = True)
    plt.xlabel('')
    plt.xticks(rotation=45, ha='right')
    plt.savefig(cdf_plot_save_file, bbox_inches='tight')
    print(f'Saved at {cdf_plot_save_file}')

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='data completeness plots')

    parser.add_argument(
        'group_num',
        type=int
    )

    args = parser.parse_args()

    main(args)