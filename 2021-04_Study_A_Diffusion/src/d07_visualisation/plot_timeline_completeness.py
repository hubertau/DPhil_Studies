#!/usr/bin/python3.9

import matplotlib.pyplot as plt
import pickle
import os
import pandas as pd


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

def main():

    selected_date_ranges_file = '../../data/02_intermediate/FAS_selected_date_ranges.obj'
    timeline_completeness_pickle_fname = '../../data/02_intermediate/timeline_completeness.obj'
    plot_save_file = '../../results/plot_timeline_completeness.png'
    cdf_plot_save_file = '../../results/plot_timeline_completeness_cdf.png'

    # first, how many users are incomplete?

    # read in selected date ranges.
    with open(selected_date_ranges_file, 'rb') as f:
        selected_date_ranges = pickle.load(f)

    # read in existing timeline completeness file
    if os.path.isfile(timeline_completeness_pickle_fname):
        with open(timeline_completeness_pickle_fname, 'rb') as f:
            timeline_completeness = pickle.load(f)

    # simple scatter plot to seee which ones are not compelte
    f = plt.figure(figsize=(25,15))
    plt.scatter(range(len(timeline_completeness)), [i[1] for i in timeline_completeness])
    plt.savefig(plot_save_file)

    print('Saved at {}'.format(plot_save_file))

    # pd.DataFrame([i[1] for i in timeline_completeness]).describe(datetime_is_numeric=True)


    # plot cdf
    stats_df = generate_cdf_values([i[1] for i in timeline_completeness])

    f = plt.figure(figsize=(25,15))
    stats_df.plot(x = 'value', y = 'cdf', grid = True)
    plt.savefig(cdf_plot_save_file)

if __name__ == '__main__':
    main()