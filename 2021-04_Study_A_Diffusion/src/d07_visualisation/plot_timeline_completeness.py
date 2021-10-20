#!/usr/bin/python3.9

import matplotlib.pyplot as plt
import pickle
import os
import pandas as pd

def main():

    selected_date_ranges_file = '../../data/02_intermediate/FAS_selected_date_ranges.obj'
    timeline_completeness_pickle_fname = '../../data/02_intermediate/timeline_completeness.obj'
    plot_save_file = '../../results/plot_timeline_completeness.png'

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

if __name__ == '__main__':
    main()