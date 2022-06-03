#!/usr/bin/python3.9

'''
Script to plot the FAS peaks
'''

import os
import datetime
import pickle
import matplotlib.pyplot as plt
import re
import pandas as pd
import numpy as np
import h5py
import glob

def main():

    FAS_object_folder = '../../data/02_intermediate/'
    FAS_object = os.path.join(FAS_object_folder, 'FAS_peak_analysis.hdf5')
    FAS_peak_plot_folder = '../../results/'

    data = pd.read_hdf(FAS_object, key='plot_data')
    with h5py.File(FAS_object,'rb') as f:
        peak_detections = f['peak_detections']

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

    # skipped = 0
    # for index,peakdetection in enumerate(results):

    #     x = data[data['hashtag']==peakdetection[0]].loc[:,['vocab:#']] 

    #     # just check if figures already exist
    #     filename = FAS_peak_plot_folder + peakdetection[0] +'.jpg'

    #     if os.path.isfile(filename):
    #         skipped+=1
    #         continue

    #     f = plt.figure()
    #     f.set_figwidth(15)
    #     f.set_figheight(10)
    #     plt.plot(x.reset_index().loc[:,'vocab:#'])
    #     plt.plot(results[index][1], x.reset_index().loc[results[index][1],'vocab:#'].values.flatten(),"x")
    #     plt.savefig(filename,bbox_inches='tight')
    #     plt.close()

    # print('{} plots skipped because they already existed.'.format(skipped))

    # continuity check
    for hashtag in data['hashtag'].unique():
        z = data[data['hashtag']==hashtag]
        first = datetime.datetime.strptime('2017-10-16', '%Y-%m-%d')
        for index,i in enumerate(z['created_at']):
            assert i == first.date(), (index, i, first)
            first += datetime.timedelta(days=1)
    print('continuity check for dates, all okay')

    def FAS_dates(FAS_filename):
        return re.split('[_.]',FAS_filename)[1:3]

    # print(FAS_dates('FAS_2017-10-16_2017-11-16.jsonl'))

    # now sanity check: which dates are inclusive?
    flist = glob.glob('/home/hubert/DPhil_Studies/2021-04_Study_A_Diffusion/data/FAS*.jsonl')

    # quick bit of code to determine the index of the main metoo peak results
    for i,e in enumerate(peak_detections):
        if e[0] == 'metoo':
            metoo_index = i
            break
    print('metoo_index: {}'.format(metoo_index))

    # collate peak locations. This is so that any overlaps can be smoothed out. This will then make the selection of which part to collect easier.

    # first, take the peak locations of #metoo and treat those as bases around which to start collection. This strategy does NOT contain peaks that are not around #metoo peaks. THIS MAY NEED TO BE REVISED OR ADDED TO.

    verbose = False

    ranges = []
    width = 21 # number of days to collect before and after each peak
    for index, peak in enumerate(peak_detections[metoo_index][1]):
        ranges.append((max(0,peak-width), min(peak+width, 1096)))

    if verbose:
        print(ranges)

    def review_ranges(ranges):

        # define output list
        new_ranges = []

        # iterate over the ranges
        for index in range(len(ranges)):

            # if there already exists one entry in output list and the latest entry has a max date later than the early date in the current range iterating over, then we can continue the scanning. Otherwise, a prospective pair that is simply the current range is created. This allows the subsequent while loop.
            if len(new_ranges) > 0 and new_ranges[-1][1]>=ranges[index][0]:
                continue
            else:
                prospective_pair = ranges[index]
            ranges_index = index

            # if we have reached the end of ranges, we need to append whatever prospective pair we have.
            if index+1 == len(ranges):
                new_ranges.append(prospective_pair)
                break

            # create prospective pair. If the current range overlaps with the next one in ranges, then a prospective pair encompassing the current one needs to be created.
            if ranges[index][1] >= ranges[ranges_index+1][0]:
                prospective_pair = (ranges[index][0], ranges[ranges_index+1][1])
                ranges_index += 1

            while prospective_pair[1] >= ranges[ranges_index+1][0]:

                if verbose:
                    print(prospective_pair)

                # depending on if the later date on the next range encompasses the current prospective max date or not, the update needs to be adapted.
                if prospective_pair[1] >= ranges[ranges_index+1][1]:

                    prospective_pair = (ranges[index][0], prospective_pair[1])
                    ranges_index += 1

                elif prospective_pair[1] < ranges[ranges_index+1][1]:

                    prospective_pair = (ranges[index][0], ranges[ranges_index+1][1])
                    ranges_index += 1

            # append prospective pair when no more updates need to be made
            new_ranges.append(prospective_pair)

        return new_ranges

    if verbose:
        print(results[metoo_index][1])
        print(review_ranges(ranges))

    # function that uses prominences to obtain a measure of strength of peaks captured within ranges proposed.
    def range_strengths(reviewed_ranges, results):
        prominences = []
        for pair in reviewed_ranges:
            pair_prominence = 0
            # start at [1:] to skip #metoo
            no_peaks = True
            filtered_results = results[:metoo_index] + results[metoo_index+1:]
            for hashtag_peaks in filtered_results:
                for index, peak in enumerate(hashtag_peaks[1]):
                    if peak > pair[0] and peak <= pair[1]:
                        no_peaks = False
                        pair_prominence += hashtag_peaks[2]['prominences'][index]
            if no_peaks:
                prominences.append(0)
            else:
                prominences.append(pair_prominence)

        return prominences

    prominences = range_strengths(review_ranges(ranges), results)

    if verbose:
        print(prominences)

    width_normalised_prominences = [prom/(width[1]-width[0]) for width, prom in zip(review_ranges(ranges), prominences)]

    # converting 0-1083 back to dates

    def unit_conv(val):
            return datetime.datetime.strptime('2017-10-16', '%Y-%m-%d') + datetime.timedelta(days=int(val))

    def conv_to_date_ranges(ranges):
        return_ranges = []

        for pair in ranges:
            return_ranges.append((unit_conv(pair[0]), unit_conv(pair[1])))

        return return_ranges

    dated_reviewed_ranges = conv_to_date_ranges(review_ranges(ranges))
    # print(dated_reviewed_ranges)

    #visualising the parts of the graph that will be sampled
    data = data[data['vocab:#']!=0]

    # convert to datetime objects in column
    data['created_at'] = pd.to_datetime(data['created_at'])

    f = plt.figure()
    f.set_figwidth(20)
    f.set_figheight(15)
    plt.vlines([i[0]+0.5*(i[1]-i[0]) for i in dated_reviewed_ranges],
        ymin = 0,
        ymax = data['vocab:#'].max(),
        linewidth = [(i[1]-i[0]).days for i in dated_reviewed_ranges],
        # linewidth = 5,
        alpha = width_normalised_prominences/max(width_normalised_prominences)
        );
    wide_data = data.pivot(index='created_at', columns='hashtag', values='vocab:#')
    # plt.plot('created_at','vocab:#', data = bscres.FAS_activity_df_long)
    plt.plot(wide_data);
    plt.savefig(FAS_peak_plot_folder + 'FAS_activity_with_peaks.png')

    # also plot detected peaks
    for peaksdetections in results:

        converted_peaks = np.array([unit_conv(i) for i in peaksdetections[1]])
        peak_plot_data = data[data['hashtag']==peaksdetections[0]]
        peak_plot_data = peak_plot_data[peak_plot_data['created_at'].isin(converted_peaks)].loc[:,['created_at','vocab:#']]
        plt.plot('created_at', 'vocab:#', 'x', data = peak_plot_data, )

    print(width_normalised_prominences/max(width_normalised_prominences))

    with open(os.path.join(FAS_object_folder,'FAS_prominences.obj'),'wb') as f:
        pickle.dump(width_normalised_prominences,f)
    with open(os.path.join(FAS_object_folder,'FAS_ranges.obj'), 'wb') as f:
        pickle.dump(dated_reviewed_ranges, f)

    # plt.semilogy(basey=2)


if __name__ == '__main__':
    main()