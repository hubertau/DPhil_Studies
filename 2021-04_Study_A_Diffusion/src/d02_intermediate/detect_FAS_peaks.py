#!/usr/bin/python3.9

'''
Script to detect the FAS peaks using scipy and save the results.
'''

import os
import pickle
import scipy.signal
import pandas as pd
import datetime

def main():

    # save bscres. Load in existing df.
    FAS_object_folder = '../../data/02_intermediate/'
    FAS_object_path = os.path.join(FAS_object_folder,'FAS_plot_data.obj')


    with open(FAS_object_path,'rb') as f:
        data = pickle.load(f)

    print('file found and loaded')

    # first convert to datetime
    data['created_at'] = pd.to_datetime(data['created_at'])

    # then set index to be created_at. This will help for the next step.
    data = data.set_index('created_at')
    # define function that will reindex the dataframe with dates. This is why we needed to set created_at as the index.
    def reindex_by_date(df):
        df = df.reindex(pd.date_range(datetime.datetime(2017,10,16), datetime.
        datetime(2020,10,15)), fill_value = 0)
        df['hashtag'] = df['hashtag'].replace(to_replace=0, method='ffill')
        return df

    # groupby each hashtag, then apply the function. Then the index needs to be reset so that we can reference the date by ['created_at']
    z = data.groupby('hashtag').apply(reindex_by_date).reset_index(0,drop=True).reset_index().rename(columns={'index':'created_at'})

    # ensure the counts are integers
    z['vocab:#'] = z['vocab:#'].astype(int)

    # ensure the totals are not messed up
    assert data['vocab:#'].sum() == z['vocab:#'].sum()

    # set equality
    data = z

    # sanity check: the number of dates for each hashtag is correct.
    assert (data.groupby('hashtag').count()['vocab:#']==1096).all()

    # actual peak finding.
    # parameters for 2021-09-20: prominence = 0.9, distance=28, height = 0.05*max(z)

    results = []
    for hashtag in data['hashtag'].unique():

        assert len(data[data['hashtag']==hashtag]) == 1096
        z = data[data['hashtag']==hashtag]['vocab:#']
        peaks, peak_properties = scipy.signal.find_peaks(
            z,
            prominence = 0.9,
            distance=28,
            height = 0.05*max(z)
        )
        results.append((hashtag, peaks, peak_properties))

    #print hashtags with no detected peaks:
    for index, peakdetection in enumerate(results):
        if len(peakdetection[1]) == 0:
            print('{} at index {} has no peaks detected.'.format(peakdetection[0], index))

    with open(os.path.join(FAS_object_folder, 'FAS_peak_detections.obj'), 'wb') as f:
        pickle.dump(results, f)
    with open(os.path.join(FAS_object_folder, 'FAS_plot_data_padded.obj'), 'wb') as f:
        pickle.dump(data, f)

if __name__ == '__main__':

    main()
