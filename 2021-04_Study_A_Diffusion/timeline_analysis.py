'''
Script for late stage analysis from bispec_clustering_eval.py df output
'''
import pandas as pd
import numpy as np
import os
import jsonlines
import datetime
import plotnine
import glob
import tqdm

from bispec_clustering_eval import BSCresults

class TimelineAnalyzer(BSCresults):

    # def get_vocab_indices(self):

    #     self.vocab_indices = [i for i, s in enumerate(list(self.df.columns)) if 'vocab' in s]

    def print_stats(self):

        percentage_rt = 100*self.df['is_retweet'].sum()/len(self.df)

        print('Number of tweets in df: {}'.format(len(self.df)))
        print('Number of retweets in df: {}'.format(self.df['is_retweet'].sum()))
        print('Percentage Retweets'.format(percentage_rt))
        print('Number of retweets of other users in the cluster: {}'.format(self.df['internal_retweet'].sum()))


    def time_function(func):

        """
        Wrapper function to time execution.
        """

        def inner(*args, **kwargs):
            timefunc_start_time = datetime.datetime.now()
            print('\nStart Time: {}'.format(timefunc_start_time))
            result = func(*args, **kwargs)
            print('Total Time Taken: {}'.format(datetime.datetime.now()-timefunc_start_time))
            return result
        return inner

    def _plot_setup(self):

        # obtain df column names
        self.colnames = self.df.columns

        # create images folder to store results
        self.image_path = os.path.join(self.data_dir, 'images/')
        if os.path.isdir(self.image_path):
            pass
        else:
            os.mkdir(self.image_path)
            print('Directory created: {}'.format(self.image_path))

        self.vocab_colnames = []
        for i,e in enumerate(self.colnames):
            if 'vocab' in e:
                self.vocab_colnames.append((i,e))

    def plot_usage_vocab(self):

        # run plot setup
        self._plot_setup()

        # histograms of user number per cluster
        for vocab_entry in self.vocab_colnames:
            plot_savename = os.path.join(self.image_path, vocab_entry[1]+ ".png") 
            plot_data = self.df.loc[:,['created_at', vocab_entry[1]]]
            plot_data.loc[:,'created_at'] = plot_data.copy()['created_at'].apply(lambda x: x.date())
            plot_data = plot_data.groupby('created_at').sum()
            if plot_data[vocab_entry[1]].sum() == 0:
                continue
            plot_data = plot_data.reset_index()
            self.userdisthist = plotnine.ggplot(plot_data) \
                + plotnine.aes(x="created_at", y=vocab_entry[1]) \
                + plotnine.geom_point(size=0.4) \
                + plotnine.geom_line(group=1) \
                + plotnine.theme(axis_text_x =  plotnine.element_text(rotation = 45, hjust=1))
            self.userdisthist.save(
                plot_savename,
                dpi=600,
                verbose = False
            )

    def plot_usage_per_user(self, size=3):

        # run plot setup
        self._plot_setup()

        # collect unique users 
        self.df_userlist = self.df.author_id.unique()

        for author in tqdm.tqdm(self.df_userlist):
            author_df_slice = self.df.copy()[self.df['author_id']==author]
            author_df_slice.loc[:,'created_at'] = author_df_slice['created_at'].apply(lambda x: x.date())
            plot_savename = os.path.join(self.image_path, str(author) + '.png')
            author_df_slice = author_df_slice.groupby('created_at').sum()
            author_df_slice = author_df_slice.reset_index()
            author_df_slice = pd.wide_to_long(author_df_slice, stubnames='vocab:', i='created_at', j='hashtag', suffix = '.+')
            author_df_slice = author_df_slice.reset_index()
            author_df_slice = author_df_slice[author_df_slice['vocab:']!=0]

            self.author_plot = plotnine.ggplot(author_df_slice, plotnine.aes(x = 'created_at', y = 'vocab:', color = 'hashtag')) + \
                plotnine.geom_point(size = size) + \
                plotnine.geom_line() + \
                plotnine.theme(axis_text_x =  plotnine.element_text(rotation = 45, hjust=1))

            self.author_plot.save(
                plot_savename,
                dpi=600,
                verbose=False
            )

        # for vocab_entry in self.vocab_colnames:
        #     for author in self.df_userlist:
        #         # author_save_path = os.path.join(self.image_path,str(author))
        #         # if os.path.isdir(author_save_path):
        #         #     pass
        #         # else:
        #         #     os.mkdir(author_save_path)

        #         author_df_slice = self.df.iloc[self.df['author_id']==author,:]

        #         plot_savename = self.image_path + author + '.png'
        #         plot_data = author_df_slice[[e for i,e in self.vocab_colnames].append('created_at')]
        #         plot_data = pd.wide_to_long(self.df, stubnames='vocab:',i='created_at', j='phrase')
        #         plot_data = plot_data.groupby(['created_at', 'phrase']).agg('sum')
        #         plot_data = plot_data[plot_data['sum']!=0]
        #         self.userdisthist = plotnine.ggplot(plot_data)\
        #             + plotnine.aes(x="created_at", y='sum', color = 'phrase') \
        #             + plotnine.geom_point() \
        #             + plotnine.labs(title = "Vocab Usage")
        #         self.userdisthist.save(
        #             plot_savename,
        #             dpi=600
        #         )

    @time_function
    def plot_FAS_activity(self, search_query_text_file):

        self._plot_setup()

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

        self.new_FAS_dir = '/home/hubert/DPhil_Studies/2021-04_Study_A_Diffusion/collection_results_2021_06_19_16_21/data'

        for user_jsonl_file in tqdm.tqdm(glob.glob(os.path.join(self.new_FAS_dir,'FAS*.jsonl')), desc='processing FAS jsonl files'):
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

        self.FAS_activity_df = FAS_activity_df

        # groupby to get sums
        self.FAS_activity_df = self.FAS_activity_df.groupby('created_at').sum()
        self.FAS_activity_df = self.FAS_activity_df.reset_index() 

        self.FAS_activity_df_long = pd.wide_to_long(self.FAS_activity_df, stubnames='vocab:#', i='created_at', j='hashtag', suffix = '.+')

        self.FAS_activity_df_long= self.FAS_activity_df_long.reset_index()
        self.FAS_activity_df_long = self.FAS_activity_df_long[self.FAS_activity_df_long['vocab:#']!=0]

        self.FAS_activity_plot = plotnine.ggplot(self.FAS_activity_df_long, plotnine.aes(x = 'created_at', y = 'vocab:#', color = 'hashtag')) + \
            plotnine.geom_line(group=1) + \
            plotnine.scale_x_datetime(date_breaks = '1 month') + \
            plotnine.theme(
                text = plotnine.element_text(family=['Noto Sans KR', 'Noto Serif JP','STIX Two Text', 'Cairo']), 
                axis_text_x =  plotnine.element_text(rotation = 45, hjust=1)) + \
            plotnine.ggtitle('Activity Plot for Searched #MeToo Hashtags') + \
            plotnine.xlab('Date') + \
            plotnine.ylab('Volume of Activity')

        # save plot
        plot_savename = os.path.join(self.image_path, 'FAS_activity.png')
        self.FAS_activity_plot.save(
                        plot_savename,
                        width=15,
                        height=10,
                        dpi=600,
                        verbose = False
                    )

        self.FAS_activity_plot_log = plotnine.ggplot(self.FAS_activity_df_long, plotnine.aes(x = 'created_at', y = 'vocab:#', color = 'hashtag')) + \
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
        plot_savename = os.path.join(self.image_path, 'FAS_activity_log.png')
        self.FAS_activity_plot_log.save(
                        plot_savename,
                        width=15,
                        height=10,
                        dpi=600,
                        verbose = False
                    )