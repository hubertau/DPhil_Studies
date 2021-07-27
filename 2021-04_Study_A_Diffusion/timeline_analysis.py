'''
Script for late stage analysis from bispec_clustering_eval.py df output
'''
import pandas as pd
import numpy as np
import os
import jsonlines
import datetime
import plotnine

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
            plot_data = self.df.iloc[:,[3, vocab_entry[0]]]
            self.userdisthist = plotnine.ggplot(plot_data)\
                + plotnine.aes(x="created_at", y=vocab_entry[1]) \
                + plotnine.geom_point() \
                + plotnine.labs(title = "Vocab Usage")
            self.userdisthist.save(
                plot_savename,
                dpi=600
            )

    def plot_usage_per_user(self):

        # run plot setup
        self._plot_setup()

        # collect unique users 
        self.df_userlist = self.df.author_id.unique()

        for vocab_entry in self.vocab_colnames:
            for author in self.df_userlist:
                author_save_path = os.path.join(self.image_path,str(author))
                if os.path.isdir(author_save_path):
                    pass
                else:
                    os.mkdir(author_save_path)

                author_df_slice = self.df.iloc[self.df['author_id']==author,:]

                for vocab_entry in self.vocab_colnames:
                    plot_savename = os.path.join(author_save_path, vocab_entry[1]+ ".png") 
                    plot_data = author_df_slice.iloc[:,[3, vocab_entry[0]]]
                    self.userdisthist = plotnine.ggplot(plot_data)\
                        + plotnine.aes(x="created_at", y=vocab_entry[1]) \
                        + plotnine.geom_point() \
                        + plotnine.labs(title = "Vocab Usage")
                    self.userdisthist.save(
                        plot_savename,
                        dpi=600
                    )

    @time_function
    def plot_FAS_activity(self, search_query_text_file):

        # read in hashtags to plot:
        with open(search_query_text_file, 'r') as f:
            FAS_hashtags = f.readlines()
            FAS_hashtags = [i.rstrip('\n') for i in FAS_hashtags]

        # create output
        FAS_activity_dict = {
            'created_at': [],
            'hashtags': []
        }

        for user_jsonl_file in self.user_file_dir:
            with jsonlines.open(user_jsonl_file) as reader:
                for tweet_jsonl in reader:
                    tweet_list_in_file = tweet_jsonl['data']
                    for tweet_data in tweet_list_in_file:
                        if 'entities' in tweet_data:
                            if 'hashtags' in tweet_data['entities']:
                                hts = [i['tag'].lower() for i in tweet_data['entities']['hashtags']]
                                if any(item in FAS_hashtags for item in hts):
                                    FAS_activity_dict['created_at'].append(datetime.datetime.fromisoformat(tweet_data['created_at'][:-1]))

                                    # get hashtag overlap
                                    hts_overlap = set(hts).intersection(set(FAS_hashtags))

                                    FAS_activity_dict['hashtags'].append(hts_overlap)

        FAS_activity_df = pd.DataFrame.from_dict(FAS_activity_dict)

        for hashtag in FAS_hashtags:
            FAS_activity_df[hashtag] = any(hashtag.lower() == item for item in FAS_activity_df['hashtags'])

            FAS_activity_df[hashtag] = FAS_activity_df['hashtags'].apply(lambda x: any(hashtag.lower() == item for item in x)

        for column in FAS_activity_df.columns:
            