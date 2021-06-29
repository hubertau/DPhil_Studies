'''
Script for late stage analysis from bispec_clustering_eval.py df output
'''
import pandas as pd
import numpy as np
import os
import plotnine

from bispec_clustering_eval import BSCresults

class TimelineAnalyzer(BSCresults):

    def get_vocab_indices(self):

        self.vocab_indices = [i for i, s in enumerate(list(self.df.columns)) if 'vocab' in s]

    def print_stats(self):

        percentage_rt = 100*self.df['is_retweet'].sum()/len(self.df)

        print('Number of tweets in df: {}'.format(len(self.df)))
        print('Number of retweets in df: {}'.format(self.df['is_retweet'].sum()))
        print('Percentage Retweets'.format(percentage_rt))
        print('Number of retweets of other users in the cluster: {}'.format(self.df['internal_retweet'].sum()))

    def plot_usage_vocab(self):

        # histogram of user number per cluster
        plot_data = self.df.iloc[:,[3,self.vocab_indices[0]]]
        self.userdisthist = plotnine.ggplot(plot_data)\
            + plotnine.aes(x="created_at", y="vocab:#femmes") \
            + plotnine.geom_point() \
            + plotnine.labs(title = "Vocab Usage")
        self.userdisthist.save(
            os.path.join(self.data_dir,"vocab_plot.png"),
            dpi=600
        )

        plot_data = self.df.iloc[:,[3,self.vocab_indices[1]]]
        self.userdisthist = plotnine.ggplot(plot_data)\
            + plotnine.aes(x="created_at", y="vocab:de harcèlement sexuel") \
            + plotnine.geom_point() \
            + plotnine.labs(title = "Vocab Usage")
        self.userdisthist.save(
            os.path.join(self.data_dir,"vocab_plot2.png"),
            dpi=600
        )

        plot_data = self.df.iloc[:,[3,self.vocab_indices[2]]]
        self.userdisthist = plotnine.ggplot(plot_data)\
            + plotnine.aes(x="created_at", y="vocab:faites aux femmes") \
            + plotnine.geom_point() \
            + plotnine.labs(title = "Vocab Usage")
        self.userdisthist.save(
            os.path.join(self.data_dir,"vocab_plot3.png"),
            dpi=600
        )

        plot_data = self.df.iloc[:,[3,self.vocab_indices[3]]]
        self.userdisthist = plotnine.ggplot(plot_data)\
            + plotnine.aes(x="created_at", y="vocab:le harcèlement sexuel") \
            + plotnine.geom_point() \
            + plotnine.labs(title = "Vocab Usage")
        self.userdisthist.save(
            os.path.join(self.data_dir,"vocab_plot4.png"),
            dpi=600
        )

        plot_data = self.df.iloc[:,[3,self.vocab_indices[4]]]
        self.userdisthist = plotnine.ggplot(plot_data)\
            + plotnine.aes(x="created_at", y="vocab:victimes de violences") \
            + plotnine.geom_point() \
            + plotnine.labs(title = "Vocab Usage")
        self.userdisthist.save(
            os.path.join(self.data_dir,"vocab_plot5.png"),
            dpi=600
        )

