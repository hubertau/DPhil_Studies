'''
Script for late stage analysis from bispec_clustering_eval.py df output
'''
import pandas as pd
import numpy as np
import plotnine

from bispec_clustering_eval import BSCresults

class TimelineAnalyzer(BSCresults):

    def __init__(self, input_df):

        self.df = input_df
        self.vocab_indices = [i for i, s in enumerate(list(self.df.columns)) if 'vocab' in s]

    def print_stats(self):
        print(self.df['is_retweet'].sum())
        print(self.df['internal_retweet'].sum())
        print(self.df['vocab:#femmes'].sum())
