'''
This script is to evaluate the clustering

For reference, this is form the evaluation
# # Sensitivity
# library(aricode)
# 
# res <- list()
# for(k in 50:300){
#   r <- biSpectralCoCluster(he_13,min_user = 10, k = k)
#   res[[k]] <- c(r[['users']]$topic_cluster, r[['hashtags']]$topic_cluster)
# }
# 
# rd <- data.frame()
# for(k in 60:300){
#   for(i in 1:10){
#     rd <- rbind(rd, data.frame(k=k,i=i, nmi=NMI(res[[k]],res[[k-i]])))
#   }
# }
# rd <- data.table(rd)
# theme_set(theme_bw(20))
# p <- ggplot(rd[, mean(nmi), by=k], aes(k, V1)) + geom_point() + geom_line() + stat_smooth()
# p <- p + xlab("Number of Clusters") + ylab("Mean\nNormalized Mutual\nInformation (NMI)")
# p

N.B. NMI is a measure of clustering results https://scikit-learn.org/stable/modules/generated/sklearn.metrics.normalized_mutual_info_score.html 

'''

import datetime
import glob
import os
import re
from typing import DefaultDict

import jsonlines
import numpy as np
import pandas as pd
import plotnine
from sklearn.metrics import cluster
from sklearn.metrics import normalized_mutual_info_score as nmi_score


class BSCresults(object):

    def __init__(self, data_dir, user_file_dir):

        self.data_dir = data_dir
        self.file_list_summary = glob.glob(os.path.join(self.data_dir,'*summary.csv'))
        self.file_list_users = glob.glob(os.path.join(self.data_dir,'*users.csv'))
        self.file_list_hashtags = glob.glob(os.path.join(self.data_dir,'*hashtags.csv'))
        self.user_file_dir = user_file_dir
        self.file_list_user_jsonl = glob.glob(os.path.join(self.user_file_dir, 'timeline*.jsonl'))

        assert len(self.file_list_user_jsonl)>0
        assert len(self.file_list_summary)>0
        assert len(self.file_list_users)>0
        assert len(self.file_list_hashtags)>0


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

    def set_coi(self, clusters_of_interest_list):

        if not type(clusters_of_interest_list) is list:
            raise TypeError("ensure clusters of interest provided is a list")

        self.coi = clusters_of_interest_list

    def read_data(self, min_user=10):

        self.min_user = min_user

        # subset the values of the file list to only those of min user we want.
        self.eval_list_summary = [file for file in self.file_list_summary if int(file.split('_')[-3])==min_user]
        self.eval_list_users = [file for file in self.file_list_users if int(file.split('_')[-3])==min_user]
        self.eval_list_hashtags = [file for file in self.file_list_hashtags if int(file.split('_')[-3])==min_user]

        # sort in order of cluster number
        self.eval_list_summary.sort(key=lambda x: int(x.split('_')[-2]))
        self.eval_list_users.sort(key=lambda x: int(x.split('_')[-2]))
        self.eval_list_hashtags.sort(key=lambda x: int(x.split('_')[-2]))

        self.data_summary = []
        self.data_users = []
        self.data_hashtags = []

        for file in self.eval_list_summary:
            self.data_summary.append(pd.read_csv(file, usecols = ['cluster','count']))
        for file in self.eval_list_users:
            self.data_users.append(pd.read_csv(file, usecols = ['ID','degree','topic_cluster']))
        for file in self.eval_list_hashtags:
            self.data_hashtags.append(pd.read_csv(file, usecols = ['hashtag','degree','topic_cluster']))

    def eval_nmi(self, min_user=10, shift=10):

        self.min_user=min_user
        self.read_data(self.min_user)
        self.shift=shift

        res_users = []
        for index, value in enumerate(self.data_users[shift:]):
            for i in range(1,shift):
                temp = []
                temp.append(nmi_score(value['topic_cluster'],self.data_users[index+shift-i]['topic_cluster']))
            res_users.append(np.mean(temp))

        self.user_eval_res = res_users
        arr = np.array(res_users)
        self.max_index_best_users = np.where(arr == np.amax(arr))[0][0] + shift
        self.best_cluster_users = np.max(self.data_users[self.max_index_best_users]['topic_cluster'])

        res_hashtags = []
        for index, value in enumerate(self.data_hashtags[shift:]):
            for i in range(1,shift):
                temp = []
                temp.append(nmi_score(value['topic_cluster'],self.data_hashtags[index+shift-i]['topic_cluster']))
            res_hashtags.append(np.mean(temp))

        self.hashtag_eval_res = res_hashtags
        arr = np.array(res_hashtags)
        self.max_index_best_hashtags = np.where(arr == np.amax(arr))[0][0] + shift
        self.best_cluster_hashtags = np.max(self.data_hashtags[self.max_index_best_hashtags]['topic_cluster'])

        return ((res_users, self.max_index_best_users, self.best_cluster_users),(res_hashtags, self.max_index_best_hashtags, self.best_cluster_hashtags))

    def plot(self, bins=100, verbose = True):

        start_val = int(self.file_list_users[0].split('_')[-3])+self.shift
        data_userplot = {
            'Cluster Number': list(range(start_val, len(self.data_users[self.shift:])+start_val)),
            'NMI Score': self.user_eval_res
        }
        data_userplot = pd.DataFrame(data_userplot, columns=['Cluster Number', 'NMI Score'])

        start_val = int(self.file_list_users[0].split('_')[-3])+self.shift
        data_hashplot = {
            'Cluster Number': list(range(start_val, len(self.data_hashtags[self.shift:])+start_val)),
            'NMI Score': self.hashtag_eval_res
        }
        data_hashplot = pd.DataFrame(data_hashplot, columns=['Cluster Number', 'NMI Score'])

        self.userplot = plotnine.ggplot(data_userplot) \
                    + plotnine.aes(x="Cluster Number", y="NMI Score") \
                    + plotnine.geom_line() \
                    + plotnine.labs(title = "User Clusters") 
        # self.userplot
        self.hashplot = plotnine.ggplot(data_hashplot) \
                    + plotnine.aes(x="Cluster Number", y="NMI Score") \
                    + plotnine.geom_line() \
                    + plotnine.labs(title = "Phrase Clusters")
        # self.hashplot

        self.userplot.save(
            os.path.join(self.data_dir,"bsc_user_eval.png"),
            dpi=600,
            verbose=verbose
        )
        self.hashplot.save(
            os.path.join(self.data_dir,"bsc_hashtags_eval.png"),
            dpi=600,
            verbose=verbose
        )

        # user number per cluster distribution
        self.best_summary = self.data_summary[self.max_index_best_users]
        self.best_users = self.data_users[self.max_index_best_users]
        self.best_hashtags = self.data_hashtags[self.max_index_best_users]
 
        self.userdistplot = plotnine.ggplot(self.best_summary) \
            + plotnine.aes(x="cluster", y="count") \
            + plotnine.geom_bar(stat='identity') \
            + plotnine.labs(title = "User Numbers per Cluster for best cluster, n={}".format(self.best_cluster_users))
        self.userdistplot.save(
            os.path.join(self.data_dir,"bsc_best_n_user_dist.png"),
            dpi=600,
            verbose = verbose
        )

        # histogram of user number per cluster
        self.userdisthist = plotnine.ggplot(self.best_summary) \
            + plotnine.aes(x="count") \
            + plotnine.geom_histogram(color="black", fill="white", bins=bins) \
            + plotnine.labs(title = "Histogram of Number of Users in Clusters, n={}".format(self.best_cluster_users)) \
            + plotnine.scale_x_log10()
        self.userdisthist.save(
            os.path.join(self.data_dir,"bsc_best_n_user_histplot.png"),
            dpi=600,
            verbose = verbose
        )

    def index_from_cluster_total(self, cluster_total):

        assert type(cluster_total) == int

        clusters_from_eval_list = [int(x.split('_')[-2]) for x in self.eval_list_users]

        cluster_index = clusters_from_eval_list.index(cluster_total)

        return cluster_index

    def user_jsonl_index_from_user_ID(self, user_ID):

        user_ID = int(user_ID)

        user_IDs_from_jsonl = [int(re.split('[_.]',x)[-2]) for x in self.file_list_user_jsonl]

        assert user_ID in user_IDs_from_jsonl
        user_jsonl_index = user_IDs_from_jsonl.index(user_ID)

        return user_jsonl_index

    def examine_cluster_words(self, cluster_total, cluster_num, show=20):

        assert type(cluster_total) == int
        assert type(cluster_num) == int

        # intended to be used in jupyter notebook
        cluster_words = self.data_hashtags[self.index_from_cluster_total(cluster_total)]
        cluster_words = cluster_words[cluster_words['topic_cluster']==cluster_num]

        if show == 0:
            cluster_words
        else:
            cluster_words.head(show)

        return cluster_words

    @time_function
    def get_user_cluster_data(self, cluster_total, cluster_num):

        assert type(cluster_total) == int
        assert type(cluster_num) == int

        cluster_users = self.data_users[self.index_from_cluster_total(cluster_total)]

        # get filename from user ID
        user_filenames = cluster_users[cluster_users['topic_cluster']==cluster_num]['ID']

        df = pd.DataFrame(columns=[
            'author_id',
            'tweet_id',
            'text',
            'created_at',
            'referenced_tw_1',
            'referenced_tw_2',
            'referenced_tw_3'

        ])

        for user_ID in user_filenames:

            user_jsonl_file = self.file_list_user_jsonl[self.user_jsonl_index_from_user_ID(user_ID)]

            user_df_dict = DefaultDict(list)
            # user_df_dict = {'id': [], 'text':[], 'created_at':[]}

            with jsonlines.open(user_jsonl_file) as reader:
                for tweet_jsonl in reader:
                    tweet_list_in_file = tweet_jsonl['data']
                    for tweet_data in tweet_list_in_file:
                        user_df_dict['author_id'].append(user_ID)
                        user_df_dict['tweet_id'].append(tweet_data['id'])
                        user_df_dict['text'].append(tweet_data['text'])
                        user_df_dict['created_at'].append(datetime.datetime.fromisoformat(tweet_data['created_at'][:-1]))

                        # check if there are referenced tweets
                        if 'referenced_tweets' in tweet_data:
                            max_index=-1
                            for index, reffed_tweet in enumerate(tweet_data['referenced_tweets']):
                                user_df_dict['referenced_tw_'+str(index+1)].append(reffed_tweet['id'])
                                max_index = index
                            if max_index == 2:
                                pass
                            elif max_index ==1:
                                user_df_dict['referenced_tw_3'].append(np.NaN)
                            elif max_index==0:
                                user_df_dict['referenced_tw_3'].append(np.NaN)
                                user_df_dict['referenced_tw_2'].append(np.NaN)
                            elif max_index == -1:
                                user_df_dict['referenced_tw_3'].append(np.NaN)
                                user_df_dict['referenced_tw_2'].append(np.NaN)
                                user_df_dict['referenced_tw_1'].append(np.NaN)
                        else:
                            user_df_dict['referenced_tw_3'].append(np.NaN)
                            user_df_dict['referenced_tw_2'].append(np.NaN)
                            user_df_dict['referenced_tw_1'].append(np.NaN) 

            assert len(user_df_dict['referenced_tw_1'])==len(user_df_dict['referenced_tw_2'])
            assert len(user_df_dict['referenced_tw_2'])==len(user_df_dict['referenced_tw_3'])
            assert len(user_df_dict['referenced_tw_1'])==len(user_df_dict['author_id'])

            user_df = pd.DataFrame.from_dict(user_df_dict)
            df = df.append(user_df)

        df['is_retweet'] = df.apply(lambda row: row['text'][:2]=='RT', axis=1)
        df['internal_retweet'] = df.apply(lambda row: row['referenced_tw_1'] in df['tweet_id'], axis=1)

        cluster_words = self.examine_cluster_words(cluster_total, cluster_num, show=0)

        for phrase in cluster_words['hashtag']:
            colname = 'vocab:' + phrase
            df[colname] = df.apply(lambda row: phrase in row['text'], axis=1)

        self.df = df

        return df

def main():

    pass

if __name__ == '__main__':

    main()
