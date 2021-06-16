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

import glob
import os

import pandas as pd
import numpy as np
import sklearn
import tqdm
from sklearn.metrics import normalized_mutual_info_score as nmi_score


class BSCresults(object):

    def __init__(self, data_dir):

        self.data_dir = data_dir
        self.file_list_summary = glob.glob(os.path.join(self.data_dir,'*summmary.csv'))
        self.file_list_users = glob.glob(os.path.join(self.data_dir,'*users.csv'))
        self.file_list_hashtags = glob.glob(os.path.join(self.data_dir,'*hashtags.csv'))

    def read_data(self):

        self.summary_data = []
        self.users_data = []
        self.hashtags_data = []

        for file in self.file_list_summary:
            self.summary_data.append(pd.read_csv(file))
        for file in self.file_list_users:
            self.users_data.append(pd.read_csv(file))
        for file in self.file_list_hashtags:
            self.hashtags_data.append(pd.read_csv(file))

    def eval_nmi(self):

        res = []
        for index, value in enumerate(self.users_data[11:]):
            for i in range(1,10):
                temp = []
                temp.append(nmi_score(value['topic_cluster'],self.users_data[index+11-i]['topic_cluster']))
            res.append(np.mean(temp))

        self.user_eval_res = res
        return res

def main():

    pass

if __name__ == '__main__':

    main()
