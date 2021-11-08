#!/usr/bin/python3.9

'''
Script to search through different bispectral clustering parameters.
'''

from argparse import ArgumentParser
import datetime
import pickle
import subprocess

import numpy as np
import numpy as np
import scipy.sparse
import os
import tqdm
from numpy.core.fromnumeric import argsort, nonzero, searchsorted
from sklearn.cluster import KMeans, SpectralCoclustering
from sklearn.metrics import consensus_score

# from generate_user_to_hashtag_matrix import TweetVocabVectorizer


class bispec_search(object):

    def __init__(self, csv_file, output_dir, range, interval, min_user, csr=None, vectorizer=None, mapping=None, implementation = 'Python'):
        self.csv_file = csv_file
        self.output_dir = output_dir
        self.range = range
        self.interval = interval
        self.min_user = min_user
        self.type = implementation.lower()

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

    def cluster(self):

        if self.type == 'r':
            for cluster in tqdm.tqdm(range(self.range[0],self.range[1]+1,self.interval)):
                subprocess.run(
                    [
                        'Rscript',
                        'bispectral_clustering.R',
                        str(self.csv_file),
                        str(self.output_dir),
                        '--min_user',
                        str(self.min_user),
                        '--ncluster',
                        str(cluster)
                    ]
                )

        elif self.type == 'python':

            # open the saved files
            with open('collection_results_2021_05_04_16_22/user_count_mat.obj', 'rb') as f:
                csr = pickle.load(f)
            with open('collection_results_2021_05_04_16_22/vectorizer.obj', 'rb') as f:
                vectorizer = pickle.load(f)
            with open('collection_results_2021_05_04_16_22/mapping.obj', 'rb') as f:
                mapping = pickle.load(f)


            # The adjacency matrix must be square. see Dhillon (2001) original paper.
            # self.square_mat = scipy.sparse.bmat([[None, self.csr],[self.csr.T,None]])

            self.new_csr = self.csr[:, np.array(self.csr.sum(axis=0) >= self.params['min_user']).flatten()]

            # 2021-07-14 Check NaN
            for row in range(self.new_csr.shape[0]):
                assert np.sum(np.isnan(np.array(self.new_csr[row,:].todense()))) == 0

            self.new_csr = np.array(self.new_csr.todense())
            # self.new_csr = self.new_csr/np.max(self.new_csr)

            # The following code is from https://github.com/acgacgacgacg/biclustering/blob/master/spectral_bicluster.py
            #
            # Input A: a relational matrix
            # Output B: clustered ralational matrix of A (default k=2)
            def spectral_bicluster(A,n_clusters = 2):
                m, n = A.shape
                D1 = np.diag(np.sum(A, axis=1))**(-0.5)
                D2 = np.diag(np.sum(A, axis=0))**(-0.5)
                An = np.dot(np.dot(D1, A), D2)
                U, s, V = np.linalg.svd(An, full_matrices=True)
                # print 'u1=', U[:,0]
                # print 'u2=', U[:,0]
                # print s
                z2 = np.hstack((np.dot(D1, U[:,1]), np.dot(D2, V.T[:, 1]))).reshape(m+n, 1)
                # print z2
                objKmeans = KMeans(n_clusters = n_clusters).fit(z2)
                mu, labels = objKmeans.cluster_centers_, objKmeans.labels_
                idx_d_1 = np.where(labels[:m]==0)[0]
                a = len(idx_d_1)
                idx_d_2 = np.where(labels[:m]==1)[0]
                print(idx_d_1, idx_d_2)

                idx_w_1 = np.where(labels[m:]==0)[0]
                b = len(idx_w_1)
                idx_w_2 = np.where(labels[m:]==1)[0]
                print(idx_w_1, idx_w_2)
                # print a, b
                B = np.vstack((A[idx_d_1], A[idx_d_2])).T
                B = np.vstack((B[idx_w_1], B[idx_w_2])).T

                # Check the Laplacian
                # L = np.vstack((np.hstack((np.zeros((m,m)),A)), np.hstack((A.T, np.zeros((n, n))))))
                L = np.vstack((np.hstack((np.diag(np.sum(A, axis=1)),-A)), np.hstack((-A.T, np.diag(np.sum(A, axis=0))))))

                # print e
                return B

            for cluster in tqdm.tqdm(range(self.params['range'][0],self.params['range'][1]+1,self.params['interval'])):
                # res = spectral_bicluster(self.square_mat, n_clusters=cluster)
                model = SpectralCoclustering(n_clusters=cluster, random_state=0)

                #preprocess csr

                model.fit(self.new_csr)
                # score = consensus_score(model.biclusters_,
                        # (rows[:, row_idx], columns[:, col_idx]))

                # print("consensus score: {:.3f}".format(score))

                fit_data = self.new_csr[np.argsort(model.row_labels_)]
                fit_data = self.new_csr[:, np.argsort(model.column_labels_)]
                save_filename = '/Users/hubert/Nextcloud/DPhil/DPhil_Studies/2021-04_Study_A_Diffusion/collection_results_2021_05_04_16_22/bsc/python_cluster_' + self.params['min_user'] + '_' + str(cluster) + '.obj'
                with open(save_filename, 'wb') as f:
                    pickle.dump(fit_data, f)

def main(args):

        clusterer = bispec_search(
            args.csv_file,
            args.output_dir,
            args.range,
            args.interval,
            args.min_user,
            implementation = args.implementation
        )

        clusterer.cluster()

if __name__ == '__main__':

    parser = ArgumentParser(description='Bispectral Cluster Search')

    parser.add_argument(
        'csv_file',
        help='csv file in the right format for R bispec clustering'
    )

    parser.add_argument(
        '--output_dir',
        help='output directory. defaults to csv_file location.'
    )

    parser.add_argument(
        '--implementation',
        help='implementation to use. either python or R',
        default='R'
    )

    parser.add_argument(
        '--range',
        help='range of clusters to work scan through. In format [low]-[high]',
        default='10-20'
    )

    parser.add_argument(
        '--interval',
        help='step value within range of search.',
        default='1'
    )

    parser.add_argument(
        '--min_user',
        help='count for which a user must have to participate in clustering',
        default='10'
    )

    args = parser.parse_args()

    args.range = args.range.split('-')
    args.range = (int(args.range[0]),int(args.range[1]))
    args.interval = int(args.interval)
    args.min_user = int(args.min_user)

    if not args.output_dir:
        args.output_dir = os.path.split(args.csv_file)[0]

    main(args)
