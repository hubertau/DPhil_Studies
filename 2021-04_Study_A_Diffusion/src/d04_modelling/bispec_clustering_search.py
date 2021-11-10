#!/usr/bin/python3.9

'''
Script to search through different bispectral clustering parameters. Allows for both Python and R implementations.
'''

import datetime
import glob
import os
import pickle
import pprint
import re
import subprocess
from argparse import ArgumentParser

import numpy as np
import scipy.sparse
import tqdm
from numpy.core.fromnumeric import argsort, nonzero, searchsorted
from sklearn.cluster import SpectralCoclustering
from sklearn.metrics import consensus_score
class bispec_search(object):

    def __init__(self,
        ngram_range,
        data_dir,
        output_dir,
        bsc_range,
        interval,
        min_user,
        implementation = 'Python',
        verbose = False
    ):
        self.ngram_range = ngram_range
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.bsc_range = bsc_range
        self.interval = interval
        self.min_user = min_user
        self.type = implementation.lower()
        self.verbose = verbose

        if self.verbose:
            print('Verbose output selected.')
            print('Data/Input directory is: {}'.format(self.data_dir))
            print('Output directory is: {}'.format(self.output_dir))
            print('ngram range is {}'.format(self.ngram_range))
            print('Searching over bsc cluster sizes {} with interval {} and min_user parameter {}'.format(self.bsc_range, self.interval, self.min_user))

        # Obtain csv file for r implementation
        if self.type == 'r':

            # get all available csv values
            self.csv_file = glob.glob(os.path.join(self.data_dir, 'bispec_ready_counts*.csv'))

            # extract ngram ranges
            self.csv_file = [(re.split('[_.]', i)[-2], i) for i in self.csv_file]

            # select the one that matches
            self.csv_file = [i[1] for i in self.csv_file if i[0] == self.ngram_range]

            # check there is only one
            assert len(self.csv_file) == 1

            # set to the string
            self.csv_file = self.csv_file[0]

            if self.verbose:
                print('Detected csv file: {}'.format(self.csv_file))

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

        # Check if user wants to continue. First glob output directory
        existing_files = glob.glob(os.path.join(self.output_dir, 'bsc_python_cluster*' + '_ngram_' + str(self.ngram_range[0] + self.ngram_range[1]) + '_min_' + str(self.min_user) + '.obj'))

        # User may be runing bsc on a new range, so only reduce to ones that would be overwritten with this script
        existing_files = [(re.split('[_.]',i),i) for i in existing_files]
        existing_files = [i[1] for i in existing_files if
            self.bsc_range[0] <= int(i[0][-6]) <= self.bsc_range[1] and
            i[0][-4] == str(self.ngram_range[0])+str(self.ngram_range[1]) and
            i[0][-2] == str(self.min_user)
        ]

        # prompt for input
        if len(existing_files) > 0:
            print('Files found at {}. See below:\n'.format(self.output_dir))
            pp = pprint.PrettyPrinter(indent=4)
            pp.pprint(existing_files)
            while input('\n Do you wish to continue and overwrite? [y/n]').lower() == 'y':
                continue
            else:
                print('Clustering Aborted.')
                return None

        if self.type == 'r':
            for cluster in tqdm.tqdm(range(self.bsc_range[0],self.bsc_range[1]+1,self.interval)):
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

            if self.verbose:
                print('Python implementation selected.')

            self.csr_path = os.path.join(self.data_dir, 'user_count_mat_ngram_' + self.ngram_range + '.obj')

            if self.verbose:
                print('Attempting to load in file {}'.format(self.csr_path))

            # open the saved files
            with open(self.csr_path, 'rb') as f:
                self.csr = pickle.load(f)

            if self.verbose:
                print('CSR loaded in.')

            # The adjacency matrix must be square. see Dhillon (2001) original paper.
            # self.square_mat = scipy.sparse.bmat([[None, self.csr],[self.csr.T,None]])

            if self.verbose:
                print('Filtering for min user value...')

            self.new_csr = self.csr[:, np.array(self.csr.sum(axis=0) >= self.min_user).flatten()]

            if self.verbose:
                print('Done.')

            # 2021-07-14 Check NaN
            for row in range(self.new_csr.shape[0]):
                assert np.sum(np.isnan(np.array(self.new_csr[row,:].todense()))) == 0

            if self.verbose:
                print('NaN check complete.')

            # if self.verbose:
            #     print('Converting to dense...')

            # # self.new_csr = np.array(self.new_csr.todense())
            # # self.new_csr = self.new_csr/np.max(self.new_csr)

            # if self.verbose:
            #     print('Done.')

            # The following code is from https://github.com/acgacgacgacg/biclustering/blob/master/spectral_bicluster.py
            #
            # Input A: a relational matrix
            # Output B: clustered ralational matrix of A (default k=2)
            # def spectral_bicluster(A,n_clusters = 2):
            #     m, n = A.shape
            #     D1 = np.diag(np.sum(A, axis=1))**(-0.5)
            #     D2 = np.diag(np.sum(A, axis=0))**(-0.5)
            #     An = np.dot(np.dot(D1, A), D2)
            #     U, s, V = np.linalg.svd(An, full_matrices=True)
            #     # print 'u1=', U[:,0]
            #     # print 'u2=', U[:,0]
            #     # print s
            #     z2 = np.hstack((np.dot(D1, U[:,1]), np.dot(D2, V.T[:, 1]))).reshape(m+n, 1)
            #     # print z2
            #     objKmeans = KMeans(n_clusters = n_clusters).fit(z2)
            #     mu, labels = objKmeans.cluster_centers_, objKmeans.labels_
            #     idx_d_1 = np.where(labels[:m]==0)[0]
            #     a = len(idx_d_1)
            #     idx_d_2 = np.where(labels[:m]==1)[0]
            #     print(idx_d_1, idx_d_2)

            #     idx_w_1 = np.where(labels[m:]==0)[0]
            #     b = len(idx_w_1)
            #     idx_w_2 = np.where(labels[m:]==1)[0]
            #     print(idx_w_1, idx_w_2)
            #     # print a, b
            #     B = np.vstack((A[idx_d_1], A[idx_d_2])).T
            #     B = np.vstack((B[idx_w_1], B[idx_w_2])).T

            #     # Check the Laplacian
            #     # L = np.vstack((np.hstack((np.zeros((m,m)),A)), np.hstack((A.T, np.zeros((n, n))))))
            #     L = np.vstack((np.hstack((np.diag(np.sum(A, axis=1)),-A)), np.hstack((-A.T, np.diag(np.sum(A, axis=0))))))

            #     # print e
            #     return B

            for cluster in tqdm.tqdm(range(self.bsc_range[0],self.bsc_range[1]+1,self.interval)):
                # res = spectral_bicluster(self.square_mat, n_clusters=cluster)
                model = SpectralCoclustering(n_clusters=cluster, random_state=0)

                results = model.fit(self.new_csr)
                # score = consensus_score(model.biclusters_,
                        # (rows[:, row_idx], columns[:, col_idx]))

                # print("consensus score: {:.3f}".format(score))

                # fit_data = self.new_csr[np.argsort(model.row_labels_)]
                # fit_data = self.new_csr[:, np.argsort(model.column_labels_)]

                save_filename = os.path.join(self.output_dir, 'bsc_python_cluster_' + str(cluster) + '_ngram_' + str(self.ngram_range[0] + self.ngram_range[1]) + '_min_' + str(self.min_user) + '.obj')

                with open(save_filename, 'wb') as f:
                    pickle.dump(results, f)

                if args.verbose:
                    print('saved to {}'.format(save_filename))

def main(args):

    clusterer = bispec_search(
        args.ngram_range,
        args.data_dir,
        args.output_dir,
        args.range,
        args.interval,
        args.min_user,
        implementation = args.implementation,
        verbose = args.verbose
    )

    clusterer.cluster()

if __name__ == '__main__':

    parser = ArgumentParser(description='Bispectral Cluster Search')

    parser.add_argument(
        'ngram_range',
        help='ngram range to consider. Enter in format e.g. 23 for range (2,3)'
    )

    parser.add_argument(
        'data_dir',
        help='Data directory'
    )

    parser.add_argument(
        '--output_dir',
        help='output directory. defaults to data_dir location.'
    )

    parser.add_argument(
        '--range',
        help='range of clusters to work scan through. In format [low]-[high]',
        default='10-20',
        type=str
    )

    parser.add_argument(
        '--interval',
        help='step value within range of search.',
        default=1,
        type=int
    )

    parser.add_argument(
        '--min_user',
        help='count for which a user must have to participate in clustering',
        default='10'
    )

    parser.add_argument(
        '--implementation',
        help='implementation to use. either python or R',
        default='R'
    )

    parser.add_argument(
        '--verbose',
        help='verbosity parameter',
        default=False,
        action='store_true'
    )

    args = parser.parse_args()

    args.range = args.range.split('-')
    args.range = (int(args.range[0]),int(args.range[1]))
    args.interval = int(args.interval)
    args.min_user = int(args.min_user)

    if not args.output_dir:
        args.output_dir = args.data_dir

    main(args)
