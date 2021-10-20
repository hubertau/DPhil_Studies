#!/usr/bin/python3.9

import os
import argparse
import pickle
import glob
from networkx.algorithms.shortest_paths.generic import average_shortest_path_length
import tqdm
from collections import Counter
import networkx as nx
import urllib



def main(args):

    user_edges_pickle_file = '../../data/02_intermediate/user_sets.obj'
    timeline_flist = glob.glob('../../data/01_raw/timeline*.jsonl')
    graph_object_file = '../../data/03_processed/rt_graph.obj'

    if os.path.isfile(user_edges_pickle_file):
        with open(user_edges_pickle_file, 'rb') as f:
            user_edges = pickle.load(f)
    else:
        print('file not found at {}. Please check setup.'.format(user_edges_pickle_file))
        return None

    with open(args.user_list_file, 'r') as f:
        p = f.readlines()
        p = [int(i.replace('\n','')) for i in p]
        user_list = p

    if args.overwrite or not os.path.isfile(graph_object_file):

        print('overwriting or creating for the first time')

        # construct graph
        G = nx.DiGraph()

        for i in user_edges.keys():
            assert int(i) in user_list

        # assert len(user_list) == len(user_edges.keys())

        user_edges_keys = user_edges.keys()

        for k,v in tqdm.tqdm(user_edges.items()):

            # add k
            G.add_node(str(k))

            v_counter = Counter(v)
            v_edge_list = [(str(k), str(node), count) for node,count in v_counter.items() if str(node) in user_edges_keys]

            G.add_weighted_edges_from(v_edge_list)

        with open(graph_object_file, 'wb') as f:
            pickle.dump(G, f)

        print('result saved at {}'.format(graph_object_file))

    else:
        print('not overwriting or creating. Reading in.')
        with open(graph_object_file, 'rb') as f:
            G = pickle.load(f)



    number_of_nodes = len(G.nodes())
    number_of_edges = len(G.edges())
    G_connected_components = nx.connected_components(G.to_undirected())
    density = nx.density(G)
    average_local_clustering_coeff = nx.average_clustering(G)
    strongly_connected = nx.is_strongly_connected(G)
    weakly_connected = nx.is_weakly_connected(G)
    if strongly_connected:
        diameter = nx.diameter(G)
    else:
        diameter = 'N/A'
    if weakly_connected:
        average_shortest_path_length = nx.average_shortest_path_length(G)
    else:
        average_shortest_path_length = 'N/A'
    transitivity = nx.transitivity(G)

    print('Some Basic Stats: \n')
    print('Number of nodes: {}'.format(number_of_nodes))
    print('Number of edges: {}'.format(number_of_edges))
    print('Number of connected components: {}'.format(len(list(G_connected_components))))
    print('Density: {}'.format(density))
    print('Average local clustering coefficient: {}'.format(average_local_clustering_coeff))
    print('Is strongly connected? {}'.format(strongly_connected))
    print('N.B. A directed graph is strongly connected if for every pair of nodes u and v, there is a directed path from u to v and v to u.')
    print('Is weakly connected? {}'.format(weakly_connected))
    print('N.B. It is weakly connected if replacing all the edges of the directed graph with undirected edges will produce a Undirected Connected Graph.')
    print('Diameter of graph: {} <- the maximum shortest distance between a pair of nodes'.format(diameter))
    print('Average shortest path length: {}'.format(average_shortest_path_length))
    print('Transitivity: {} <- the '.format(transitivity))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Script to generate the basic timeline stats from the users.')

    parser.add_argument(
        'user_list_file',
        help = 'which user_list file to use.'
    )

    parser.add_argument(
        '--overwrite',
        help = ''
    )

    args = parser.parse_args()

    main(args)
