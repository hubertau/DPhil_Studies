#!/usr/bin/python3.9

import argparse
import datetime
import json
import logging
import os
from itertools import repeat
import pickle
import random
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from typing import NamedTuple

import h5py
import networkx as nx
import jsonlines
import numpy as np
import pandas as pd
import tqdm
from sklearn.model_selection import ParameterGrid

class Interaction_Record(NamedTuple):
    source: str
    target: str
    time: int
    experimentation_success: bool
    interact_result : bool

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def load_in_search_ht(search_hashtags_path):
    #load in search hashtags
    with open(search_hashtags_path, 'r') as f:
        search_hashtags = f.readlines()
        search_hashtags = [i.replace('\n', '') for i in search_hashtags]
        search_hashtags = [i.replace('#', '') for i in search_hashtags]
        search_hashtags = [i.lower() for i in search_hashtags]
        search_hashtags.remove('وأناكمان')

    return search_hashtags

################################################################################
# Define useful functions
################################################################################

def unit_conv(val):
    return datetime.datetime.strptime('2017-10-16', '%Y-%m-%d') + datetime.timedelta(days=int(val))

def reverse_unit_conv(date):
    return (datetime.datetime.strptime(date, '%Y-%m-%d') - datetime.datetime.strptime('2017-10-16', '%Y-%m-%d')).days

class daterange(NamedTuple):
    start: str
    end: str

def date_to_array_index(date, daterange):
    return (datetime.datetime.strptime(date, '%Y-%m-%d') - datetime.datetime.strptime(daterange.start, '%Y-%m-%d')).days

def group_peaks_and_daterange(peak_analysis_file, group_num):

    #obtain peak times again
    with h5py.File(peak_analysis_file, 'r') as f:
        FAS_peaks = f['peak_detections']
        x = f['segments']['selected_ranges'][int(group_num)-1]
        group_date_range = daterange(
            start = x[0].decode(),
            end = x[1].decode()
        )

        # group_start_index = reverse_unit_conv(group_date_range.start)
        # group_end_index = reverse_unit_conv(group_date_range.end)

        most_prominent_peaks = {}
        for name, h5obj in FAS_peaks.items():

            peak_locations = h5obj['peak_locations']
            peak_locations = [(i,e) for i,e in enumerate(h5obj['peak_locations']) if (unit_conv(e) > datetime.datetime.strptime(group_date_range.start, '%Y-%m-%d')) and (unit_conv(e) < datetime.datetime.strptime(group_date_range.end, '%Y-%m-%d'))]
            peak_indices = [i[0] for i in peak_locations]
            prominences = [element for index, element in enumerate(h5obj['prominences']) if index in peak_indices]
            if len(prominences) == 0:
                continue
            max_prominence = np.argmax(prominences)
            most_prominent_peaks[name] = unit_conv(peak_locations[max_prominence][1])

    daterange_length = (datetime.datetime.strptime(group_date_range.end, '%Y-%m-%d') - datetime.datetime.strptime(group_date_range.start, '%Y-%m-%d')).days

    return most_prominent_peaks, group_date_range, daterange_length

def generate_network(df, follows_dir):

    # construct graph
    G = nx.DiGraph()

    # add nodes to graph
    unique_users = df['author_id'].unique()

    # for the most common hasthags used by an interacted user
    filtered_ht = df.groupby('author_id')[['ht','gender','age','int_pre_peak','act_pre_peak','norm_act_pre_peak','org','lang', 'primary_ht']].agg(pd.Series.mode)

    assert os.path.isdir(follows_dir)

    successful_follows=0
    # extract users whom a user follows and following
    for user_id in tqdm.tqdm(unique_users):

        # add user to graph as a node with attributes.
        G.add_node(
            user_id,
            gender       = filtered_ht.loc[user_id]['gender'],
            age          = filtered_ht.loc[user_id]['age'],
            int_pre_peak = filtered_ht.loc[user_id]['int_pre_peak'],
            act_pre_peak = filtered_ht.loc[user_id]['act_pre_peak'],
            norm_act_pre_peak = filtered_ht.loc[user_id]['norm_act_pre_peak'],
            org               = filtered_ht.loc[user_id]['org'],
            lang              = filtered_ht.loc[user_id]['lang'],
            primary_ht        = filtered_ht.loc[user_id]['primary_ht']
        )

        # logging.info(f'Processing {user_id}')
        follows_filepath = os.path.join(follows_dir, f'following_{user_id}.txt')
        try:
            total_edges_to_add = df[df['author_id'] == user_id]['interacted_users'].sum()
        except:
            logging.info(user_id)
        if isinstance(total_edges_to_add, int):
            total_edges_to_add = np.array([])
        total_edges_to_add = np.intersect1d(total_edges_to_add, unique_users)

        if os.path.isfile(follows_filepath):
            try:
                edges_to = pd.read_table(follows_filepath).values.flatten().astype(str)
                successful_follows+=1
                new_total_edges_to_add = np.union1d(total_edges_to_add, edges_to)
                new_total_edges_to_add = np.intersect1d(new_total_edges_to_add, unique_users)
                length_diff = len(new_total_edges_to_add) - len(total_edges_to_add)
                assert length_diff >= 0
                # logging.info(f'Length diff: {length_diff} for {user_id}')
                total_edges_to_add = new_total_edges_to_add
            except pd.errors.EmptyDataError:
                pass

        for interacted_user in total_edges_to_add:
            ht = filtered_ht.loc[interacted_user]['ht']
            G.add_edge(user_id, interacted_user, ht=ht)

    logging.info(f'Graph Generation: Total successfully read follows: {successful_follows}')

    return G

class Agent(object): 

    # initialise internal variables
    def __init__(self, ID, df, search_hashtags):
        self.ID = ID
        self.search_hashtags = search_hashtags
        self.supporting_metoo = False    # initial assumption.
        self.supporting_metoo_dict = {i:0 for i in search_hashtags}
        self.interacts_with   = []       # I wrote this to represent a symmetric interaction, rather than following.
        # self.interaction_counter = defaultdict(int)    # counts interactions
        self.forget_all_interactions()
        # logging.info(ID)
        # row = df[df['author_id'] == ID].iloc[0,:]
        # self.total_hashtags = row.author_total_hashtags
        self.primary_ht = df.loc[ID,'primary_ht']
        self.support_tracker = np.zeros(shape=(35,1))
        self.activity_tracker = np.zeros(shape=(35,1))
        self.individual_propensity = np.zeros(shape=(35,1))
        self.experimentation_count = 0


    def update_tracker(self):

        support_update_array = np.array(list(self.supporting_metoo_dict.values())).reshape(-1,1)

        self.support_tracker = np.hstack((self.support_tracker, support_update_array))

    def simulate(self, search_hashtag_propensity):

        keys = list(self.supporting_metoo_dict.keys())
        propensities = np.array([search_hashtag_propensity[i] for i in keys])

        self.probability_matrix = 1-np.power(1-propensities.reshape(-1,1),self.support_tracker)

        self.simulated = np.random.binomial(1, self.probability_matrix)

    def interact(self, other, experimentation_success_chance):

        # Keep track of interactions with others.
        # This step is asymmetric: 'self' keeps track, but 'other' does not.

        if other.ID not in self.interaction_counter:
            self.interaction_counter[other.ID]  = 1
        else:
            self.interaction_counter[other.ID] += 1

        # For later models implementing likes
        experimentation_trial = np.random.uniform()
        if experimentation_trial <= experimentation_success_chance:
            self.experimentation_success = True
        else:
            self.experimentation_success = False
        self.experimentation_count += self.experimentation_success

    # How do they behave?
    # How do agents change?
    def maybe_join(self, other, interact_threshold = 1, model_num = None, verbose=False):


        # Model num. This is just for ease of reproducibility. model_num being None means take the latest (probably most complex) ABM.

        if model_num == 1:

            # Simplest model. No 
            if self.supporting_metoo == False and \
                other.supporting_metoo == True and \
                self.interaction_counter[other.ID] > interact_threshold:

                self.supporting_metoo = True

                if verbose:
                    logging.info(f'Agent {self.ID} has spoken a lot to Agent {other.ID} and now supports {other.primary_ht}')

                return True, other.primary_ht

        elif model_num == 2:

            # Model 2:
            # still have singular 'metoo' supporting
            # ADD different primary language requirement.
            if  self.supporting_metoo == False and \
                other.supporting_metoo == True and \
                other.primary_ht != self.primary_ht and \
                self.interaction_counter[other.ID] > interact_threshold:

                self.supporting_metoo = True

                if verbose:
                    logging.info(f'Agent {self.ID} has spoken a lot to Agent {other.ID} and now supports {other.primary_ht}')

                return True, other.primary_ht

        elif model_num == 3:

            # Model 3:
            # still have singular 'metoo' supporting
            # and different language requirement.
            # ADD minimum reciprocal interaction
            if  self.supporting_metoo == False and \
                other.supporting_metoo == True and \
                other.primary_ht != self.primary_ht and \
                self.interaction_counter[other.ID] > interact_threshold and \
                other.interaction_counter[self.ID] > interact_threshold:

                self.supporting_metoo = True

                if verbose:
                    logging.info(f'Agent {self.ID} has spoken a lot to Agent {other.ID} and now supports {other.primary_ht}')

                return True, other.primary_ht

        elif model_num == 4:

            # Model 4:
            # ADD metoodict
            # and different language requirement.
            # and minimum reciprocal interaction.
            #
            # Now there is the possibility of those who are 'supporting metoo' already in one language to be influenced to support another language.
            #
            # This model only allows for one user to influence another on their primary hashtag.

            # if  (self.supporting_metoo is False) and \
            if  (self.interaction_counter[other.ID] > interact_threshold) and \
                (other.primary_ht != self.primary_ht) and \
                (self.supporting_metoo_dict[other.primary_ht] == 0) and \
                (other.interaction_counter[self.ID] > interact_threshold):

                self.supporting_metoo_dict[other.primary_ht] += 1

                if verbose:
                    logging.info(f'Agent {self.ID} has spoken a lot to Agent {other.ID} and now supports {other.primary_ht}')

                return True, other.primary_ht

        elif model_num == 5:

            # Model 5:
            # use metoodict
            # and different language requirement.
            # and minimum reciprocal interaction.
            #
            # ADD experimentation success
            #
            # This model only allows for one user to influence another on their primary hashtag.

            # if  (self.supporting_metoo is False) and \
            if  (self.interaction_counter[other.ID] > interact_threshold) and \
                (other.primary_ht != self.primary_ht) and \
                (self.supporting_metoo_dict[other.primary_ht] == 0) and \
                (other.interaction_counter[self.ID] > interact_threshold or self.experimentation_success):

                self.supporting_metoo_dict[other.primary_ht] += 1

                if verbose:
                    logging.info(f'Agent {self.ID} has spoken a lot to Agent {other.ID} and now supports {other.primary_ht}')

                return True, other.primary_ht

        elif model_num == 6:

            # Model 6:
            # use metoodict
            # and different language requirement.
            # and minimum reciprocal interaction.
            #
            # ADD ability to influence other users within your community too.
            #
            # This model only allows for one user to influence another on their primary hashtag.

            # if  (self.supporting_metoo is False) and \
            if  (self.interaction_counter[other.ID] > interact_threshold) and \
                (other.primary_ht != self.primary_ht) and \
                (self.supporting_metoo_dict[other.primary_ht] == 0) and \
                (other.interaction_counter[self.ID] > interact_threshold):

                self.supporting_metoo_dict[other.primary_ht] += 1

                if verbose:
                    logging.info(f'Agent {self.ID} has spoken a lot to Agent {other.ID} and now supports {other.primary_ht}')

                return True, other.primary_ht

            elif (self.interaction_counter[other.ID] > interact_threshold) and \
                 (other.primary_ht == self.primary_ht) and \
                 (other.interaction_counter[self.ID] > interact_threshold):

                samplelist = [k for k,v in other.supporting_metoo_dict.items() if v>0]

                if samplelist:
                    # select a random value within
                    sampled_ht_for_influence = random.choices(
                        samplelist,
                        [v for _,v in other.supporting_metoo_dict.items() if v>0],
                        k=1
                    )

                    # random.choices returns a list so take first index
                    self.supporting_metoo_dict[sampled_ht_for_influence[0]] += 1

                if verbose:
                    logging.info(f'Agent {self.ID} has influenced someone of their own primary ht community.')
                return True, sampled_ht_for_influence[0]

        elif model_num is None:
            pass

        return False, None

    def forget_all_interactions(self):
        self.interaction_counter = defaultdict(int)

    def forget_support_metoo(self):
        self.supporting_metoo = False
        self.supporting_metoo_dict = {i:0 for i in self.search_hashtags}

def produce_agents(df, search_hashtags):
    temp_df = df.groupby('author_id').first()
    agents = {}
    for user in list(df['author_id'].unique()):
        agents[user] = Agent(user,temp_df, search_hashtags)

    return agents

def reset_abm(
        args,
        agents,
        initial_activity_threshold,
        search_hashtag_propensity,
        peak_delta_init = 10
    ):

    # First, reset everyone's memory of their interactions
    for user_id, agent in agents.items():
        agent.forget_all_interactions()
        agent.forget_support_metoo()
        agent.propensity_params = search_hashtag_propensity


    group_start_date = datetime.datetime.strptime(args.group_date_range.start, '%Y-%m-%d')

    # Alternate second step: activate everyone above a certain activity threshold before the daterange. This activation will be done by primary language they have expressed some 
    # pre_val = {}
    with h5py.File(args.activity_file, 'r') as f:

        # read in feature order before iteration. More efficient.
        feature_order = f[f'group_{args.group_num}'][user_id]['hashtagged'].attrs['feature_order']
        feature_order = feature_order.split(';')

        activity_base = f[f'group_{args.group_num}']
        for user_id, agent in agents.items():

            # obtain user activity
            activity = activity_base[user_id]['hashtagged'][:]

            # obtain values for the hashtags that have peaks in this time period
            for hashtag_in_period in args.most_prominent_peaks:
                hashtag_in_period_index = feature_order.index(hashtag_in_period)

                # obtain the index offset from the detected peak of the hashtag to collect initial time window.
                offset_index = args.most_prominent_peaks[hashtag_in_period] - group_start_date
                offset_index = offset_index.days
                offset_index -= peak_delta_init
                offset_index = max(0,offset_index)+1
                # logging.info(f'Offset for {hashtag_in_period} is {offset_index}')

                if np.sum(activity[hashtag_in_period_index,:offset_index]) > initial_activity_threshold:
                    agent.supporting_metoo = True
                    agent.supporting_metoo_dict[hashtag_in_period] = True

    return None

def run_model(
        agents,
        params,
        verbose = False
    ):

    # 2022-06-09: using this Generator is faster than np.random directly apparently. cf. https://github.com/numpy/numpy/issues/2764
    gen = np.random.default_rng()

    # Create the transaction history object:
    # history = []
    history = nx.DiGraph()
    if args.history_logging:
        logging.info(f'Populating history graph...')
        for _, agent in agents.items():
            history.add_node(agent.ID, primary_ht = agent.primary_ht)


    for time in range(args.daterange_length):
        if verbose:
            logging.info(f'Starting interactions on day {time+1}')
        for _, agent in agents.items():

            # pick a random person that the agent interacts with
            try:

                other_agent = agents[gen.choice(agent.interacts_with)]

                # by default, no interaction occurs.
                interact_result = None

                # interact with them
                if gen.uniform()<=(params['interact_prob'])*params['interact_prob_multiplier']**(agent.interaction_counter[other_agent.ID]):
                    agent.interact(other_agent, params['experimentation_chance'])

                    # if you've interacted with them many times recently, say something
                    interact_result, interact_ht = agent.maybe_join(
                        other_agent,
                        params['interact_threshold'],
                        model_num = params['model_num'],
                        verbose=verbose)

                if args.history_logging:
                    # history.append(Interaction_Record(agent.ID, other_agent.ID, time, agent.experimentation_success, interact_result))
                    history.add_edge(other_agent.ID, agent.ID, interact_result = interact_result, time = time, experimentation_success=agent.experimentation_sucess, ht=interact_ht)

                agent.update_tracker()

            except ValueError:

                agent.update_tracker()
                continue

    return (agents, history)

def reset_and_run_model(args, agents, current_params_tuple):

    index          = current_params_tuple[0]
    current_params = current_params_tuple[1]

    logging.info(f'RUNNING NUMBER {index}')

    current_search_hashtag_propensity = {k: current_params['search_hashtag_propensity'] for k, _ in args.search_hashtag_propensity_base.items()}

    logging.info(f'BEGIN for param set: {current_params}')
    logging.info('Setting initial support for agents for ABM:')
    _ = reset_abm(args, agents,
        initial_activity_threshold=current_params['initial_activity_threshold'],
        search_hashtag_propensity=current_search_hashtag_propensity,
        peak_delta_init=current_params['peak_delta_init'])

    logging.info(f'Running model {index}...')
    modelled_agents, history = run_model(
            agents,
            current_params,
            verbose = False
        )
    logging.info(f'Modelling complete for {index}.')

    if args.simulate:
        logging.info(f'Simulating {index}...')
        for _, agent in modelled_agents.items():
            agent.simulate(current_search_hashtag_propensity)
        logging.info(f'Simulating {index} complete.')
    else:
        logging.info(f'Simulating OFF. Ending...')

    return (current_params, modelled_agents, history)

def main(args):


    ############################################################################
    # Read in peaks
    ############################################################################

    args.most_prominent_peaks, args.group_date_range, args.daterange_length = group_peaks_and_daterange(args.peak_analysis_file, args.group_num)

    ############################################################################
    # Determine if ABM df has already been processed.
    ############################################################################

    if (os.path.isfile(args.abm_processed_df_savepath) and args.overwrite) or not os.path.isfile(args.abm_processed_df_savepath):

        logging.info('overwriting or writing for the first time')

        # read df raw for ABM
        stats_df_save_dir = '/home/hubert/DPhil_Studies/2021-04_Study_A_Diffusion/data/06_reporting/'
        df_filename = os.path.join(stats_df_save_dir, f'ABM_raw_df_group_{args.group_num}.obj')
        stats_filename = os.path.join(stats_df_save_dir, f'ABM_stats_df_group_{args.group_num}.obj')

        if os.path.isfile(df_filename):
            logging.info('reading in df')
            with open(df_filename, 'rb') as f:
                df = pickle.load(f)
        if os.path.isfile(stats_filename):
            logging.info('reading in stats_df')
            with open(stats_filename, 'rb') as f:
                stats_df = pickle.load(f)

        logging.info('N.B. users are not included in stats df because in creating the activity counts users were split into before and after peak interactions')

        logging.info(f'Length of df: {len(df)}')
        unique_author_stats_df_count = len(stats_df['author_id'].unique())
        logging.info(f'Number of unique authors in stats df: {unique_author_stats_df_count}')
        unique_author_df_count = len(df['author_id'].unique())
        logging.info(f'Number of unique authors in df: {unique_author_df_count}')

        # generate ht column
        df_colnames = df.columns
        vocab_colnames = [i for i in df_colnames if i.startswith('vocab')][::-1]
        def process_row_ht(row):
            for col in vocab_colnames:
                if col == 'vocab:#timesup':
                    continue
                if row[col] == 1:
                    return col.split('#')[-1]
            return None

        df['ht'] = df.apply(process_row_ht, axis=1)
        df['ht'] = df['ht'].fillna('metoo')

        df = df.merge(stats_df, on=['author_id','ht'], how='right')

        # incorporate primary ht
        with open(f'/home/hubert/DPhil_Studies/2021-04_Study_A_Diffusion/data/03_processed/primary_ht_global.obj', 'rb') as f:
            user_order, res = pickle.load(f)

        unknown_count = 0
        def process_primary_res(author_id):
            global unknown_count
            if author_id not in user_order:
                # logging.info(f'{author_id} not in users')
                unknown_count += 1
                return 'metoo'
            return args.search_hashtags[np.argmax(res[user_order.index(author_id),:])]


        df['primary_ht'] = df['author_id'].map(df.groupby('author_id').apply(lambda x: process_primary_res(x.name)))
        logging.info(f'Number of unknown primary hashtags for users: {unknown_count}')

        df['interacted_users'].loc[df['interacted_users'].isnull()] = df['interacted_users'].loc[df['interacted_users'].isnull()].apply(lambda x: [])

        with open(args.abm_processed_df_savepath, 'wb') as f:
            pickle.dump(df, f)


    elif os.path.isfile(args.abm_processed_df_savepath) and args.read_in:
        logging.info('reading in')
        with open(args.abm_processed_df_savepath, 'rb') as f:
            df = pickle.load(f)


    ############################################################################
    # Generate the df for authors later
    ############################################################################

    # temp_df = df.groupby('author_id').first()

    ############################################################################
    # Generate graph of users
    ############################################################################

    if os.path.isfile(args.graph_savepath):
        logging.info('Graph Generation: loading in')
        with open(args.graph_savepath, 'rb') as f:
            G = pickle.load(f)
    else:
        logging.info('Graph Generation: generating network')
        G = generate_network(df, args.follows_dir)
        with open(args.graph_savepath, 'wb') as f:
            pickle.dump(G,f)

    ############################################################################
    # Generate Agents
    ############################################################################
    agents_overwrite = True
    agents_read_in = True

    if os.path.isfile(args.agents_savepath) and agents_overwrite:
        logging.info('Agent Creation: File exists and overwriting')
        agents = produce_agents(df, args.search_hashtags)
        # with open(args.agents_savepath, 'wb') as f:
            # pickle.dump(agents, f)
    elif os.path.isfile(args.agents_savepath) and agents_read_in:
        logging.info('Agent Creation: reading in')
        with open(args.agents_savepath, 'rb') as f:
            agents = pickle.load(f)
    elif not os.path.isfile(args.agents_savepath):
        logging.info('Agent Creation: producing agents for the first time')
        agents = produce_agents(df, args.search_hashtags)
        # with open(args.agents_savepath, 'wb') as f:
            # pickle.dump(agents, f)

    logging.info('Agent Creation: Complete.')

    for _, agent in agents.items():
        agent.forget_all_interactions()
        agent.forget_support_metoo()
    for edge in G.edges():
        i,j = edge
        agents[i].interacts_with.append(j)

    ############################################################################
    # Run ABM
    ############################################################################

    if args.line_profiler:
        for param_tuple in enumerate(args.param_grid):
            modelled_agents = reset_and_run_model(args, agents, param_tuple)
    else:
        with ProcessPoolExecutor(max_workers = args.max_workers) as executor:
            logging.info(f'Beginning ProcessPool Executor with {args.max_workers} workers.')

            ########################################################################
            # Reset ABM
            ########################################################################

            results = executor.map(reset_and_run_model, repeat(args), repeat(agents), enumerate(args.param_grid))

        logging.info(f'Attempting to save at {args.model_output_savepath}')

        if args.history_logging:
            full_history = []
        with h5py.File(args.model_output_savepath, 'a') as f:

            # dimensions of output array: batch_size * num_of_agents * hashtags_to_support * days_to_model
            batch_result_array = np.zeros(shape = (args.batch_size, len(agents), len(args.search_hashtags), args.daterange_length+1))

            # create dataset
            batch_array_hdf5 = f.create_dataset('batch_result',batch_result_array.shape, compression = 'gzip', compression_opts=9)

            logging.debug(f'shape of BATCH RESULT output array: {batch_result_array.shape}')

            # create corresponding params dataset
            params_result_array = np.zeros(shape = (args.batch_size, len(args.param_grid[0])))

            # give the param order as an attribute of the params array
            params_array_hdf5 = f.create_dataset('params_array', params_result_array.shape)
            params_array_hdf5.attrs['param_order'] = str(list(args.param_grid[0].keys()))
            logging.debug(f'shape of PARAM output array: {params_array_hdf5.shape}')

            # now iterate over the results in the batch, this is a length of batch_size
            logging.debug(f'begin iterating over batch results:')
            for counter, current_results in enumerate(results):

                # expand tuple of current results
                current_params, modelled_agents, history = current_results

                # assign agent order to its own dataset
                if counter == 0:
                    f.create_dataset('agent_order', data=list(modelled_agents.keys()))

                # keep track of full history
                if args.history_logging:
                    full_history.append((current_params, history))

                # update params results coutner with the current set of param values
                params_result_array[counter] = np.array(list(current_params.values()))

                # iterate over agents and assign the right slice to batch_result_array
                for inside_counter, current_agent_item in enumerate(modelled_agents.items()):

                    _, agent = current_agent_item

                    if inside_counter == 0:
                        batch_array_hdf5.attrs['key_order'] = str(list(agent.supporting_metoo_dict.keys()))

                    batch_result_array[counter, inside_counter] = agent.support_tracker

            # finally, assign the arrays to the hdf5 objects.
            logging.debug(f'assigning arrays to respective hdf5 outputs...')
            batch_array_hdf5[:] = batch_result_array
            params_array_hdf5[:] = params_result_array

        logging.info(f'Writing results to hdf5 complete.')


        if args.history_logging:
            logging.info(f'Complete. Writing to history file...')
            # with jsonlines.open(args.model_output_jsonl) as writer:
            #     for h in history:
            #         writer.write([i.to_dict()])
            with open(args.model_output_history, 'wb') as f:
                pickle.dump(full_history, f)


    # logging.info(f'Saving to {args.model_output_savepath}')
    # with open(args.model_output_savepath, 'wb') as f:
    #     pickle.dump(modelled_agents, f)

    # with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
    #     logging.info(f'Beginning ProcessPool Executor with {args.max_workers} workers.')
    #     results = executor.map(process_one_file_pair, zip(timeline_file_list, augmented_file_list, repeat(user_list)))

    # flatten list of lists
    # results = [item for sublist in results for item in sublist if item is not None]

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Agent-Based Modelling Script.')

    parser.add_argument(
        '--search_hashtags',
        help='txt file with search hashtags'
    )

    parser.add_argument(
        '--group_num',
        help='group number'
    )

    parser.add_argument(
        '--overwrite',
        help='Overwrite files?',
        default=False,
        action='store_true'
    )

    parser.add_argument(
        '--read_in',
        help='Whether to read in data',
        default=True
    )

    parser.add_argument(
        '--data_path',
        help='Data path where abm df can be found.'
    )

    parser.add_argument(
        '--activity_file'
    )

    parser.add_argument(
        '--history_logging',
        help='whether to log each interaction. Use only for individual param configs are the output is large.',
        default=False,
        action='store_true'
    )

    parser.add_argument(
        '--output_path',
        help='Directory to place outputs.'
    )

    parser.add_argument(
        '--params_file',
        help='json file of parameters and their respective ranges to run over.'
    )

    parser.add_argument(
        '--line_profiler',
        help='whether to switch to non ProcessPoolExecutor for line profiling.',
        default = False,
        action='store_true'
    )

    parser.add_argument(
        '--debug_len',
        default=None,
        type=int
    )

    parser.add_argument(
        '--max_workers',
        default=None,
        type=int
    )

    parser.add_argument(
        '--batch_size',
        default=48,
        type=int
    )

    parser.add_argument(
        '--batch_num',
        help = 'For ARC usage, to determine which batch the node running this script should use.',
        type = int,
        default=None
    )

    parser.add_argument(
        '--simulate',
        default=False,
        action='store_true'
    )

    parser.add_argument(
        '--log_dir',
        help='director to place log in. Defaults to $HOME',
        default='$HOME'
    )

    parser.add_argument(
        '--log_level',
        help='logging_level',
        type=str.upper,
        choices=['INFO','DEBUG','WARNING','CRITICAL','ERROR','NONE'],
        default='DEBUG'
    )

    parser.add_argument(
        '--log_handler_level',
        help='log handler setting. "both" for file and stream, "file" for file, "stream" for stream',
        default='both',
        choices = ['both','file','stream']
    )

    # parse args
    args = parser.parse_args()

    logging_dict = {
        'NONE': None,
        'CRITICAL': logging.CRITICAL,
        'ERROR': logging.ERROR,
        'WARNING': logging.WARNING,
        'INFO': logging.INFO,
        'DEBUG': logging.DEBUG
    }

    logging_level = logging_dict[args.log_level]

    if logging_level is not None:

        logging_fmt   = '[%(levelname)s] %(asctime)s - %(message)s'
        today_datetime = str(datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S'))
        logging_file  = os.path.join(args.log_dir, f'{today_datetime}_abm.log')
        if args.log_handler_level == 'both':
            handlers = [
                logging.FileHandler(filename=logging_file,mode='w'),
                logging.StreamHandler()
            ]
        elif args.log_handler_level == 'file':
            handlers = [logging.FileHandler(filename=logging_file,mode='w')]
        elif args.log_handler_level == 'stream':
            handlers = [logging.StreamHandler()]
        logging.basicConfig(
            handlers=handlers,
            format=logging_fmt,
            level=logging_level,
            datefmt='%m/%d/%Y %I:%M:%S %p'
        )

        logging.info(f'Start time of script is {today_datetime}')

    args = parser.parse_args()

    logging.info(f'GROUP NUM: {args.group_num}')

    # process search hashtags.
    args.search_hashtags = load_in_search_ht(args.search_hashtags)

    args.follows_dir    = os.path.join(args.data_path, f'0{args.group_num}_group')
    logging.info(f'Directory where followers info will be searched for: {args.follows_dir}')


    args.peak_analysis_file = os.path.join(args.data_path, f'02_intermediate/FAS_peak_analysis.hdf5')
    assert os.path.isfile(args.peak_analysis_file)

    args.abm_processed_df_savepath = os.path.join(args.data_path, f'06_reporting/ABM_processed_df_group_{args.group_num}.obj')
    logging.info(f'df save path is {args.abm_processed_df_savepath}')

    args.graph_savepath = os.path.join(args.data_path, f'06_reporting/ABM_graph_group_{args.group_num}.obj')
    logging.info(f'Graph save path is {args.graph_savepath}')

    args.agents_savepath = os.path.join(args.data_path, f'06_reporting/ABM_agents_group_{args.group_num}.obj')
    logging.info(f'Agents savepath is {args.agents_savepath}')

    args.model_output_savepath = os.path.join(args.output_path, f'abm/0{args.group_num}_group/ABM_output_group_{args.group_num}_batch_{args.batch_num}.hdf5')
    logging.info(f'Model output savepath is {args.model_output_savepath}')

    if args.history_logging:
        args.model_output_history = os.path.join(args.output_path, f'abm/0{args.group_num}_group/ABM_output_group_{args.group_num}_batch_{args.batch_num}_history.obj')
        logging.info(f'History output savepath is {args.model_output_history}')

    # read in params
    with open(args.params_file, 'r') as f:
        args.params = json.load(f)

    # allow for repetition of particular parameter combination
    if 'repeat' in args.params:
        logging.info(f'Repeat parameter detected.')
        args.repeat_num = int(args.params['repeat'])
        args.params.pop('repeat')
        assert 'repeat' not in args.params
    else:
        logging.info(f'Repeat parameter not detected and is therefore set to 0.')
        args.repeat_num = 1

    args.search_hashtag_propensity_base =  {
        "metoo":                    0.1,
        "balancetonporc":           0.1,
        "moiaussi":                 0.1,
        "نه_یعنی_نه":               0.1,
        "米兔":                      0.1,
        "我也是":                    0.1,
        "gamani":                   0.1,
        "tôicũngvậy":               0.1,
        "私も":                      0.1,
        "watashimo":                0.1,
        "나도":                      0.1,
        "나도당했다":                  0.1,
        "גםאנחנו":                  0.1,
        "ятоже":                    0.1,
        "ricebunny":                0.1,
        "enazeda":                  0.1,
        "anakaman":                 0.1,
        "yotambien":                0.1,
        "sendeanlat":               0.1,
        "kutoo":                    0.1,
        "withyou":                  0.1,
        "wetoo":                    0.1,
        "cuentalo":                 0.1,
        "quellavoltache":           0.1,
        "niunamenos":               0.1,
        "woyeshi":                  0.1,
        "myharveyweinstein":        0.1,
        "noustoutes":               0.1,
        "stilleforopptak":          0.1,
        "nårdansenstopper":         0.1,
        "nårmusikkenstilner":       0.1,
        "memyös":                   0.1,
        "timesup":                  0.1,
        "niere":                    0.1,
        "jotambe":                  0.1
    }

    # process param combinations:
    args.param_grid = list(ParameterGrid(args.params))*args.repeat_num

    # obtain full number of unrolled combinations
    args.full_len = len(args.param_grid)

    # if debugging/line profiling, restrict to the set amount:
    if args.debug_len:
        args.param_grid = args.param_grid[:args.debug_len]

    # now if batching on ARC, check that batch num is supplied and split accordingly
    if args.batch_num is not None:
        args.param_grid = list(chunks(args.param_grid, args.batch_size))
        args.total_batch_count = len(args.param_grid)
        args.param_grid = args.param_grid[args.batch_num]

    # otherwise, if line profiling we just need the number we're running
    else:
        args.total_batch_count = len(args.param_grid)


    if args.batch_num and args.batch_num >= args.total_batch_count:
        logging.warning(f'Out of range for param_grid. Ending...')
    else:
        logging.info(f'Number of combinations: {args.total_batch_count} for a total of {args.full_len}. This is batch number {args.batch_num}')

        # print if line profiling
        if args.line_profiler:
            logging.warning(f'Line Profiler mode activated. Not running multiprocessing pipeline.')

        logging.info(f'History logging is {args.history_logging}')

        main(args)
