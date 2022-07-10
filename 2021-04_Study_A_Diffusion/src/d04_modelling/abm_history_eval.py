'''
Script to process the checks on Awareness, Translation, and Experimentation for history logging.
'''

import argparse
import logging
import datetime
import os
import glob
import re
import pickle
from typing import NamedTuple
from concurrent.futures import ProcessPoolExecutor

class graph_score_result(NamedTuple):
    awareness_count: int
    awareness_total: int
    experimentation_count: int
    application_count: int


# function to process history graph:
# @profile
def score_graph(history_graph, kind='awareness'):

    # extract graph node primary_ht data:
    # logging.info(f'Processing attribute dict for nodes')
    primary_ht_dict = {}
    for node, attributes in history_graph.nodes(data=True):
        primary_ht_dict[node] = attributes['primary_ht']

    if kind=='awareness':

        # if kind is awareness then check the interactions between agents of different primary hashtags

        # history_graph.nodes(data=True)

        # structure of history_graph edges:
        # interact_result
        # time
        # experimentation_success
        # ht - ht of influence here.
        # N.B. also source is the the influencing the target, i.e. the target's support dict is potentially changed after the interaction.

        # nodex have primary_ht as a data attribute.

        awareness_total = 0
        awareness_count = 0
        experimentation_count = 0
        application_count = 0

        for source, target, datadict in  history_graph.edges(data=True):

            # extract primary ht
            # source_primary_ht = history_graph.nodes(data=True)[source]['primary_ht']
            # target_primary_ht = history_graph.nodes(data=True)[target]['primary_ht']
            source_primary_ht = primary_ht_dict[source]
            target_primary_ht = primary_ht_dict[target]

            awareness_total+=1

            # check if awareness happens on this interaction
            if datadict['interact_result']:
                if source_primary_ht != target_primary_ht:
                    awareness_count += 1
                    if (datadict['ht'] != source_primary_ht) and (datadict['ht'] != target_primary_ht):
                        application_count += 1
                if datadict['experimentation_success']:
                    experimentation_count += 1

        return graph_score_result(
            awareness_count=awareness_count,
            awareness_total=awareness_total,
            experimentation_count=experimentation_count,
            application_count=application_count
        )

def process_one_history_file(history_file_tuple):

    index = history_file_tuple[0]
    history_file = history_file_tuple[1]

    with open(history_file, 'rb') as f:
        history_file_data = pickle.load(f)

    logging.info(f'Processing number {index}: {os.path.split(history_file)[-1]}')
    results = []
    for params, history_graph in history_file_data:

        one_graph_score = score_graph(history_graph)
        results.append((params, one_graph_score))

    return results

def main(args):

    if args.lineprofile:
        process_one_history_file(args.history_files[0])
    else:
        logging.info('Beginning ProcessPoolExecutor')
        with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
            result_iterator = executor.map(process_one_history_file, enumerate(args.history_files))

        x = list(result_iterator)
        logging.debug('Success')

        with open(args.output_savename, 'wb') as f:
            pickle.dump(x,f)
        logging.info('Saving complete.')

if __name__ == '__main__':

    parser = argparse.ArgumentParser('Process history obj files for a group.')

    parser.add_argument(
        'history_dir',
        help='Directory where history files are collected'
    )

    parser.add_argument(
        '--output_dir',
        help='Where to place output. Default is history_dir.'
    )

    parser.add_argument(
        '--max_workers',
        default=48,
        type=int
    )

    parser.add_argument(
        '--lineprofile',
        help='whether kernprof lineprofiler is being run.',
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
        logging_file  = os.path.join(args.log_dir, f'{today_datetime}_abm_history_eval.log')
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

    # check that history dir is a directoy
    assert os.path.isdir(args.history_dir)

    # collect list of history files
    args.history_files = glob.glob(os.path.join(args.history_dir, '*history.obj'))
    logging.info(f'{len(args.history_files)} history files detected.')

    # split the first one to extract group number. Expected format is: ABM_output_group_2_batch_0_history.obj
    history_name_split = re.split('[_.]',os.path.split(args.history_files[0])[-1])
    group_num = history_name_split[3]
    logging.info(f'Detected group num is {group_num}')

    # process save name
    args.output_savename = f'ABM_history_eval_group_{group_num}.obj'
    if args.output_dir:
        args.output_savename = os.path.join(args.output_dir, args.output_savename)
    else:
        args.output_savename = os.path.join(args.history_dir, args.output_savename)
    logging.info(f'Savepath will be: {args.output_savename}')

    main(args)
