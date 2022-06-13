################################################################################
# Script to evaluate the ABM outputs so that space can be freed and also because the data cannot be copied back over from ARC
################################################################################

from abm import *
import glob
from concurrent.futures import ProcessPoolExecutor

def print_abm_results(agents_dict, model_num=None):

    output_dict={}

    total = len(agents_dict)
    # num_supporting = 0
    # num_not_supporting = 0

    for user_id, agent in agents_dict.items():
        output_dict[user_id] = agent.supporting_metoo_dict

    output_df = pd.DataFrame.from_dict(output_dict, orient='index').reset_index()

    num_supporting = output_df.iloc[:,1:].sum(axis=0)
    num_supporting = num_supporting[num_supporting>0]
    num_not_supporting = (output_df==False).sum(axis=0)

    # print(num_supporting)
    # print(num_not_supporting)
    return output_df


def reference_results(args, agents):
    act_val = {}
    with h5py.File(args.activity_file, 'r') as f:
        activity_base = f[f'group_{args.group_num}']
        feature_order = f[f'group_{args.group_num}'][list(agents.keys())[0]]['hashtagged'].attrs['feature_order']
        feature_order = feature_order.split(';')

        for user_id, agent in agents.items():
            # obtain user activity
            act_val[user_id] = {}
            activity = activity_base[user_id]['hashtagged'][:]

            # act_val[user_id] = np.sum(activity[:,-int(daterange_length/2):])

            for hashtag_in_period in args.most_prominent_peaks:
                hashtag_in_period_index = feature_order.index(hashtag_in_period)

                # obtain the index offset from the detected peak of the hashtag to collect initial time window.
                peak_index_index = (datetime.datetime.strptime(args.group_date_range.end, '%Y-%m-%d')-args.most_prominent_peaks[hashtag_in_period]).days
                # offset_index -= peak_delta_init
                # offset_index = max(0,offset_index)+1
                # print(f'Offset for {hashtag_in_period} is {offset_index}')

                act_val[user_id][hashtag_in_period_index]= np.sum(activity[hashtag_in_period_index,-peak_index_index-1:])

    act_val = pd.DataFrame.from_dict(act_val, orient='index').reset_index()
    act_val.columns = ['user_id'] + list(args.most_prominent_peaks.keys())

    return act_val

def process_one_abm_res_file(res_pointer):

    logging.info(f'Processing {res_pointer}')
    with open(res_pointer, 'rb') as f:
        temp = pickle.load(f)

    results = []
    for tup in temp:
        params = tup[0]
        agents = tup[1]
        res = print_abm_results(agents)
        num_supporting = res.iloc[:,1:].sum(axis=0)
        num_supporting = num_supporting[num_supporting>0]
        num_supporting = num_supporting.to_frame().reset_index()
        num_supporting.columns = ['index', 'abm']

        # comparison = act_val_df.merge(num_supporting, on='index', how='right').fillna(0)

        results.append((params, num_supporting))

    return results

def main(args):

    args.most_prominent_peaks, args.group_date_range, args.daterange_length = group_peaks_and_daterange(args.peak_analysis_file, args.group_num)

    args.results_list = glob.glob(os.path.join(args.data_path, f'abm/0{args.group_num}_group/ABM_*.obj' ))
    assert len(args.results_list) > 0
    logging.info(f'To assess: {len(args.results_list)} files.')

    with open(args.results_list[0], 'rb') as f:
        res = pickle.load(f)
        first_agents = res[0][1]
    act_val = reference_results(args, first_agents)
    act = (act_val.iloc[:,1:]>2).sum(axis=0).to_frame().reset_index()
    act.columns = ['index', 'actual']

    logging.info ('Running process pool executor...')
    with ProcessPoolExecutor(max_workers = args.max_workers) as executor:
        output = executor.map(process_one_abm_res_file, args.results_list)

    logging.info ('Collecting results...')
    final = list(output) + [('act_val_reference', act)]

    with open(args.eval_output_savepath, 'wb') as f:
        pickle.dump(final, f)
        logging.info(f'Saved at {args.eval_output_savepath}')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='')

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
        '--data_path',
        help='Data path where abm df can be found.'
    )

    parser.add_argument(
        '--activity_file'
    )


    parser.add_argument(
        '--output_path',
        help='Directory to place outputs.'
    )


    parser.add_argument(
        '--debug_len',
        default=None,
        type=int
    )

    parser.add_argument(
        '--peak_analysis_file'
    )

    parser.add_argument(
        '--max_workers',
        default=None,
        type=int
    )

    parser.add_argument(
        '--batch_num',
        help = 'For ARC usage, to determine which batch the node running this script should use.',
        type = int,
        default=None
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


    parser.add_argument(
        '--option',
        help=''
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
        logging_file  = os.path.join(args.log_dir, f'{today_datetime}_abm_eval.log')
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


    # load in search hashtags
    args.search_hashtags = load_in_search_ht(args.search_hashtags)

    args.eval_output_savepath = os.path.join(args.output_path, f'ABM_summary_group_{args.group_num}.obj')
    logging.info(f'Summary output savepath is {args.eval_output_savepath}')

    # read in params

    main(args)
