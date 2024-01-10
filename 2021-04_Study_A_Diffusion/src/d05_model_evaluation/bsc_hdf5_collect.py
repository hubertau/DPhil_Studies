import h5py
import os
import logging
import glob
import argparse
import datetime

def main(args):

    glob_path = os.path.join(args.bsc_eval_dir,f'0{args.group_num}_group',f'bispec_cluster_eval*.hdf5')
    hdf5_list = glob.glob(glob_path)
    logging.debug(f'Glob is {glob_path}')
    logging.debug(f'Folder is {hdf5_list}')

    # loop over hdf5 list and add incrementally to output file
    with h5py.File(args.output_file,'a') as output:
        for hdf5_file in hdf5_list:
            logging.info(f'Processing {hdf5_file}')
            with h5py.File(hdf5_file, 'r') as f:
                group_key = list(f.keys())[0]
                logging.debug(f'group key is {group_key}')
                ngram_key = list(f[group_key].keys())[0]
                logging.debug(f'ngram key is {ngram_key}')
                min_user_key = list(f[group_key][ngram_key].keys())[0]
                logging.debug(f'min_user_key is {min_user_key}')
                ht_key = list(f[group_key][ngram_key][min_user_key].keys())[0]
                logging.debug(f'ht_key is {ht_key}')

                g = output.require_group(group_key)
                n = g.require_group(ngram_key)
                x = n.require_group(min_user_key)
                y = x.require_group(ht_key)
                before = y.require_group('before')
                after = y.require_group('after')
                for string, obj in [('before',before),('after',after)]:
                    for cluster_key in f[group_key][ngram_key][min_user_key][ht_key][string].keys():
                        if cluster_key in output[group_key][ngram_key][min_user_key][ht_key][string].keys():
                            del output[group_key][ngram_key][min_user_key][ht_key][string][cluster_key]
                        # logging.debug('writing to new hdf5...')
                        obj.create_dataset(cluster_key, data = f[group_key][ngram_key][min_user_key][ht_key][string][cluster_key])

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='collect bsc hdf5 results')

    parser.add_argument(
        'bsc_eval_dir',
        help='folder for bsc eval files'
    )

    parser.add_argument(
        'output_file',
        help='output file. May already exist'
    )

    parser.add_argument(
        'group_num'
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
        logging_file  = os.path.join(args.log_dir, f'{today_datetime}_bsc_hdf5_collect.log')
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

    main(args)