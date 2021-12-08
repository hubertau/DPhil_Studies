#!/usr/bin/python3.9

'''
Obtain the inferences for a given group though m3

Citation:
@inproceedings{wang2019demographic,
  title={Demographic inference and representative population estimates from multilingual social media data},
  author={Wang, Zijian and Hale, Scott and Adelani, David Ifeoluwa and Grabowicz, Przemyslaw and Hartman, Timo and Fl{\"o}ck, Fabian and Jurgens, David},
  booktitle={The World Wide Web Conference},
  pages={2056--2067},
  year={2019},
  organization={ACM}
}
'''

import argparse
import datetime
import glob
import json
import logging
import os
import re

import jsonlines
from m3inference import M3Twitter


def m3_input_file_creator(args):

    """Generate the jsonl object to be transformed by m3twitter instance.

    Returns:
        filename: filename of jsonl object saved
    """

    # extract user ids
    user_files = glob.glob(os.path.join(args.data_dirs, f'0{args.group_num}_group/timeline*.jsonl'))
    # user_ids = [re.split('[_.]',i)[-2] for i in user_files]

    m3_input_file_name = os.path.join(args.data_dirs, f'0{args.group_num}_group/m3_users.jsonl')
    logging.info(f'm3 input file being created, to be saved at {m3_input_file_name}')

    default_profile_image_url = "https://abs.twimg.com/sticky/default_profile_images/default_profile_normal.png"

    user_objects = []
    for file in user_files:
        current_user_id = re.split('[_.]',file)[-2]
        with jsonlines.open(file, 'r') as f:
            for line in f:

                # N.B. the following adaptations are necessary for 
                obj_to_append = line['includes']['users'][0]
                assert current_user_id==obj_to_append['id']
                if 'default_profile_image' not in obj_to_append:
                    if obj_to_append['profile_image_url'] == default_profile_image_url:
                        obj_to_append['default_profile_image'] = True
                    else:
                        obj_to_append['default_profile_image'] = False
                obj_to_append['profile_image_url_https'] = obj_to_append['profile_image_url']
                obj_to_append['id_str'] = obj_to_append['id']
                obj_to_append['screen_name'] = obj_to_append['username']
                user_objects.append(obj_to_append)

                break

    with jsonlines.open(m3_input_file_name, 'w') as writer:
        writer.write_all(user_objects)

    return m3_input_file_name

def main(args):

    # create m3 file to be transformed and have images downloaded
    m3_input_file = m3_input_file_creator(args)

    # 
    cache_dir = os.path.join(args.data_dirs, f'0{args.group_num}_group')
    m3twitter=M3Twitter(cache_dir=cache_dir) #Change the cache_dir parameter to control where profile images are downloaded
    logging.info(f'Cache dir set to {cache_dir}')
    m3_input_transformed = os.path.join(args.data_dirs, f'0{args.group_num}_group/m3_users_transformed.jsonl')
    m3twitter.transform_jsonl(
        input_file=m3_input_file,
        output_file=m3_input_transformed
    )
    logging.info(f'Input transformed.')

    output = m3twitter.infer(m3_input_transformed)
    logging.info('Properties inferred')
    output_filename = os.path.join(args.output_dir, f'm3inferred_group_{args.group_num}')

    with open(output_filename, 'w') as f:
        json.dump(output,f)
        logging.info(f'Saved to {output_filename}')

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Obtain m3inferences for group')

    parser.add_argument(
        '--data_dirs',
        help='directory where timeline (and augmentation) files are. This is to '
    )

    parser.add_argument(
        '--output_dir',
        help='where to place the final m3 results for users'
    )

    parser.add_argument(
        '--group_num'
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
        logging_file  = os.path.join(args.log_dir, f'{today_datetime}_activity_counts_stats.log')
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
