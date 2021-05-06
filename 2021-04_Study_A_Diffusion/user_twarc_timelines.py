import subprocess
import os
import argparse 
import logging
import tqdm

def main():

    with open(args.user_list) as f:
        users = f.readlines()

    for user_id in tqdm.tqdm(users):

        save_filename = os.path.join(args.output_dir,'data/timeline_' + user_id + '.jsonl')

        subprocess.run(
            ['twarc2',
            'timeline',
            '--end-time',
            '2017-12-31T23:59:59',
            user_id,
            save_filename]
        )

        logging.info(user_id)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='timeline collection')

    parser.add_argument(
        'user_list',
        help='list of users in a txt file.'
    )

    parser.add_argument(
        '--output_dir',
        help='directory to place outputs'
    )

    args = parser.parse_args()

    logging.basicConfig(filename=os.path.join(args.output_dir, 'user_timelines.log'),
                        encoding='utf-8',
                        format='%(levelname)s:%(message)s',
                        level=logging.DEBUG)

    main()