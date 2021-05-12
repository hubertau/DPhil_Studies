'''
Script to process incoming json files to csv for further processing.

DEPRECATED AS OF 2021-05-12. NOT USEFUL ANYMORE.

Files are just to be kept as json. csv is not a good format for Twitter data.

'''


import pandas as pd
import glob
import argparse
import os

def convert_json_to_csv(filepath, out_file):

    # open file with pandas
    df = pd.read_json(filepath)

    # convert ISO 8601 format to datetime object
    df['created_at'] = pd.to_datetime(df['created_at'])

    if 'geo' not in df.columns:
        df['geo']=''

    df.to_csv(out_file, mode='a+', columns=['conversation_id', 'text', 'possibly_sensitive', 'lang',
       'entities', 'referenced_tweets', 'created_at', 'public_metrics',
       'author_id', 'id', 'in_reply_to_user_id', 'geo'], index=False, line_terminator='z')

    print('{} appended'.format(os.path.split(filepath)[-1]))


def main():

    parser = argparse.ArgumentParser(description='convert directory of json files into csv for further processing')

    parser.add_argument(
        'indir',
        type=str
    )

    args = parser.parse_args()

    # check that the given directory is indeed a directory
    if os.path.isabs(args.indir):
        assert os.path.isdir(args.indir)
    else:
        assert os.path.isdir(os.path.join(os.getcwd(), args.indir))

    # define data path
    data_path = os.path.join(args.indir, 'data')

    # define output file:
    output_file = os.path.join(data_path,'parsed_FAS.csv')

    # get list of json files collected
    file_list = sorted(glob.glob(data_path + '/FAS_*.json'), key=os.path.getctime)

    for each_file in file_list:
        convert_json_to_csv(each_file, output_file)

if __name__ == '__main__':
    main()