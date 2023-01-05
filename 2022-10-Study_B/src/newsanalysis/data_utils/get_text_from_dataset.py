'''
Script to collect article texts and titles from an existing dataset of links
'''
from newspaper import Article
import csv
import argparse
import os
import pandas as pd
import time
import tqdm

parser = argparse.ArgumentParser(description = 
        'get text from existing link dataset')
parser.add_argument('file',
                    type = str,
                    help = 'file to be augmented with text')
parser.add_argument('--outdir',
                    type = str,
                    help = 'output directory',
                    default = os.path.dirname(os.path.abspath(__file__)))
parser.add_argument('--cat_filter',
                    type = str,
                    help = 'category to only produce results for',
                    default = None)



args = parser.parse_args()
original_file = args.file
CAT_FILTER = args.cat_filter

# Check if cat filter is applied
if CAT_FILTER is not None:
    CAT_FILTER_LIST = CAT_FILTER.split(',')
    for i in CAT_FILTER_LIST:

        # THIS LIST MUST BE UPDATED IF NEW CATEGORIES ARE INTRODUCED (e.g. antivaxx)
        # error will be thrown otherwise.
        assert i.lower() in ['antivaxx','foreign.state', 'jn', 'mainstream']

    CAT_FILTER_LIST = [element.lower() for element in CAT_FILTER_LIST]
    print('cat filter applied. collecting text information on {}.'.format(CAT_FILTER_LIST))

    # This next if section is to standardise foreign.state to sb.
    # Artifact of non-standard nomenclature somewhere.
    if 'foreign.state' in CAT_FILTER_LIST:
        print_list = CAT_FILTER_LIST.copy()
        print_list.remove('foreign.state')
        print_list.append('sb')
        print_list = '-'.join(print_list)
    outfile = os.path.join(args.outdir, os.path.split(original_file)[-1][:-4] + '-with-titles-' + print_list + '.csv')
else:
    outfile = os.path.join(args.outdir, os.path.split(original_file)[-1][:-4] + '-with-titles_combined.csv')
    print('no cat filter applied.')

print(outfile)

# get length of file to run through
with open(original_file, 'r') as f:
    TOTAL_LINK_NUM = len(f.readlines())

start_time = time.time()
with open(original_file, 'r') as f:
    original_csv = csv.reader(f)
    headers = next(original_csv)
    print(headers)
    headers = [headers[0]] + ['title','text'] + headers[1:]
    with open(outfile, 'w', encoding = 'utf-8', newline='') as out_f:
        outfile_csv = csv.writer(out_f)
        outfile_csv.writerow(headers)
        
        print('collecting article text')
        for original_row in tqdm.tqdm(original_csv, total=TOTAL_LINK_NUM):
            if args.cat_filter is None or original_row[4] in CAT_FILTER_LIST:
                try:
                    article = Article(original_row[0])
                    article.download()
                    article.parse()
                    outfile_csv.writerow([original_row[0]] + [article.title, article.text] + original_row[1:])
                except:
                    pass
print('Total time taken: {}'.format(time.time()-start_time))
