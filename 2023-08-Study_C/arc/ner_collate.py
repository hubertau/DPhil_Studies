import os
import glob
import pickle
from loguru import logger
from collections import defaultdict
from pathlib import Path

def collate_ner(ner_batch_dir, outpath, omit_tokens = ['<pad>']):

    '''Function to collate batched ner results into one dataset file
    '''

    # check directory
    assert os.path.isdir(ner_batch_dir)

    # obtain files

    filenames = glob.glob(os.path.join(ner_batch_dir, 'raw_ner_*.pkl'))

    # function to sort glob data
    def extract_number(filename):
        # use only last part of filename
        filename=Path(filename).stem
        # Extract the number using regular expressions (regex)
        match = re.search(r'\d+', filename)
        # If a number is found, return it as an integer
        if match:
            return int(match.group())
        else:
            return 0

    # Sort the filenames based on the number
    filenames_sorted = sorted(filenames, key=extract_number)

    combined_result = defaultdict(set)
    for counter, file in enumerate(filenames_sorted):
        if counter % 100 == 0:
            logger.info(f'Processing {counter} of {len(filenames_sorted)}: {100*counter/len(filenames_sorted):.1f}%')
        with open(file,'rb') as f:
            data = pickle.load(f)

        for result, part_id in data:    
            processed_stories_id = part_id.split('_')[0]
            for item in result:
                if item['entity_group']=='PER':
                    combined_result[processed_stories_id].add(item['word'])

    with open(outpath, 'wb') as f:
        pickle.dump(combined_result, f)

    logger.info(f'Saved to {outpath}')


def main():
    collate_ner()

if __name__=='__main__':
    main()
