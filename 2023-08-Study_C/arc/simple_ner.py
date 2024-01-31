import transformers
from transformers import pipeline, AutoTokenizer
from transformers.utils import logging
import datasets
import torch
import psutil
from loguru import logger
import click
import pickle
from pathlib import Path

@click.command()
@click.argument('start', type=int)
@click.argument('outdir')
@click.option('--size', default=1000000, type=int)
def main(start, outdir, size):

    logging.set_verbosity_error()

    batch_size = 100
    logger.info(f'batch size {batch_size}')

    out = []
    model_path='/data/inet-large-scale-twitter-diffusion/ball4321/data_b/04_models/roberta-large-NER'
    tokenizer=AutoTokenizer.from_pretrained(model_path, model_max_length=512)

    data = datasets.load_from_disk('/data/inet-large-scale-twitter-diffusion/ball4321/data_b/01_raw/data_cleaned_bt_split')
    pipe = pipeline(
        task = 'ner',
        model=model_path,
        tokenizer=tokenizer,
        aggregation_strategy='simple',
        batch_size=batch_size,
        device='cuda'
    )

    
    #adjust size if at end
    size = min(len(data)-start, size)
    logger.info(f'PARAMS: Start - {start}, size - {size}')


    total_batches = (size + batch_size - 1) // batch_size

    
    for batch_num, batch_start in enumerate(range(start,start+size,batch_size)):
        batch_end = batch_start+batch_size
        if batch_num == 0:
            logger.info(f'0% complete, Batch number: 0 of {total_batches}')
        if batch_num % (total_batches // 10) == 0 and batch_num > 0:
            percent_complete = (batch_num / total_batches) * 100
            logger.info(f'{percent_complete:.0f}% complete, Batch number: {batch_num} of {total_batches}')
        temp = pipe(
            data[batch_start:batch_end]['text']
        )

        out.extend(list(zip(temp, data[batch_start:batch_end]['part_id'])))

    if total_batches > 0:
        logger.info(f"Processing: 100% complete, Batch number: {total_batches}")

    savefilename = Path(outdir) / f'raw_ner_{start}-{start+size-1}.pkl'
    
    with open(savefilename, 'wb') as f:
        pickle.dump(out, f)
    logger.info(f'saved to {savefilename}')


if __name__=='__main__':

    main()
