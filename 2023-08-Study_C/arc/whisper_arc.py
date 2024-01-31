import glob
import os
import matplotlib.pyplot as plt
import click
from pathlib import Path
import json
from loguru import logger
from faster_whisper import WhisperModel

@click.command()
@click.option('--filelist', '-f', required=False)
@click.option('--indir', '-i')
@click.option('--format', '-fo', required=False, default='m4a')
@click.option('--start', '-s', required=True, type=int)
@click.option('--end', '-e', required=True, type=int)
@click.option('--outdir', '-o', required=True)
def main(filelist, indir, format, start, end, outdir):

    # get files
    if filelist:
        with open(filelist, 'r') as f:
            x = f.readlines()
            x = [i.replace('\n', '') for i in x]
    elif indir:
        x = sorted(glob.glob(os.path.join(indir, f'*.{format}')))
    else:
        raise Error

    # Specify your size threshold in bytes (e.g., 100000 bytes = 100 KB)
    size_threshold = 1024*1024*50
    
    # Filter files by size
    x = [file for file in x if os.path.getsize(file) <= size_threshold]
    logger.info(f'Length of x: {len(x)}')

    model_size = "medium"

    # Run on GPU with FP16
    # model = WhisperModel(model_size, device="cuda", compute_type="float16")
    
    # or run on GPU with INT8
    # model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")
    # or run on CPU with INT8
    model = WhisperModel(model_size, device="cpu", compute_type="int8")

    logger.info(f'Start: {start}. End: {end}.')
    files = x[start:end+1]

    if start > len(x):
        logger.info(f'Reached end of files. Returning...')
        return None

    for index, file in enumerate(files):
        outpath = os.path.join(outdir,Path(file).stem+'.txt')
        if os.path.isfile(outpath):
            logger.info(f'{outpath} Already exists. Continuing...')
            continue

        logger.info(f'Processing {start+index}: {file}')
        segments, info = model.transcribe(
            file,
            beam_size=5,
            vad_filter=True
        )
        text_to_save = [s.text for s in segments]
        text_to_save = ". ".join(text_to_save)

        with open(outpath, 'w') as f:
            f.write(text_to_save)
        logger.info('Done.')

if __name__=='__main__':
    main()
