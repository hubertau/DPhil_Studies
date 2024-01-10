### SCRIPT TO PROFILE NER ANNOTATION

from newsanalysis.data_utils.preprocess import annotate
import click
from loguru import logger
from line_profiler import LineProfiler
from pathlib import Path

@click.command()
@click.option('--log', '-l')
@click.option('--dataset', '-d')
@click.option('--outpath', '-o')
@click.option('--model', '-m')
@click.option('--batchsizepergpu', '-b', default = 800, type = int)
@click.option('--num_batches', '-n', default=2, type=int)
def prof_annotate(log, dataset, model, outpath, num_batches, batchsizepergpu):
    logger.info(log)

    lp = LineProfiler()
    lp_wrapper = lp(annotate)

    lp_wrapper(dataset,
        outpath,
        model = model,
        tok = None,
        num_batches=num_batches,
        kind = 'ner',
        max_length=512,
        batch_size_per_gpu=batchsizepergpu
    )

    with open(log, "w", encoding="utf-8") as f:
        lp.print_stats(f, output_unit=1e-03)

if __name__ == '__main__':

    prof_annotate()