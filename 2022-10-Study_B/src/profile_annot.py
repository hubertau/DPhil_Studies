### SCRIPT TO PROFILE NER ANNOTATION

from newsanalysis.data_utils.preprocess import annotate
import click

def profile(func, outtxt):
    from functools import wraps

    @wraps(func)
    def wrapper(*args, **kwargs):
        from line_profiler import LineProfiler
        prof = LineProfiler()
        try:
            return prof(func)(*args, **kwargs)
        finally:
            with open(outtxt, "w", encoding="utf-8") as f:
                prof.print_stats(f)

    return wrapper

@click.command()
@click.option('--log', '-l')
@click.option('--dataset', '-d')
@click.option('--outpath', '-o')
@click.option('--batchsizepergpu', '-b', default = 800, type = int)
@click.option('--num_batches', '-n', default=2, type=int)
def prof_annotate(log, dataset, outpath, num_batches, batchsizepergpu):

    profile(annotate(dataset,
        outpath,
        model = "51la5/roberta-large-NER",
        tok = None,
        num_batches=num_batches,
        kind = 'ner',
        max_length=512,
        batch_size_per_gpu=batchsizepergpu
        ),
        log
    )

if __name__ == '__main__':

    prof_annotate()