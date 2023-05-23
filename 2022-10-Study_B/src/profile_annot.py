### SCRIPT TO PROFILE NER ANNOTATION

from newsanalysis.data_utils.preprocess import annotate

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

prof_annotate = profile(annotate, '/home/hubert/DPhil_Studies/2022-10-Study_B/src/profile_annot.txt')

prof_annotate('/home/hubert/DPhil_Studies/2022-10-Study_B/data/01_raw/data_cleaned_bt_ner_test',
        '/home/hubert/DPhil_Studies/2022-10-Study_B/data/03_raw/NER_TEST_PROFILE',
        model = "51la5/roberta-large-NER",
        tok = None,
        num_batches=2,
        kind = 'ner',
        max_length=512,
        batch_size=8
        )