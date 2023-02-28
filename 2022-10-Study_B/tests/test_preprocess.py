import subprocess
import glob
import os

def test_obtain_clusters(tmp_path):
    _ = subprocess.run([
        'newsanalysis',
        'obtain-clusters',
        '../data/01_raw/data_ndid.jsonl',
        tmp_path,
        '-u',
        '50',
        '-p',
        '5'
    ])
    end_files = glob.glob(os.path.join(tmp_path, '*'))
    assert len(end_files) == 2