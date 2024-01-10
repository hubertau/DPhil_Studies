import gzip
# !python -m pip install jsonlines
from typing import List, Dict
import jsonlines

def writeall_jsonl_gz(filename, payload: List[Dict], dumps=None):
    with gzip.open(filename, 'wb') as fp:
        json_writer = jsonlines.Writer(fp, dumps=dumps)
        json_writer.write_all(payload)

def read_jsonl_gz(filename) -> List[Dict]:
    data = []
    with gzip.open(filename, 'rb') as fp:
        j_reader = jsonlines.Reader(fp)

        for obj in j_reader:
            data.append(obj)

    return data