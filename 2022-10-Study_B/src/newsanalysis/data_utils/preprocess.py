'''
Script to preprocess obtained and enriched MediaCloud data. The enriching steps are:
* deduplicate
* remove stories that are too short
* remove stories that contain irrelevant information
* remove irrelevant stories
    * this might be with, e.g. words that often indicate excluion: 'SUMMARY:' etc.

'''

import numpy as np
import pandas as pd
import jsonlines
from time import perf_counter
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging
import faiss
import pickle

from newsanalysis.dataviz.plots import retrieve_story_lens, retrieve_story_and_lang

def unique_story_ids(file):
    '''Get unique story ids from jsonl file'''
    unique_ids = set()
    with jsonlines.open(file, 'r') as reader:
        for story in reader.iter(skip_empty=True, skip_invalid=True):
            story_id = story.get('processed_stories_id')
            if story_id:
                unique_ids.add(story_id)
    return unique_ids

def story_iter(file, only_text = True, match_list = None):
    if match_list is not None:
        match_list = set(match_list)
    with jsonlines.open(file, 'r') as reader:
        for story in reader.iter(skip_empty=True, skip_invalid=True):
            if 'query' in story:
                continue
            story_id = story.get('processed_stories_id')
            if (match_list and story_id in match_list) or not match_list:
                if only_text:
                    yield story.get('text')
                else:
                    yield story_id

def deduplicate(file):
    '''Return dict of story ids and their duplicates'''

    df = retrieve_story_and_lang(file)
    dim=10000

    # unique_ids = list(story_iter(file, only_text=False))

    # Process by language
    grouped = df.groupby('lang').apply(lambda x: x['id'].unique())

    for l in df['lang'].unique():
        print(f'Processing {l}')
        m_list = grouped.loc[l]
        print(len(m_list))
        vectorizer = TfidfVectorizer(
            analyzer='word',
            norm='l2',
            max_features=dim
        )
        print('start fit transform')
        t1_start = perf_counter()
        csr = vectorizer.fit_transform(story_iter(file, match_list=m_list))
        t1_stop = perf_counter()
        print(f'end fit transform. Seconds taken: {t1_stop-t1_start:.2f}') 

        print(f'Shape: {csr.shape}')

        print('Starting FAISS')
        start=perf_counter()
        index = faiss.IndexFlatIP(dim)
        #add the rows of the dataframe into Faiss
        index.add(csr)

        k = csr.shape[0]
        D, I = index.search(csr, k) 
        stop=perf_counter()
        print(f'End FAISS: {stop-start:.2f}s elapsed')



    # print('start fit transform')
    # t1_start = perf_counter()
    # csr = vectorizer.fit_transform(story_iter(file))
    # t1_stop = perf_counter()
    # print(f'end fit transform. Seconds taken: {t1_stop-t1_start:.2f}')

    # with open('/home/hubert/DPhil_Studies/2022-10-Study_B/data/03_processed/csr.pkl', 'wb') as f:
    #     pickle.dump(csr, f)

    # # assert len(unique_ids) == csr.shape[0]

    # dim = csr.shape[1]
    # print(csr.shape)

    # # print(f'Calculating similarity...')
    # # sim_matrix = cosine_similarity(csr)
    # # print('Similarity calculated')

    # # array1 = np.random.random((csr.shape[0], dimension)).astype('float32')

    # print('Starting FAISS')
    # start=perf_counter()
    # index = faiss.IndexFlatIP(dim)
    # #add the rows of the dataframe into Faiss
    # index.add(csr)

    # k = csr.shape[0]
    # D, I = index.search(csr, k) 
    # stop=perf_counter()
    # print(f'End FAISS: {stop-start:.2f}s elapsed')

    # # similarity by faiss


    # return unique_ids, D, I