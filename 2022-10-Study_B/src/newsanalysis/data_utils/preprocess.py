'''
Script to preprocess obtained and enriched MediaCloud data. The enriching steps are:
* deduplicate
* remove stories that are too short
* remove stories that contain irrelevant information
* remove irrelevant stories
    * this might be with, e.g. words that often indicate excluion: 'SUMMARY:' etc.
    * BERTopic?

'''

import numpy as np
import pandas as pd
import os
from loguru import logger
from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
import pandas as pd
import jsonlines
from time import perf_counter
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import faiss
import h5py

from newsanalysis.dataviz.plots import retrieve_story_lens, retrieve_story_and_lang

def chunks(iter1, iter2, n):
    """Yield successive n-sized chunks from iter1 and iter2."""
    assert iter1.shape[0] == iter2.shape[0]
    for i in range(0, iter1.shape[0], n):
        yield (i, iter1[i:i + n], iter2[i:i + n])

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

def deduplicate(file, savepath, gpu=False):
    '''Return dict of story ids and their duplicates'''

    df = retrieve_story_and_lang(file)
    dim=10000

    # unique_ids = list(story_iter(file, only_text=False))

    # Process by language
    grouped = df.groupby('lang').apply(lambda x: x['id'].unique())

    for l in df['lang'].unique():
        logger.info(f'Processing {l}')
        m_list = grouped.loc[l]
        logger.info(len(m_list))
        ordered_ids = np.array(list(story_iter(file, only_text = False, match_list=m_list)))
        vectorizer = TfidfVectorizer(
            analyzer='word',
            norm='l2',
            max_features=dim
        )
        logger.info('start fit transform')
        t1_start = perf_counter()
        csr = vectorizer.fit_transform(story_iter(file, match_list=m_list))
        t1_stop = perf_counter()
        logger.info(f'end fit transform. Seconds taken: {t1_stop-t1_start:.2f}') 

        # with open(os.path.join(savepath, 'csr.pkl'), 'wb') as f:
            # pickle.dump(csr, f)

        logger.info(f'Shape: {csr.shape}')

        logger.info('Starting FAISS')
        start=perf_counter()

        # cf. https://towardsdatascience.com/ivfpq-hnsw-for-billion-scale-similarity-search-89ff2f89d90e#9718

        d = 10000       # Dimension (length) of vectors.
        M = 32         # Number of connections that would be made for each new vertex during HNSW construction.
        nlist = min(10000, int(np.floor(csr.shape[0]/2)))  # Number of inverted lists (number of partitions or cells).
        nsegment = 16  # Number of segments for product quantization (number of subquantizers).
        nbit = 8       # Number of bits to encode each segment.

        # Create the index.
        coarse_quantizer = faiss.IndexHNSWFlat(d, M)
        index = faiss.IndexIVFPQ(coarse_quantizer, d, nlist, nsegment, nbit)
        if gpu:
            # declare GPU resource
            res = faiss.StandardGpuResources()
            logger.info(f'Running with GPU')
            index = faiss.index_cpu_to_gpu(
                res,
                0,
                index
            )
        # Run training to perform k-means clustering (xt are vectors used for training).
        index.train(csr[:min(40*nlist, csr.shape[0])].todense().astype(np.float32))

        # Adding vectors to the index (xb are database vectors that are to be indexed).
        logger.info('Adding rows to index...')
        batch_size = 10000
        for _, sparse_vectors, ids in chunks(csr, ordered_ids, batch_size):
            index.add_with_ids(sparse_vectors.todense().astype(np.float32), ids)
        logger.info('Done adding rows to index')

        # Setting the number of partitions to search.
        index.nprobe = 100

        # xq are query vectors, for which we need to search in xb to find the k nearest neighbors.
        # The search returns D, the pairwise distances, and I, the indices of the nearest neighbors.
        k = 10
        D = np.zeros((csr.shape[0],k))
        I = np.zeros((csr.shape[0],k))
        for num, sparse_vectors, ids in chunks(csr, ordered_ids, batch_size):
            D[num*batch_size:num*batch_size+batch_size,:], I[num*batch_size:num*batch_size+batch_size,:] = index.search(sparse_vectors.todense().astype(np.float32), k)

        savename = os.path.join(savepath, f'results_{"gpu" if gpu else "cpu"}.hdf5') 
        with h5py.File(savename, 'a') as f:
            g = f.require_group(l)
            # clear previous datasets:
            for dsetname in ['D', 'I', 'ids']:
                if dsetname in g.keys():
                    del g[dsetname]
            g.create_dataset('D', data=D)
            g.create_dataset('I', data=I)
            g.create_dataset('ids', data=ordered_ids)
        # with open(savename, 'wb') as f:
        #     pickle.dump(ordered_ids, f)
        # savename = os.path.join(savepath, f'indices_{l}.pkl') 
        # with open(savename, 'wb') as f:
        #     pickle.dump(I, f)
        # savename = os.path.join(savepath, f'distances_{l}.pkl') 
        # with open(savename, 'wb') as f:
        #     pickle.dump(D, f)
        stop=perf_counter()
        logger.info(f'End FAISS: {stop-start:.2f}s elapsed')



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


    # return unique_ids, D, Inlist

def filter_by_cluster(file):

    # Step 1 - Extract embeddings.
    embedding_model = SentenceTransformer("xlm-roberta-large")

    # Step 2 - Reduce dimensionality.
    umap_model = UMAP(
        n_neighbors=15,
        n_components=5,
        min_dist=0.0,
        metric='cosine',
        random_state=42
    )

    # Step 3 - Cluster reduced embeddings.
    hdbscan_model = HDBSCAN(
        min_cluster_size=15,
        metric='euclidean',
        cluster_selection_method='eom',
        prediction_data=True
    )

    # Step 4 - Tokenize topics.
    vectorizer_model = CountVectorizer(
        stop_words="english"
    )

    # Step 5 - Create topic representation.
    ctfidf_model = ClassTfidfTransformer()

    topic_model = BERTopic(
        language='multilingual', # Set to 'multilingual' for datasets with languages other than English.
        top_n_words=10,
        n_gram_range=(1, 1),
        min_topic_size=10,
        nr_topics=None,
        low_memory=False,
        calculate_probabilities=True, # The probabilities of all topics per document.
        diversity=None,
        seed_topic_list=None, # Like CorEx
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        ctfidf_model=ctfidf_model,
        verbose=False
    )

    topics, probs = topic_model.fit_transform(docs) # Fit the model and predict documents.