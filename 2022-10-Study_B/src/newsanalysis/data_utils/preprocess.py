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
import re
import functools
import pandas as pd
import os
from loguru import logger
from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer
from sentence_transformers import SentenceTransformer
from transformers import BertTokenizerFast
try:
    from cuml.cluster import HDBSCAN
    logger.info(f'cuML HDBSCAN imported')
except ImportError:
    from hdbscan import HDBSCAN
    logger.info('Regular HSBDSCAN imported')
try:
    from cuml.dask.cluster import DBSCAN
    logger.info(f'cuML DBSCAN DASK imported')
except ImportError:
    logger.info(f'cuML DBSCAN DASK failed to import')
import pickle
import pandas as pd
import jsonlines
from time import perf_counter
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import faiss
import h5py
try:
    from cuml.manifold import UMAP
    logger.info('cuML UMAP imported')
except ImportError:
    from umap import UMAP
    logger.info('Regular UMAP imported')

from newsanalysis.dataviz.plots import retrieve_story_lens, retrieve_story_and_lang

def chunks(iter1, iter2, n):
    """Yield successive n-sized chunks from iter1 and iter2."""
    assert iter1.shape[0] == iter2.shape[0]
    for i, e in enumerate(range(0, iter1.shape[0], n)):
        yield (i, iter1[e:e + n], iter2[e:e + n])

def unique_story_ids(file):
    '''Get unique story ids from jsonl file'''
    unique_ids = set()
    with jsonlines.open(file, 'r') as reader:
        for story in reader.iter(skip_empty=True, skip_invalid=True):
            story_id = story.get('processed_stories_id')
            if story_id:
                unique_ids.add(story_id)
    return unique_ids

def story_iter(file, only_text = True, match_list = None, up_to = None, progress_check = None):
    """Iterator to yield stories from a jsonlines data file

    Args:
        file (str): jsonlines file to read
        only_text (bool, optional): whether to yield text or story id. Defaults to True.
        match_list (list/set, optional): list or set of ids to return only if matching. None means everything is yielded. Defaults to None.
        up_to (int, optional): Maximum number of stories to yield, for debugging. Defaults to None.
        progress_check (int, optional): Log a progress check every {progress_check} stories yielded. Defaults to None.

    Yields:
        _type_: _description_
    """

    # enforce match_list as a set
    if match_list is not None:
        match_list = set(match_list)

    # if up_to or progress check options present, then instantiate counter variable
    if up_to or progress_check:
        c = 0

    # begin iteration
    with jsonlines.open(file, 'r') as reader:
        for story in reader.iter(skip_empty=True, skip_invalid=True):
            # ignore query
            if 'query' in story:
                continue

            story_id = story.get('processed_stories_id')

            if (match_list and story_id in match_list) or not match_list:
                if up_to and c >= up_to:
                    break
                elif up_to or progress_check:
                    c += 1
                    if progress_check and c % progress_check == 0:
                        logger.info(f"Yielding story number {c}")
                if only_text:
                    yield story.get('text')
                else:
                    yield story_id

def remove_redundant_ids(file, savepath):
    logger.info('Removing duplicate stories by id. Keep the first one')
    logger.info(f'Input file is {file}')
    logger.info(f'Output file is {savepath}')
    present = set()
    with jsonlines.open(file, 'r') as reader:
        with jsonlines.open(savepath, 'w') as writer:
            for story in reader.iter(skip_empty=True, skip_invalid=True):
                if 'query' in story:
                    writer.write(story)
                story_id = story.get('processed_stories_id')
                if story_id not in present:
                    present.add(story_id)
                    writer.write(story)

def build_en_tokenizer(token_pattern=r"\b\w\w+\b"):
    """Return a function that splits a string into a sequence of tokens.
    Returns
    -------
    tokenizer: callable
            A function to split a string into a sequence of tokens.
    """
    token_pattern = re.compile(token_pattern)
    return token_pattern.findall


def split_and_tokenize(string, lang, tok, en_tok):
    '''Custom tokenizer to handle the multiple languages for sklearn vectorizers. Models have a cap of 512 for input length.'''

    if lang in ['zh', 'ja', 'ko', 'bn', 'hi', 'ta', 'gu', 'kn', 'te', 'ml', 'th', 'my', 'mr', 'vi', 'ne', 'lo', 'ur', 'si', 'or', 'ceb']:
        words = re.split('[ ，,".]', string)
        to_return = []
        for group in words:
            if len(group) > 510:
                logger.warning('Brute force applied')
                group = list([group[i:i+500] for i in range(0, len(group), 500)])
                for x in group:
                    to_return.append(tok.tokenize(x))
            else:
                to_return.append(tok.tokenize(group))
        # flatten
        to_return = [item for sublist in to_return for item in sublist]
        return to_return
    else:
        return en_tok(string)



def deduplicate(file, savepath, gpu=False):
    '''Return dict of story ids and their duplicates'''
    tokenizer = BertTokenizerFast.from_pretrained("sentence-transformers/LaBSE")
    en_tok    = build_en_tokenizer()

    # define params
    M = 32         # Number of connections that would be made for each new vertex during HNSW construction.
    nsegment = 16  # Number of segments for product quantization (number of subquantizers).
    nbit = 8       # Number of bits to encode each segment.
    batch_size = 10000 # batch size with which to cycle through vectors.
    k = 10 # number of nearest neighbours

    df = retrieve_story_and_lang(file)

    # Process by language
    grouped = df.groupby('lang').apply(lambda x: x['id'].unique())

    for l in df['lang'].unique():
        d = 10240       # Dimension (length) of vectors.
        # if l == 'en':
            # continue
        if l is None:
            logger.warning('Nonetype Language. Skipping...')
            continue
        elif l == '' or l.strip() == '':
            continue
        logger.info(f'Processing {l}')
        m_list = grouped.loc[l]
        logger.info(len(m_list))
        if len(m_list) < 256:
            logger.info('Cannot cluster below 256 due to IVFPQ. Continuing...')
            continue
        ordered_ids = np.array(list(story_iter(file, only_text = False, match_list=m_list)))
        if len(ordered_ids) != len(m_list):
            logger.warning('Length of ordered ids and match list not equal. Have you deduplicated by removing redundant story ids?')
        custom_tok = functools.partial(split_and_tokenize, lang=l, tok=tokenizer, en_tok=en_tok)
        vectorizer = TfidfVectorizer(
            analyzer='word',
            norm='l2',
            tokenizer=custom_tok,
            max_features=d
        )
        logger.info('start fit transform')
        t1_start = perf_counter()
        csr = vectorizer.fit_transform(story_iter(file, match_list=m_list))
        t1_stop = perf_counter()
        # fix d not matching for low vocabulary
        if csr.shape[1] < d:
            d = csr.shape[1]
        logger.info(f'end fit transform. Seconds taken: {t1_stop-t1_start:.2f}') 

        logger.info(f'Shape: {csr.shape}')

        logger.info('Starting FAISS')
        start=perf_counter()

        # cf. https://towardsdatascience.com/ivfpq-hnsw-for-billion-scale-similarity-search-89ff2f89d90e#9718


        # Create the index.
        nprobe = 100  # number of clusters to probe
        coarse_quantizer = faiss.IndexHNSWFlat(d, M)
        potential_nlist = int(np.floor(csr.shape[0]/40))
        if potential_nlist == 0:
            potential_nlist = csr.shape[0]
        nlist = min(10000, potential_nlist)  # Number of inverted lists (number of partitions or cells).
        logger.info(f'nlist is {nlist}')
        # N.B. https://github.com/facebookresearch/faiss/wiki/FAQ#can-i-ignore-warning-clustering-xxx-points-to-yyy-centroids 
        if nprobe > nlist:
            logger.info('nprobe was greater than nlist. set to equal.')
            nprobe = nlist
        try:
            index = faiss.IndexIVFPQ(coarse_quantizer, d, nlist, nsegment, nbit)
        except:
            logger.exception('Instantiating IndexIVFPQ Failed.')
            continue
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

        for _, sparse_vectors, ids in chunks(csr, ordered_ids, batch_size):
            index.add_with_ids(sparse_vectors.todense().astype(np.float32), ids)
        logger.info('Done adding rows to index')

        # Setting the number of partitions to search.
        index.nprobe = nprobe

        # xq are query vectors, for which we need to search in xb to find the k nearest neighbors.
        # The search returns D, the pairwise distances, and I, the indices of the nearest neighbors.
        D = np.zeros((csr.shape[0],k))
        I = np.zeros((csr.shape[0],k))
        logger.debug(f'Shape of D and I is {D.shape}')
        for num, sparse_vectors, ids in chunks(csr, ordered_ids, batch_size):
            logger.debug(f'{num}, {D[num*batch_size:num*batch_size+batch_size,:].shape}')
            D[num*batch_size:num*batch_size+batch_size,:], I[num*batch_size:num*batch_size+batch_size,:] = index.search(sparse_vectors.todense().astype(np.float32), k)

        savename = os.path.join(savepath, f'deduplicate_{"gpu" if gpu else "cpu"}.hdf5') 
        with h5py.File(savename, 'a') as f:
            f.attrs['d']          = d
            f.attrs['M']          = M
            f.attrs['nsegment']   = nsegment
            f.attrs['nbit']       = 8
            f.attrs['nprobe']     = 100
            f.attrs['batch_size'] = batch_size
            f.attrs['k']          = k

            g = f.require_group(l)
            g.attrs['nlist']      = nlist
            # clear previous datasets:
            for dsetname in ['D', 'I', 'ids']:
                if dsetname in g.keys():
                    del g[dsetname]
            g.create_dataset('D', data=D)
            g.create_dataset('I', data=I)
            g.create_dataset('ids', data=ordered_ids)

        stop=perf_counter()
        logger.info(f'End FAISS: {stop-start:.2f}s elapsed')

def remove_duplicates(dedup_faiss_file, original_file, savepath, skip_hdf5_read = False, threshold=0.35):
    '''Function to remove duplicates from faiss output. Saves ids to discard into savepath and a new cleaned datafile into the same directory as the original file.
    '''

    savename = os.path.join(savepath, 'deduplicate_discard_list.pkl')
    logger.info(f'Threshold is {threshold}')

    if not skip_hdf5_read:
        to_discard = set()
        do_not_discard = set()
        threshold=threshold

        with h5py.File(dedup_faiss_file, 'r') as f:
            for lang in f.keys():
                logger.info(f'Processing {lang}')
                D = f[lang]['D'][:]
                I = f[lang]['I'][:]
                ids = f[lang]['ids'][:]

                id_len = len(ids)
                for c in range(id_len):
                    if c%np.floor(id_len/10) == 0:
                        logger.info(f'{100*c/len(ids):.2f}% complete')
                    if ids[c] not in to_discard.union(do_not_discard):
                        do_not_discard.add(ids[c])
                    ids_to_process = I[c, D[c,:]<threshold]
                    for k in ids_to_process:
                        k = int(k)
                        if k not in do_not_discard and k != ids[c]:
                            to_discard.add(k)
        assert len(do_not_discard.intersection(to_discard)) == 0

        with open(savename, 'wb') as f:
            pickle.dump(to_discard, f)

    else:
        with open(savename, 'rb') as f:
            to_discard = pickle.load(f)
        logger.info(f'Discard list loaded in from {savename}')

    original_file_dir = os.path.dirname(original_file)
    sole_filename = os.path.split(original_file)[-1].split('.jsonl')[0]
    deduped_filename = os.path.join(original_file_dir, f'{sole_filename}_nodup.jsonl')
    with jsonlines.open(original_file, 'r') as reader:
        with jsonlines.open(deduped_filename, 'w') as writer:
            for story in reader.iter(skip_invalid=True, skip_empty=True):
                if 'query' in story:
                    writer.write(story)
                    continue
                id = int(story.get('processed_stories_id'))
                if id and id not in to_discard:
                    writer.write(story)

    logger.info(f'Written to {deduped_filename}')


def embed_docs(file, savepath, up_to = None, progress_check = None):
    assert os.path.isdir(savepath)

    embedding_model = SentenceTransformer("sentence-transformers/LaBSE")

    #Start the multi-process pool on all available CUDA devices
    pool = embedding_model.start_multi_process_pool()

    #Compute the embeddings using the multi-process pool
    emb = embedding_model.encode_multi_process(list(story_iter(
        file,
        only_text=True,
        up_to=up_to,
        progress_check=progress_check
    )), pool)
    logger.info("Embeddings computed. Shape:", emb.shape)

    unique_ids = list(story_iter(
        file,
        only_text=False,
        up_to = up_to,
        progress_check = None
    ))

    #Optional: Stop the proccesses in the pool
    embedding_model.stop_multi_process_pool(pool)

    # Store sentences & embeddings on disc
    savename = os.path.join(savepath,'embeddings.pkl')
    with open(savename, "wb") as fOut:
        pickle.dump({'ids': unique_ids, 'embeddings': emb}, fOut, protocol=pickle.HIGHEST_PROTOCOL)

    logger.info(f'Saved to {savename}')


# custom dbscan class for dask


def filter_by_cluster(file, savepath, embeddings = None, up_to=None, progress_check=None, dask = False):
    assert os.path.isdir(savepath)

    if dask:
        logger.info(f'Dask flag is TRUE')
        from dask_cuda import LocalCUDACluster
        from dask.distributed import Client
        class bertopic_compatible_dask_dbscan(DBSCAN):

            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)

            def predict(self, *args, **kwargs):
                self.labels_ = super().fit_predict(*args, **kwargs)

    # Step 1 - Extract embeddings.
    embedding_model = SentenceTransformer("sentence-transformers/LaBSE")

    # Step 2 - Reduce dimensionality.
    umap_model = UMAP(
        n_neighbors=10,
        n_components=5,
        min_dist=0.0,
        metric='cosine',
        random_state=42,
        verbose=True
    )

    # Step 3 - Cluster reduced embeddings.
    if dask:
        cluster = LocalCUDACluster(threads_per_worker=1)
        logger.info(f'CUDA VISIBLE DEVICES: {cluster.cuda_visible_devices}')
        client = Client(cluster)
        dbscan_model = bertopic_compatible_dask_dbscan(
            verbose=True,
            client=client,
            min_samples=15,
            output_type='numpy',
            max_mbytes_per_batch = 10000
        )
    else:
        dbscan_model = HDBSCAN(
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
        language='multilingual', # Set to 'multilingual' for datasets with languages other than English. N.B. documentation says this is not used if embedding_model is provided
        top_n_words=10,
        n_gram_range=(1, 1),
        min_topic_size=10,
        nr_topics=None,
        low_memory=False,
        calculate_probabilities=True, # The probabilities of all topics per document.
        seed_topic_list=None, # Like CorEx
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=dbscan_model,
        vectorizer_model=vectorizer_model,
        ctfidf_model=ctfidf_model,
        verbose=True
    )
    if embeddings is not None:
        logger.info(f'Embeddings shape: {embeddings.shape}')
    topics, probs = topic_model.fit_transform(list(story_iter(
        file,
        only_text=True,
        up_to=up_to,
        progress_check=progress_check
    )),
    embeddings=embeddings) # Fit the model and predict documents.

    topic_model_savename = os.path.join(savepath, 'topic_model.pkl')
    with open(topic_model_savename, 'wb') as f:
        pickle.dump(topic_model, f)
    logger.info(f'Saved to {topic_model_savename}')

    return topics, probs