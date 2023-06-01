'''
Script to preprocess obtained and enriched MediaCloud data. The enriching steps are:
* deduplicate
* remove stories that are too short
* remove stories that contain irrelevant information
* remove irrelevant stories
    * this might be with, e.g. words that often indicate excluion: 'SUMMARY:' etc.
    * BERTopic?

'''

from pathlib import Path
import numpy as np
import re
import functools
import pandas as pd
import os
from csv import DictWriter
from loguru import logger
from datasets import Dataset
from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer
import nltk
from sentence_transformers import SentenceTransformer
from transformers import BertTokenizerFast, AutoTokenizer, AutoModelForTokenClassification, pipeline, AutoModelForSequenceClassification
import torch
from torch.utils.data import DataLoader
try:
    from cuml.cluster import HDBSCAN
    # logger.info(f'cuML HDBSCAN imported')
except ImportError:
    from hdbscan import HDBSCAN
    # logger.info('Regular HSBDSCAN imported')
import pickle
import glob
import pandas as pd
import jsonlines
from time import perf_counter
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import faiss
import h5py
try:
    from cuml.manifold import UMAP
    # logger.info('cuML UMAP imported')
except ImportError:
    from umap import UMAP
    # logger.info('Regular UMAP imported')

from ..dataviz import retrieve_story_and_lang, retrieve_story_lens

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

def story_iter(file, only_text = True, match_list = None, up_to = None, progress_check = None, full_object = False):
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
                elif full_object:
                    yield story
                else:
                    yield story_id

def remove_redundant(file, savepath, by='id'):
    valid_by = ['id', 'url']
    if by not in valid_by:
        raise ValueError(f'by must be in {valid_by}. "{by}" was given')
    logger.info(f'Removing redundant stories by {by}. Keep the first one')
    logger.info(f'Input file is {file}')
    logger.info(f'Output file is {savepath}')
    present = set()
    unwritten = 0
    with jsonlines.open(file, 'r') as reader:
        with jsonlines.open(savepath, 'w') as writer:
            for story in reader.iter(skip_empty=True, skip_invalid=True):
                if 'query' in story:
                    writer.write(story)
                if by == 'id':
                    story_id = story.get('processed_stories_id')
                    if story_id not in present:
                        present.add(story_id)
                        writer.write(story)
                    else:
                        unwritten += 1
                elif by == 'url':
                    story_url = story.get('url')
                    if story_url not in present:
                        present.add(story_url)
                        writer.write(story)
                    else:
                        unwritten += 1
    logger.info(f'{unwritten} stories unwritten/discarded')

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

        with h5py.File(dedup_faiss_file, 'r') as f:
            for lang in f.keys():
                logger.info(f'Processing {lang}')
                D = f[lang]['D'][:]
                I = f[lang]['I'][:]
                ids = f[lang]['ids'][:]

                to_discard.update(
                    set(I[np.logical_and(
                        np.logical_not(np.equal(I, ids.reshape(-1,1))),
                        D<threshold
                    )].astype(int))
                )

        logger.info(f'Length of discard list is {len(to_discard)}')

        with open(savename, 'wb') as f:
            pickle.dump(to_discard, f)
        logger.info(f'Discard list saved to {savename}')

    else:
        with open(savename, 'rb') as f:
            to_discard = pickle.load(f)
        logger.info(f'Discard list loaded in from {savename}')

    original_file_dir = os.path.dirname(original_file)
    sole_filename = os.path.split(original_file)[-1].split('.jsonl')[0]
    deduped_filename = os.path.join(original_file_dir, f'{sole_filename}_nodup.jsonl')
    with jsonlines.open(original_file, 'r') as reader:
        with jsonlines.open(deduped_filename, 'w') as writer:
            for counter, story in enumerate(reader.iter(skip_invalid=True, skip_empty=True)):
                if counter>0 and counter %100000 == 0:
                    logger.info(f'Processed {counter} stories')
                if 'query' in story:
                    writer.write(story)
                    continue
                id = int(story.get('processed_stories_id'))
                if id and id not in to_discard:
                    writer.write(story)

    logger.info(f'Written to {deduped_filename}')

def embed_docs(file, savepath, up_to = None, progress_check = None):
    assert os.path.isdir(savepath), savepath

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
    logger.info(f"Embeddings computed. Shape: {emb.shape}")

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

def filter_by_cluster(file, savepath, embeddings = None, up_to=None, progress_check=None, nr_topics = None, top_n_words = 10, save_without_gpu_elements = True):
    assert os.path.isdir(savepath)
    # Step 1 - Extract embeddings.
    logger.info(f'nr topics set to {nr_topics}')
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
    dbscan_model = HDBSCAN(
        min_cluster_size=100,
        min_samples=50,
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
        top_n_words=top_n_words,
        n_gram_range=(1, 1),
        min_topic_size=10,
        nr_topics=nr_topics,
        low_memory=False,
        calculate_probabilities=False, # The probabilities of all topics per document. Might need to set to false for gpu https://github.com/rapidsai/cuml/issues/5127. see also https://github.com/rapidsai/cuml/issues/4879
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
    _ = topic_model.fit_transform(list(story_iter(
        file,
        only_text=True,
        up_to=up_to,
        progress_check=progress_check
    )),
    embeddings=embeddings) # Fit the model and predict documents.

    # cf. https://stackoverflow.com/questions/74860769/loading-a-gpu-trained-bertopic-model-on-cpu about saving and loading with a GPU and loading onto a computer with CPU
    topic_model_savename = os.path.join(savepath, f'topic_model_{nr_topics}.bertopic')
    # with open(topic_model_savename, 'wb') as f:
    #     pickle.dump(topic_model, f)
    if save_without_gpu_elements:
        del topic_model.hdbscan_model
        del topic_model.umap_model
    topic_model.save(topic_model_savename, save_embedding_model=False)
    logger.info(f'Saved to {topic_model_savename}')

    return None

def remove_by_len(file, lo = None, hi = None):
    '''Function to remove articles below and above a certain length
    '''

    if lo is None and hi is None:
        logger.info(f'No lo or hi boundary provided, returning...')
        return None
    logger.info(f'lo is {lo}')
    logger.info(f'hi is {hi}')
    original_file_dir = os.path.dirname(file)
    sole_filename = os.path.split(file)[-1].split('.jsonl')[0]
    deduped_filename = os.path.join(original_file_dir, f'{sole_filename}{f"_nolo{lo}" if lo is not None else ""}{f"_nohi{hi}" if hi is not None else ""}.jsonl')
    unwritten = 0
    with jsonlines.open(file, 'r') as reader:
        with jsonlines.open(deduped_filename, 'w') as writer:
            for story in reader.iter(skip_invalid=True, skip_empty=True):
                if 'query' in story:
                    writer.write(story)
                    continue
                length = len(story.get('text'))
                if lo is not None:
                    if length <= lo:
                        unwritten += 1
                        continue
                if hi is not None:
                    if length <= hi:
                        unwritten += 1
                        continue
                writer.write(story)

    logger.info(f'{unwritten} stories unwritten/discarded')
    logger.info(f'Written to {deduped_filename}')

def export_to(data_file, source = None, outpath = None, id = None, format='txt', count = None) -> None:
    if outpath is None and source is None:
        outpath = Path(data_file).parent
    elif source is not None:
        outpath = Path(source).parent / Path(source).stem
        os.makedirs(outpath, exist_ok=True)
        logger.info(f'Outpath created is {outpath}')
    else:
        outpath = Path(outpath)

    if count is None and source is None:
        with jsonlines.open(data_file, 'r') as reader:
            for story in reader.iter(skip_invalid=True, skip_empty=True):
                if 'query' in story:
                    continue
                if id is None or id == story.get('processed_stories_id'):
                    outfile = outpath / f'{story.get("processed_stories_id")}.{format}'
                    with open(outfile, 'w') as f:
                        f.write(story.get('text'))
                    logger.info(f'Saved to {outfile}')
                    return
    elif count is not None and format == 'csv':
        outfile = outpath / f'dedoose_ready_{count}.csv'
        with jsonlines.open(data_file, 'r') as reader:
            with open(outfile, 'w', newline='') as outp:
                writer = DictWriter(outp, fieldnames=[
                    'collect_date',
                    'language',
                    'media_id',
                    'media_name',
                    'media_url',
                    'processed_stories_id',
                    'publish_date',
                    'Text',
                    'Title',
                    'url'
                ], extrasaction='ignore')
                writer.writeheader()
                for counter, story in enumerate(reader.iter(skip_invalid=True, skip_empty=True)):
                    if 'query' in story:
                        continue
                    if counter == count:
                        logger.info(f'{count} stories written, exiting...')
                        return
                    story['Text'] = story['text']
                    story['Title'] = story['title']
                    writer.writerow(story)
    elif source is not None and format == 'csv':
        ids = pd.read_csv(source)['id'].unique()
        outfile = outpath / f'dedoose_descriptors_{Path(source).stem}.csv'
        logger.info(f'outfile is {outfile}')
        with jsonlines.open(data_file, 'r') as reader:
            with open(outfile, 'w', newline='', encoding='utf-8') as outp:
                writer = DictWriter(outp, fieldnames=[
                    'collect_date',
                    'language',
                    'media_id',
                    'media_name',
                    'media_url',
                    'publish_date',
                    'Title',
                    'title_text',
                    'url'
                ], extrasaction='ignore', delimiter='\t')
                writer.writeheader()
                for counter, story in enumerate(reader.iter(skip_invalid=True, skip_empty=True)):
                    if 'query' in story:
                        continue
                    if story.get('processed_stories_id') in ids:
                        # story['Text'] = story['text']
                        story['title_text'] = story['title']
                        story['Title'] = f"{story['processed_stories_id']}.txt"
                        writer.writerow(story)
    elif source is not None and format == 'txt':
        ids = pd.read_csv(source)['id'].unique()
        with jsonlines.open(data_file, 'r') as reader:
            for story in reader.iter(skip_invalid=True, skip_empty=True):
                if 'query' in story:
                    continue
                if story.get('processed_stories_id') in ids:
                    outfile = outpath / f'{story.get("processed_stories_id")}.{format}'
                    with open(outfile, 'w') as f:
                        f.write(story.get('text'))


def sample(data_file, savepath, by=None, total=20, lang=None, with_text = True, 
exclude = None, min_date = None, max_date = None):
    to_exclude = []
    if exclude:
        for i in exclude:
            to_exclude.append(pd.read_csv(i))
    df_records = []
    for story in story_iter(data_file, only_text= False, full_object=True):
        df_records.append(
            {
                'id': story.get('processed_stories_id'),
                'lang': story.get('language'),
                'publish_date': story.get('publish_date'),
                'media_id': story.get('media_id'),
                # 'text': story.get('text') if with_text else None
            }
        )
    df = pd.DataFrame.from_records(df_records)
    df['media_count'] = df.groupby('media_id')['media_id'].transform('count')
    df['date'] = pd.to_datetime(df['publish_date'])
    df['year'] = df['date'].dt.year

    if lang:
        assert isinstance(lang, tuple)
        logger.info(f'Filtering to lang list {lang}')
        df = df[df['lang'].isin(lang)]

    # each_group_min = np.ceil(total/len(lang))
    if exclude:
        for i in to_exclude:
            df = df[~df['id'].isin(i['id'])]
        logger.info('Exclusion applied')

    if min_date:

        # filter dataframe by dates later than a given date
        df = df[df['date'] >= pd.to_datetime(min_date)]
        logger.info(f'{min_date} applied')

    if max_date:
        df = df[df['date'] <= pd.to_datetime(min_date)]
        logger.info(f'{max_date} applied')



    intermediate_df = df.groupby(['lang', 'year']).filter(lambda x: len(x) >= total)
    sampled_df = intermediate_df.groupby(['lang', 'year']).sample(n=total, weights=df['media_count'], random_state=1)

    savename = Path(savepath) / f"dedoose_sampled_{Path(data_file).stem}{'_' if lang else ''}{'_'.join(lang)}{f'_min{min_date}' if min_date else ''}{f'_min{max_date}' if max_date else ''}.csv"

    sampled_df.to_csv(savename)
    logger.info(f'saved to {savename}')


def remove_by_bt(data_file, bertopic_file, outfile, remove):


    # records which data and bertopic file
    logger.info(f'Data file: {data_file}')
    logger.info(f'BERTopic file: {bertopic_file}')

    # we need at least one group to remove
    assert len(remove) > 0, 'Please specify at least one topic to remove'
    logger.info(f'TOPICS SET TO BE REMOVED: {remove}')

    # load in bertopic file
    embedding_model = SentenceTransformer("sentence-transformers/LaBSE")
    topic_model = BERTopic.load(bertopic_file, embedding_model=embedding_model)
    logger.info('BERTopic loaded in')

    # get stories and langs data from data_file
    df = retrieve_story_and_lang(data_file)
    logger.info('Story ids loaded in')

    # check that the number of stories in bertopic file match that of the bertopic result
    assert len(df['id'].unique()) == len(topic_model.topics_), 'Are you sure this BERTopic was run on this data file? The number of labels in BERTopic do not match the length of this data file'

    # check that all the topics set for removal are within the BERTopic result
    max_topic = topic_model.get_topic_freq()['Topic'].max()
    assert all([i<=max_topic for i in remove]), 'At least one topic set to be removed is not in the topics present. Please revise'

    # generate set of story ids to be discarded
    df['topic'] = topic_model.topics_
    to_discard = set(df[df['topic'].isin(remove)]['id'])

    # write to result
    skipped = 0
    with jsonlines.open(data_file, 'r') as reader:
        with jsonlines.open(outfile, 'w') as writer:
            for story in reader.iter(skip_invalid=True, skip_empty=True):
                if 'query' in story:
                    writer.write(story)
                elif story.get('processed_stories_id') in to_discard:
                    skipped += 1
                    continue
                else:
                    writer.write(story)

    logger.info(f'Saved to {outfile}')

    logger.info(f'To discard: {len(to_discard)}')
    logger.info(f'Skipped: {skipped} ')
    assert skipped == len(to_discard)


def chunk_text(text, max_length=500):
    words = nltk.word_tokenize(text)

    chunks = []
    current_chunk = []
    current_length = 0

    for word in words:
        # +1 is to account for a space
        if current_length + len(word) + 1 <= max_length:
            current_length += len(word) + 1
            current_chunk.append(word)
        else:
            # Join the current chunk into a string and add it to the list of chunks
            chunks.append(' '.join(current_chunk))
            # Start a new chunk with the current word
            current_chunk = [word]
            current_length = len(word)

    # Don't forget to add the last chunk
    chunks.append(' '.join(current_chunk))

    return chunks

def jsonl_to_dataset(jsonl_file, dataset_out, keys_to_read = ['text', 'processed_stories_id'], splitting= False, up_to=None):
    '''Convert JSONL to dataset object ready for ML interface'''

    data = []
    logger.info(f'Keys to read are {keys_to_read}')

    count = 0
    # Read data from the JSONLines file line by line
    for story in story_iter(jsonl_file, only_text = False, full_object=True):

        if up_to:
            if count >= up_to:
                break
        count += 1
        if count % 10000 == 0:
            logger.info(f'{count} processed.')

        if 'query' in story:
            metadata = story
            continue

        if keys_to_read:
            # Select the specified keys from the JSON object
            selected_data = {key: story.get(key) for key in keys_to_read}
        else:
            selected_data = story

        # check split
        sent_tok_dict = {
            'cs': 'czech',
            'da': 'danish',
            'nl': 'dutch',
            'en': 'english',
            'et': 'estonian',
            'fi': 'finnish',
            'fr': 'french',
            'de': 'german',
            'el': 'greek',
            'it': 'italian',
            'no': 'norwegian',
            'pl': 'polish',
            'pt': 'portuguese',
            'ru': 'russian',
            'sl': 'slovene',
            'es': 'spanish',
            'sv': 'swedish',
            'tr': 'turkish'
        }
        if splitting:
            # begin part counter to unique ids for every split of a doc
            part_counter = 0

            if story.get('language') in sent_tok_dict:
                # nltk.download('punkt')
                sentences = nltk.sent_tokenize(story.get('text'), language = sent_tok_dict[story.get('language')])
                for sentence in sentences:
                    if len(nltk.word_tokenize(sentence)) > 500:
                        logger.debug('Brute force applied')
                        chunked = chunk_text(sentence)
                        for x in chunked:
                            to_append = selected_data.copy()
                            to_append['text'] = x
                            to_append['part_id'] = f'{selected_data.get("processed_stories_id")}_{part_counter}'
                            part_counter += 1
                            data.append(to_append)
                    else:
                        to_append = selected_data.copy()
                        to_append['text'] = sentence
                        to_append['part_id'] = f'{selected_data.get("processed_stories_id")}_{part_counter}'
                        part_counter += 1
                        data.append(to_append)
            else: 
            # #story.get('language') in ['zh', 'ja', 'ko', 'bn', 'hi', 'ta', 'gu', 'kn', 'te', 'ml', 'th', 'my', 'mr', 'vi', 'ne', 'lo', 'ur', 'si', 'or', 'ceb']:
                words = re.split('[ ，,".]', story.get('text'))
                # to_return = []
                for group in words:
                    if len(group) > 500:
                        # logger.warning('Brute force applied')
                        chunks = []
                        for i in range(0, len(group), 500):
                            chunks.append(group[i:i+500])
                        for x in chunks:
                            to_append = selected_data.copy()
                            to_append['text'] = x
                            to_append['part_id'] = f'{selected_data.get("processed_stories_id")}_{part_counter}'
                            part_counter += 1
                            data.append(to_append)
                    else:
                        to_append = selected_data.copy()
                        to_append['text'] = group
                        to_append['part_id'] = f'{selected_data.get("processed_stories_id")}_{part_counter}'
                        part_counter += 1
                        data.append(to_append)
        else:
            data.append(selected_data)

    # Create a dataset from the list of dictionaries
    dataset = Dataset.from_list(data)

    dataset.save_to_disk(dataset_out)
    logger.info(f'Saved to {dataset_out}')



def combine_person_tags(indexed_iob2_sequence, iper_id = 0):
    entities = []
    current_entity_tokens = []
    for index, token, tag in indexed_iob2_sequence:
        # If token starts with '_' or tag is not 'I-PER', add current_entity_tokens to entities
        if token.startswith('_') or tag != iper_id:
            if current_entity_tokens:
                entities.append(''.join([t.lstrip('_') for t in current_entity_tokens]))
                current_entity_tokens = []
        # Add the token to current_entity_tokens (removing the '_' if it's there)
        current_entity_tokens.append(token.lstrip('_'))
    # Don't forget to add the last entity to the list
    if current_entity_tokens:
        entities.append(''.join([t for t in current_entity_tokens]))
    entities = [i.replace('▁', ' ').replace("_", " ").strip() for i in entities]
    return entities

def annotate(dataset_path,
             outpath,
             model = "julian-schelb/roberta-ner-multilingual/",
             tok = None,
             num_batches=None,
             kind = 'ner',
             max_length=512,
             batch_size_per_gpu=800,
             from_batch = None
             ):

    logger.info(f'Model: {model}')
    logger.info(f'Annotation type: {kind}')
    logger.info(f'Batch size per GPU: {batch_size_per_gpu}')
    logger.info(f'Savepath: {outpath}')

    os.makedirs(Path(outpath).absolute(), exist_ok = True)

    if tok is None:
        tok = model

    #load model
    if kind == 'ner':
        annot_model = AutoModelForTokenClassification.from_pretrained(model)
        annot_tokenizer = AutoTokenizer.from_pretrained(model, add_prefix_space=True)
    else:
        annot_model = AutoModelForSequenceClassification.from_pretrained(model)
        annot_tokenizer = AutoTokenizer.from_pretrained(tok)

    # Create a Dataset object from the data generator function
    logger.info(f'Loading Dataset.')
    dataset = Dataset.load_from_disk(dataset_path)
    logger.info('Loading Dataset Complete.')

    # tokenized_dataset = dataset.map(lambda examples: annot_tokenizer(examples['text'], return_tensors='pt', padding=True, truncation=True, max_length=512), batched=True)

    # Move model to GPU if available
    # label_dict = annot_model.config.id2label
    label2id = annot_model.config.label2id
    iper_id = label2id['I-PER']
    assert isinstance(iper_id, int)
    if kind == 'ner':
        ids_of_interest = [v for k,v in label2id.items() if k in ['I-PER', 'B-PER']]
    if torch.cuda.device_count() > 1:
        logger.info('Multiple GPUs detected, applying torch.nn.DataParallel')
        annot_model = torch.nn.DataParallel(annot_model)
        # label_dict = annot_model.module.config.id2label
        batch_size = batch_size_per_gpu*torch.cuda.device_count()
    else:
        batch_size = batch_size_per_gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    annot_model = annot_model.to(device)
    logger.info(f'Adjusted batch size with GPU count of {torch.cuda.device_count()}: {batch_size}')

    if num_batches:
        logger.info(f'Maximum batch number specified: {num_batches}')

    if kind == 'ner':
        result = []
    elif kind == 'relevance':
        result = {}

    batch_iterator = list(enumerate(range(0, len(dataset), batch_size)))
    for counter, i in batch_iterator:
        if from_batch and counter < from_batch:
            continue
        batch_result = []
        # if counter % 10000 == 0:
        if counter % 100 == 0:
            logger.info(f'Processing batch {counter}: {counter/len(batch_iterator)*100:.2f}%')
        batch = dataset[i: i + batch_size]
        batch_ids = batch['processed_stories_id']
        batch_texts = batch['text']

        inputs = annot_tokenizer(batch_texts, padding='max_length', truncation= True, return_tensors='pt', max_length=max_length).to(device)

        # Move the inputs to device
        # inputs = {name: tensor.to(device) for name, tensor in inputs.items()}

        with torch.no_grad():
            # Forward pass
            outputs = annot_model(**inputs)

        if  kind == 'ner_new':
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            for j, prediction in enumerate(predictions):
                # tokens = annot_tokenizer.convert_ids_to_tokens(inputs['input_ids'][j])
                tokens = np.array(inputs[j].tokens)

                # Append to result
                batch_result.append({
                    'processed_stories_id': batch_ids[j],
                    'tokens': tokens,
                    'prediction': prediction
                    # 'original': filtered_tokens_labels
                })

                with open(Path(outpath) / f'ner_batch_{counter}.pkl', 'wb') as f:
                    pickle.dump(batch_result, f)

        elif kind == 'ner':
            # Convert logits into labels
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            for j, prediction in enumerate(predictions):
                # tokens = annot_tokenizer.convert_ids_to_tokens(inputs['input_ids'][j])
                tokens = np.array(inputs[j].tokens)

                # labels = [label_dict[label_id.item()] for label_id in prediction]
                # get indices of tokens we care about
                indices_of_relevant_labels = torch.where(prediction == torch.tensor(ids_of_interest).to(device))[0].cpu().numpy().astype(int)

                tokens_of_relevant_labels = tokens[indices_of_relevant_labels]

                filtered_tokens_labels = [(index, token, id) for index, token, id in zip(
                    indices_of_relevant_labels,
                    tokens_of_relevant_labels,
                    prediction[indices_of_relevant_labels]
                )]

                # Filter out tokens that are not 'I-PER' or 'B-PER'
                # filtered_tokens_labels = [(index, token, label) for index, token, label in zip(range(len(tokens)), tokens, labels) if (label in ['I-PER', 'B-PER'] and token != '<pad>')]


                # Append to result
                batch_result.append({
                    'processed_stories_id': batch_ids[j],
                    'NER': combine_person_tags(filtered_tokens_labels, iper_id = iper_id)
                })

                with open(Path(outpath) / f'ner_batch_{counter}.pkl', 'wb') as f:
                    pickle.dump(batch_result, f)

        elif kind == 'relevance':
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)

            for i, j in zip(batch['part_id'], predictions):
                result[i] = j

        if num_batches and (counter == num_batches - 1):
            logger.info(f'REACHED SPECIFIED MAX OF {num_batches} BATCHES')
            break


    if kind == 'relevance':
        with open(outpath, 'wb') as f:
            pickle.dump(result, f)

    logger.info('Complete')
    return None


def collate_ner(ner_batch_dir, outpath, omit_tokens = ['<pad>']):
    '''Function to collate batched ner results into one dataset file
    '''

    # check directory
    assert os.path.isdir(ner_batch_dir)

    # obtain files
    filenames = glob.glob(os.path.join(ner_batch_dir, 'ner_batch*.pkl'))

    # function to sort glob data
    def extract_number(filename):
        # Extract the number using regular expressions (regex)
        match = re.search(r'\d+', filename)
        # If a number is found, return it as an integer
        if match:
            return int(match.group())
        else:
            return 0

    # function to extend list
    def extend_set_with_list(existing_set=None, list_of_values=None, omit_tokens = None):
        # If existing_set is None, initialize it as an empty set
        if existing_set is None:
            existing_set = set()

        # omit tokens
        if omit_tokens:
            for tok_to_omit in omit_tokens:
                list_of_values = [i.replace(f'{tok_to_omit}', '') for i in list_of_values]

        # If list_of_values is not None, add its elements to the set
        if list_of_values is not None:
            existing_set.update(list_of_values)

        return existing_set

    # Sort the filenames based on the number
    filenames_sorted = sorted(filenames, key=extract_number)

    combined_result = {}
    for counter, file in enumerate(filenames_sorted):
        if counter % 100 == 0:
            logger.info(f'Processing {counter} of {len(filenames_sorted)}: {100*counter/len(filenames_sorted):.1f}%')
        with open(file,'rb') as f:
            data = pickle.load(f)

        for item in data:
            combined_result[item['processed_stories_id']] = extend_set_with_list(
                existing_set=combined_result.get(item['processed_stories_id']).get('NER'),
                list_of_values=item['NER'],
                omit_tokens = omit_tokens
            )

    output = Dataset.from_dict(combined_result)

    output.save_to_disk(outpath)
