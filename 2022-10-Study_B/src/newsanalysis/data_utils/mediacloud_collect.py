"""Collect MediaCloud data based on a query, desired outfile, start date and end date.

"""
import mediacloud.error
import mediacloud.api
import datetime
import os
import time
import pandas as pd
import click
import requests
import json
import jsonlines
from loguru import logger
import pickle

from . import constants

@click.group(help='Commands relating to MediaCloud')
@click.pass_context
def mediac(ctx):
    pass

@mediac.command()
@click.argument('publisher_csv')
@click.option('--outfile', '-o', required=True)
def mediainfo(publisher_csv, outfile):
    '''Collect media list data from a publisher csv file'''
    pubs = pd.read_csv(publisher_csv)
    # INDIA_NATIONAL_COLLECTION = 34412118
    # SOURCES_PER_PAGE = 100  # the number of sources retrieved per page
    # mc_directory = mediacloud.api.DirectoryApi(constants.API_KEY)
    sources = []

    # # while True:
    # for row in pubs.itertuples():
    #     if row.Index % 10 == 0:
    #         logger.info(f'Processing {row.Index+1} of {len(pubs)}: {(row.Index+1)/len(pubs):.1f}%')
    #     offset = 0   # offset for paging through
    #     while True:
    #         # grab a page of sources in the collection
    #         # response = mc_directory.source_list(collection_id=INDIA_NATIONAL_COLLECTION, limit=SOURCES_PER_PAGE, offset=offset)
    #         response = mc_directory.source_list(
    #             name=row.name.strip(),
    #             limit=SOURCES_PER_PAGE,
    #             offset=offset
    #         )
    #         check = [i.get('id') for i in response['results']]
    #         if row.id in check:
    #             # add it to our running list of all the sources in the collection
    #             sources.extend(response['results'])
    #             break
    #         # if there is no next page then we're done so bail out
    #         if response['next'] is None:
    #             logger.warning(f'WARNING: {row.id} ({row.name}) not found in API call.')
    #             break
    #         # otherwise setup to fetch the next page of sources
    #         offset += len(response['results'])

    for row in pubs.itertuples():
        if row.Index % 100 == 0:
            logger.info(f'Processing {row.Index+1} of {len(pubs)}: {100*(row.Index+1)/len(pubs):.1f}%')
        try:
            x = requests.get(url = f'https://api.mediacloud.org/api/v2/media/single/{row.id}', params= {'key': constants.V2_API_KEY}, timeout=60)
            sources.append(x.json())
        except:
            logger.warning(f'WARNING: {row.id} ({row.name}) not found in API call.')

    with open(outfile, 'wb') as f:
        pickle.dump(sources, f)

    logger.info("Sources collected: {}".format(len(sources)))



    # convert to df
    # sources_df = pd.DataFrame.from_records(sources)
    # sources_df.to_csv(outfile)
    # check
    # valid = pubs['id'].isin(sources_df['id'])
    # print(f'Numer of pubs in sources scraped: {100*valid.sum()/len(valid):.1f}%')

@mediac.command()
@click.argument('file')
def consolidatemc(file):
    '''Function to consolidate and process mediacloud source info, from mediainfo command.'''

    with open(file, 'rb') as f:
        raw_mc_source_info = pickle.load(f)

    result = {}
    for source in raw_mc_source_info:
        for tag_info in source['media_source_tags']:
            if tag_info['tag_set'] ==  'pub_country':
                result[source.get('media_id')] = tag_info['tag']
                break
            elif tag_info['tag_set'] == 'geographic_collection':
                result[source.get('media_id')] = tag_info['tag']
                break
            elif tag_info['tag_set'] == 'emm_country':
                result[source.get('media_id')] = tag_info['tag']
                break
            elif tag_info['tag_set'] in ['mexico_state', 'portuguese_state', 'portuguese_media_type', 'egypt_media_type', 'kenya_media_source', 'gv_country', 'usnewspapercirculation']:
                result[source.get('media_id')] = tag_info['tag']
                break
            elif tag_info['tag_set'] == 'subject_country':
                result[source.get('media_id')] = tag_info['tag']
                break


@mediac.command()
def profile():
    url = 'https://api.mediacloud.org/api/v2/auth/profile'
    response = requests.get(url=url, params={
        "key":'1dfb1fc62779662c965c8b197cae48c05eae20d8cd2403f0296513d9e56a33e5'
    })
    print(json.dumps(response.json(),indent=4))

@mediac.command()
@click.option('--outfile', required=True)
@click.option('--query', required=True, type=str)
@click.option('--start', help='In format YYYY-MM-DD', required=True)
@click.option('--end', help='In format YYYY-MM-DD', required=True)
@click.option('--count', help='Just get count for the query specified. Does not log, just prints to console.', is_flag=True)
def story_collect(
    outfile,
    query,
    start,
    end,
    count
):

    # if just getting a count, no need to log.
    if not count:
        pass
        # logging_dict = {
        #     'NONE': None,
        #     'CRITICAL': logging.CRITICAL,
        #     'ERROR': logging.ERROR,
        #     'WARNING': logging.WARNING,
        #     'INFO': logging.INFO,
        #     'DEBUG': logging.DEBUG
        # }

        # logging_level = logging_dict[log_level]

        # if logging_level is not None:

        #     logging_fmt   = '[%(levelname)s] %(asctime)s - %(name)s - %(message)s'
        #     today_datetime = str(datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S'))
        #     if log_dir is not None:
        #         assert os.path.isdir(log_dir)
        #         logging_file  = os.path.join(log_dir, f'{os.path.split(outfile)[-1].split(".")[0]}_mediacloud_collect.log')

        #     if log_handler_level == 'both':
        #         handlers = [
        #             logging.FileHandler(filename=logging_file,mode='w+'),
        #             logging.StreamHandler()
        #         ]
        #     elif log_handler_level == 'file':
        #         handlers = [logging.FileHandler(filename=logging_file,mode='w+')]
        #     elif log_handler_level == 'stream':
        #         handlers = [logging.StreamHandler()]
        #     logging.basicConfig(
        #         handlers=handlers,
        #         format=logging_fmt,
        #         level=logging_level,
        #         datefmt='%m/%d/%Y %I:%M:%S %p'
        #     )
        #     logger = logging.getLogger(__name__)
    else:
        print('ONLY GETTING COUNT')

    # Set up mediacloud client
    mc = mediacloud.api.MediaCloud(constants.API_KEY)

    # define start and end dates from inputs
    START_DATE = datetime.datetime.strptime(start, '%Y-%m-%d').date()
    END_DATE = datetime.datetime.strptime(end, '%Y-%m-%d').date()

    # set fetch size
    fetch_size = 1000
    total_collected = 0
    max_retries = 10
    retry_count = 0

    # if the outfile already exists, then resume from the last collected point
    if os.path.isfile(outfile) and not count:
        max_id = 0
        with jsonlines.open(outfile, 'r') as f:
            for line in f:
                # check that the query is the same before proceeding
                if 'query' in line:
                    existing_query = line.get('query')
                    if query != existing_query:
                        raise Exception('The queries between the existing outfile and this input do not match!')
                line_id = line.get('processed_stories_id',0)
                if line_id > max_id:
                    max_id = line_id
        last_processed_stories_id = max_id
        logger.info(f'Outfile at {outfile} detected. Reading in latest processed story id as {last_processed_stories_id}')
    else:
        last_processed_stories_id = 0

    if count:
        count_result = mc.storyCount(
            query,
            solr_filter=mc.dates_as_query_clause(
                START_DATE,
                END_DATE
            )
        )
        print(f'QUERY: {query}')
        print(f'Count result for this query is {count_result}')
        return None

    # begin timing
    start_time = time.time()

    logger.info(f'Collecting to {outfile}')
    with jsonlines.open(outfile, 'w') as f:
        fetched_stories = [0]

        #write first line of json file
        f.write({
            "query": query,
            "date_of_collection": datetime.datetime.now().strftime('%Y-%m-%d'),
            "start_date": START_DATE.strftime('%Y-%m-%d'),
            "end_date": END_DATE.strftime('%Y-%m-%d')
        })

        while len(fetched_stories) > 0 and retry_count < max_retries:
            try:
                fetched_stories = mc.storyList(
                    query,
                    solr_filter=mc.dates_as_query_clause(
                        START_DATE,
                        END_DATE
                    ),
                    last_processed_stories_id=last_processed_stories_id,
                    rows = fetch_size
                )
                last_processed_stories_id = fetched_stories[-1]['processed_stories_id']
                for story in fetched_stories:
                    f.write(story)
                total_collected += len(fetched_stories)
                logger.info(f'Collected {total_collected} after {time.time()-start_time:.2f} seconds up to date {fetched_stories[-1].get("publish_date")}')
                retry_count = 0
            except Exception as e:
                retry_count += 1
                logger.warning('Error encountered:')
                logger.warning(e)
                time.sleep(60)
                continue
        if retry_count >= max_retries:
            logger.warning(f'Max number of retries ({max_retries}) attained, ending...')
            return None
    logger.info(f'Collection complete. Time taken: {time.time()-start_time:.2f} seconds')
    return None

if __name__=='__main__':
    mediac()