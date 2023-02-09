"""Collect MediaCloud data based on a query, desired outfile, start date and end date.

"""
import mediacloud.error
import mediacloud.api
import newsanalysis.data_utils.constants
import datetime
import os
import time
import click
import jsonlines
import logging

@click.command()
@click.option('--outfile', required=True)
@click.option('--query', required=True, type=str)
@click.option('--start', help='In format YYYY-MM-DD', required=True)
@click.option('--end', help='In format YYYY-MM-DD', required=True)
@click.option('--count', help='Just get count for the query specified. Does not log, just prints to console.', is_flag=True)
@click.option('--log_level',
    required=False,
    default='INFO',
    type=click.Choice(['NONE','CRITICAL','ERROR','WARNING','INFO','DEBUG']),
)
@click.option('--log_dir',
    required=False,
    help='Directory in which to save logs',
    default = os.getcwd()
)
@click.option('--log_handler_level',
    required=False,
    default='stream',
    type=click.Choice(['both', 'file', 'stream']),
    help='Whether to log to both a file and stream to console, or just one.'
)
def main(
    outfile,
    query,
    start,
    end,
    count,
    log_level,
    log_dir,
    log_handler_level
):

    # if just getting a count, no need to log.
    if not count:
        logging_dict = {
            'NONE': None,
            'CRITICAL': logging.CRITICAL,
            'ERROR': logging.ERROR,
            'WARNING': logging.WARNING,
            'INFO': logging.INFO,
            'DEBUG': logging.DEBUG
        }

        logging_level = logging_dict[log_level]

        if logging_level is not None:

            logging_fmt   = '[%(levelname)s] %(asctime)s - %(name)s - %(message)s'
            today_datetime = str(datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S'))
            if log_dir is not None:
                assert os.path.isdir(log_dir)
                logging_file  = os.path.join(log_dir, f'{os.path.split(outfile)[-1].split(".")[0]}_mediacloud_collect.log')

            if log_handler_level == 'both':
                handlers = [
                    logging.FileHandler(filename=logging_file,mode='w+'),
                    logging.StreamHandler()
                ]
            elif log_handler_level == 'file':
                handlers = [logging.FileHandler(filename=logging_file,mode='w+')]
            elif log_handler_level == 'stream':
                handlers = [logging.StreamHandler()]
            logging.basicConfig(
                handlers=handlers,
                format=logging_fmt,
                level=logging_level,
                datefmt='%m/%d/%Y %I:%M:%S %p'
            )
            logger = logging.getLogger(__name__)
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
    main()