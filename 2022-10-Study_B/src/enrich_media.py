import newspaper
import time
import click
import gzip
import os
import jsonlines
import datetime
import logging
from concurrent.futures import ThreadPoolExecutor

def collect_one_url(story):
    logger = logging.getLogger(__name__)
    try:
        article = newspaper.Article(story.get('url'))
        article.download()
        article.parse()
        return article.text
    except:
        logger.info(f'Failed to collect {story.get("processed_stories_id")}')
        return None


@click.command()
@click.option('--infile', required=True, help='input JSONL file of stories collected from MediaCloud.')
@click.option('--outfile', required=False, help='output file name. Can be left blank and will be [infile]_enriched.jsonl.gz2')
@click.option('--continue_from', required=False, help='Continue from this processed story id.', type=int)
@click.option('--multithread', is_flag=True, required=False)
@click.option('--workers', default=25, type=int, help='number of threads', required=False)
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
def main(infile, outfile, continue_from, multithread, workers, log_level, log_dir, log_handler_level):

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
        if log_dir is not None:
            assert os.path.isdir(log_dir)
            logging_file  = os.path.join(log_dir, f'{os.path.split(outfile)[-1].split(".")[0]}_enrich.log')

        if log_handler_level == 'both':
            handlers = [
                logging.FileHandler(filename=logging_file,mode='a'),
                logging.StreamHandler()
            ]
        elif log_handler_level == 'file':
            handlers = [logging.FileHandler(filename=logging_file,mode='a')]
        elif log_handler_level == 'stream':
            handlers = [logging.StreamHandler()]
        logging.basicConfig(
            handlers=handlers,
            format=logging_fmt,
            level=logging_level,
            datefmt='%m/%d/%Y %I:%M:%S %p'
        )
        logger = logging.getLogger(__name__)

    start_time = time.time()
    logger.info("#############################################################")
    logger.info("")
    logger.info(f'NEW RUN AT {datetime.datetime.now()}')
    logger.info(f'Outfile: {outfile}')
    logger.info(f'Infile: {infile}')

    # get length of the iterables
    total_infile = 0
    processed_stories = []
    enriched_stories = []
    query_already_written = False
    with jsonlines.open(infile, 'r') as reader:
        for story in reader:
            if 'query' in story:
                continue
            total_infile += 1
            processed_stories.append(story.get('processed_stories_id'))
    logger.info(f"Stories to enrich: {total_infile}")
    if os.path.isfile(outfile):
        logger.info('Outfile detected as existing. Reading in already enriched stories...')
        with jsonlines.open(outfile, 'r') as json_reader:
            try:
                for story in json_reader:
                    if 'query' in story:
                        query_already_written = True
                        continue
                    enriched_stories.append(story.get('processed_stories_id'))
            except:
                logger.warning(f'FAILED TO READ IN: {story}')
        assert set(enriched_stories).issubset(set(processed_stories))
        logger.info(f'To collect: {total_infile - len(enriched_stories)}')

    with jsonlines.open(infile, 'r') as reader:
        with jsonlines.open(outfile, 'a') as json_writer:
            logger.info('Collecting article text')
            count = 0
            success = 0
            already_enriched = 0
            multithread_stories_to_collect = []
            for story in reader.iter(skip_invalid=True, skip_empty=True):
                if 'query' in story:
                    if not query_already_written:
                        json_writer.write(story)
                    continue
                if story.get('processed_stories_id') in enriched_stories:
                    already_enriched += 1
                    logger.info(f'Story ID {story.get("processed_stories_id")} already enriched. Continuing...')
                    continue
                elif continue_from and story.get('processed_stories_id',0) < continue_from:
                    logger.info(f'Story ID {story.get("processed_stories_id")} lower than continue from id {continue_from}')
                    continue
                elif story.get('url', 'nourl').lower().strip('/').endswith('json'):
                    logger.warning(f'Story ID {story.get("processed_stories_id")} has url {story.get("url")} ending with .json -> do not collect for memory reasons')
                    continue


                if multithread:
                    if len(multithread_stories_to_collect)<workers:
                        multithread_stories_to_collect.append(story)
                        count+=1
                    else:
                        # logger.info(f'Processing {count} of {total_infile - len(enriched_stories)} = {count*100/(total_infile - len(enriched_stories)):.2f}%')
                        with ThreadPoolExecutor(max_workers=workers) as executor:
                            results = executor.map(collect_one_url, multithread_stories_to_collect)

                        for ind, story_text in enumerate(results):
                            if story_text is not None:
                                multithread_stories_to_collect[ind]['text'] = story_text
                                json_writer.write(multithread_stories_to_collect[ind])
                                success += 1
                        if count%100==0:
                            logger.info(f'PROGRESS: Processed {count}, successfully {success}, out of {total_infile-len(enriched_stories)}. Scan progress = {100*count/(total_infile-len(enriched_stories)):.2f}%, Scan success rate = {100*success/count:.2f}%')
                        multithread_stories_to_collect = [story]

                else:
                    try:
                        logger.debug(f"Collecting {story.get('processed_stories_id')}")
                        article = newspaper.Article(story.get('url'))
                        article.download()
                        article.parse()
                        story['text'] = article.text
                        json_writer.write(story)
                        success += 1
                        if count%100==0:
                            logger.info(f'PROGRESS: Processed {count}, successfully {success}, out of {total_infile-len(enriched_stories)}. Scan progress = {100*count/(total_infile-len(enriched_stories)):.2f}%, Scan success rate = {100*success/count:.2f}%')
                    except Exception:
                        logger.info(f'Failed to collect {story.get("processed_stories_id")}')
                        continue

            # wrap up remaining stories at the end
            if multithread:
                with ThreadPoolExecutor(max_workers=workers) as executor:
                    results = executor.map(collect_one_url, multithread_stories_to_collect)

                for ind, story_text in enumerate(results):
                    if story_text is not None:
                        multithread_stories_to_collect[ind]['text'] = story_text
                        json_writer.write(multithread_stories_to_collect[ind])
                        success += 1


    logger.info(f'Total time taken: {time.time()-start_time:.2f}s for {count} scanned out of {total_infile-len(enriched_stories)}. Scanned {100*count/total_infile:.2f}%, Scan success rate = {100*success/count:.2f}, Overall collect success rate {100*success/(total_infile-len(enriched_stories)):.2f}%')
    return None

if __name__ == '__main__':
    main()