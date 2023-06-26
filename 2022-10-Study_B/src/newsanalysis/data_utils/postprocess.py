'''Functions to handle post processing
'''

import click
from loguru import logger
import pandas as pd
from collections import Counter
from itertools import repeat
from datasets import Dataset
from pathlib import Path
import os
import pickle
import numpy as np
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
from tensorflow_probability import sts
import tensorflow as tf
from dateutil.relativedelta import relativedelta
from datetime import datetime, timedelta
from causalimpact import CausalImpact
from causalimpact.misc import standardize


import pickle
from pathlib import Path
import rapidfuzz
from concurrent.futures import ProcessPoolExecutor

@click.group(help='Commands for postprocessing')
@click.pass_context
def postprocess(ctx):
    pass

@postprocess.command()
@click.argument('subsfile')
@click.option('--outfolder', '-o', required=True)
@click.option('--original_data', '-d', required=False)
@click.option('--relfile', '-r', required=False)
@click.option('--mcsourceinfo', '-m', required=False)
@click.option('--substhresh', '-st', type=float, default=None)
@click.option('--relthresh', '-rt', type=float, default=None)
def consolidatesubs(subsfile, outfolder, original_data, mcsourceinfo, relfile, substhresh, relthresh):
    '''Conslidate substance annotations from split back into stories'''

    # Load in substance annotation file
    with open(subsfile, 'rb') as f:
        original_dict = pickle.load(f)

    # Load in relevance annotation file
    with open(relfile, 'rb') as f:
        rel_dict = pickle.load(f)

    for k,v in original_dict.items():
        has_logits = isinstance(v, tuple)
        break
    logger.info(f'Logits is {has_logits}')

    # get the substance annotations back into stories and not parts
    new_dict = {}
    if has_logits:
        for k, predict_tuple in original_dict.items():
            id, _ = k.split('_')

            # check relevance threshold (N.B. don't need to check if predict is 1 because otherwise it wouldn't be in subs dict):
            if rel_dict[k][0] < relthresh:
                continue

            # check subs threshold
            logit, v  = predict_tuple
            if logit < substhresh:
                continue

            if id not in new_dict:
                new_dict[id] = {v: 1}
            else:
                if v in new_dict[id]:
                    new_dict[id][v] += 1
                else:
                    new_dict[id][v] = 1

    else:
        for k, v in original_dict.items():
            id, _ = k.split('_')
            if id not in new_dict:
                new_dict[id] = {v: 1}
            else:
                if v in new_dict[id]:
                    new_dict[id][v] += 1
                else:
                    new_dict[id][v] = 1

    # Convert the new dictionary to a DataFrame
    df = pd.DataFrame.from_dict({k: Counter(v) for k, v in new_dict.items()}, orient='index')
    df.fillna(0, inplace=True)
    df = df.astype(int)

    df.index.names=['processed_stories_id']

    if original_data:
        # supply media id information
        original_raw = Dataset.load_from_disk(original_data)
        original_df = original_raw.to_pandas()
        original_df = original_df.set_index('processed_stories_id')
        original_df.index = original_df.index.map(str)

        df = df.join(original_df[['media_id', 'language', 'publish_date']])

    # supply media locale information:
    if mcsourceinfo:
        with open(mcsourceinfo, 'rb') as f:
            mc = pickle.load(f)
        df['country'] = df['media_id'].replace(mc)

    df.to_csv(f'{outfolder}/{Path(subsfile).stem}{"_" if any([original_data, mcsourceinfo]) else ""}{"d" if original_data else ""}{"m" if mcsourceinfo else ""}{f"_r{relthresh}" if relthresh else ""}{f"_s{substhresh}" if substhresh else ""}.csv')

def process_one_token(number, token, names_ref):
    if number % 10000 == 0:
        logger.info(f'{number}')
    return (token, rapidfuzz.process.extractOne(token.upper(), names_ref))

@postprocess.command()
@click.argument('ner_file')
@click.argument('outfile')
@click.option('--dataset', '-d', required=True)
@click.option('--names', '-n', required=True)
@click.option('--surnames', '-s', required=True)
@click.option('--up_to', '-u', type=int)
def consolidatener(ner_file, outfile, dataset, names, surnames, up_to):
    with open(ner_file, 'rb') as f:
        ner_annot = pickle.load(f)

    logger.info('NER loaded in')

    # get languages of articles
    og_dataset = Dataset.load_from_disk(dataset)
    og_dataset = og_dataset.to_pandas()
    og_dataset = og_dataset.set_index('processed_stories_id')
    logger.info("Dataset loaded in")

    ner_filtered = {}
    for k, v in ner_annot.items():
        lang = og_dataset.loc[k]['language']
        if lang in ['zh', 'ja', 'ko', 'bn', 'hi', 'ta', 'gu', 'kn', 'te', 'ml', 'th', 'my', 'mr', 'vi', 'ne', 'lo', 'ur', 'si', 'or', 'ceb']:
            ner_filtered[k] = v
        else:

            ner_filtered[k] = [i for i in v if 1 < len(i.split())<5 ]
    logger.info('Dataset filtered')

    # import name list to compare against
    name_database = pd.read_csv(names,header=None, names = ['name', 'frequency', 'males', 'females'])
    surname_database = pd.read_csv(surnames, header=None, names = ['name', 'frequency'])

    #combine
    allname_database = pd.concat((name_database['name'], surname_database['name'])).to_list()
    logger.info('Names loaded in')

    unique_tokens = set()
    for k, v in ner_filtered.items():
        lang = og_dataset.loc[k]['language']
        if lang in ['zh', 'ja', 'ko', 'bn', 'hi', 'ta', 'gu', 'kn', 'te', 'ml', 'th', 'my', 'mr', 'vi', 'ne', 'lo', 'ur', 'si', 'or', 'ceb']:
            pass
        else:
            for i in v:
                unique_tokens.update(i.split())
    unique_tokens = list(unique_tokens)
    if up_to:
        logger.info(f'DEBUGGING UP TO {up_to}')
        unique_tokens = unique_tokens[:up_to]
    logger.info('Unique tokens extracted')
    logger.info(f'Beginning ProcessPoolExecutor')
    processpoolout = []
    with ProcessPoolExecutor(max_workers=None) as executor:
        # for number, token in zip(range(len(unique_tokens)), unique_tokens):
        #     if number % 10000 == 0:
        #         logger.info(f'{number}')
        #     processpoolout.append(executor.submit(process_one_token, token, allname_database))
        processpoolout = executor.map(process_one_token, range(len(unique_tokens)), unique_tokens, repeat(allname_database))
    logger.info('ProcessPool done')
    # processpoolout = [i.result() for i in processpoolout]

    final_dict = {}
    for tok, corrected in processpoolout:
        final_dict[tok] = corrected[0]

    with open(outfile, 'wb') as f:
        pickle.dump(final_dict, f)
    logger.info(f'Saved to {outfile}')


def extract_trend(complete_df, country, min_count = 500, resample_time = 'W'):

    logger.info(f'Processing {country}')

    reg_df = complete_df.copy()
    reg_df = reg_df.set_index('publish_date')
    reg_df = reg_df.groupby('country').resample(resample_time).agg(
        {'processed_stories_id':'count', 'myth_total':'sum'}
    ).reset_index()
    reg_df.rename({'processed_stories_id':'count'}, axis=1, inplace=True)

    if country in ['XXX', 'ZZZ']:
        return None

    obs_data = reg_df[reg_df['country']==country]['myth_total']
    coverage = reg_df[reg_df['country']==country]['count']
    if coverage.sum() < min_count:
        return None

    impact = tfp.sts.LocalLinearTrend(
        observed_time_series=obs_data, name = f'#MeToo Trend - {country}'
    )
    coverage_effect = tfp.sts.LinearRegression(
        design_matrix=tf.reshape(coverage - np.mean(coverage), (-1, 1)), name='Coverage Effect'
    )
    residual_level = tfp.sts.Autoregressive(
        order=1,
        observed_time_series=obs_data,
        name='Residual'
    )
    model = tfp.sts.Sum([impact, coverage_effect, residual_level],
                        observed_time_series=obs_data)

    # Build the variational surrogate posteriors `qs`.
    variational_posteriors = tfp.sts.build_factored_surrogate_posterior(
        model=model)

    # Allow external control of optimization to reduce test runtimes.
    num_variational_steps = 200 # @param { isTemplate: true}
    num_variational_steps = int(num_variational_steps)

    # Build and optimize the variational loss function.
    elbo_loss_curve = tfp.vi.fit_surrogate_posterior(
        target_log_prob_fn=model.joint_distribution(
            observed_time_series=obs_data).log_prob,
        surrogate_posterior=variational_posteriors,
        optimizer=tf.optimizers.Adam(learning_rate=0.1),
        num_steps=num_variational_steps,
        jit_compile=True)

    # plt.plot(elbo_loss_curve)
    # plt.show()

    # Draw samples from the variational posterior.
    param_samples = variational_posteriors.sample(50)

    component_dists = sts.decompose_by_component(
        model,
        observed_time_series=obs_data,
        parameter_samples=param_samples
    )

    component_means_, component_stddevs_ = (
    {k.name: c.mean() for k, c in component_dists.items()},
    {k.name: c.stddev() for k, c in component_dists.items()})

    param_names = [i.name for i in model.parameters]

    logger.info(f'DONE Processing {country}')

    return country, param_names, param_samples, component_means_, component_stddevs_

@postprocess.command()
@click.argument('original_df_file')
@click.argument('outdir')
@click.option('--resample', '-r', help='timeframe to resample. Can be W, M, or Y', default='W')
@click.option('--min_count', '-m', type=int, default=500)
def structuralts(original_df_file, outdir, resample, min_count):


    tf.get_logger().setLevel('INFO')

    complete_df = pd.read_pickle(original_df_file)
    logger.info(f'Data loaded in from {original_df_file}')

    countries = list(complete_df['country'].unique())
    logger.info(f'Unique contries collected')

    results = {}
    for i, country in enumerate(countries):
        output = extract_trend(
            complete_df,
            country
        )
        results[output[0]] =  {
            'param_names': i[1],
            'param_samples': i[2],
            'component_means': i[3],
            'component_stddevs' : i[4]
        }

    outfile = Path(outdir) / f'sts_results_{min_count}_{resample}.pkl'
    with open(outfile, 'wb') as f:
        pickle.dump(results, f)

def last_day_of_month(any_day):
    # The day 28 exists in every month. 4 days later, it's always next month
    next_month = any_day.replace(day=28) + timedelta(days=4)
    # subtracting the number of the current day brings us back one month
    return next_month - timedelta(days=next_month.day)

def cimpact(complete_df, country, peaks, min_count = 500, resample_time = 'W'):
    # N.B.  The linear regression must contain the union of pre and post data as required by TensorFlow Probability.

    if country in ['XXX', 'ZZZ']:
        logger.info(f'DONE {country}')
        return None

    resample_time='M'
    logger.info(f'Processing {country}')

    reg_df = complete_df.copy()
    reg_df = reg_df.set_index('publish_date')
    reg_df = reg_df.groupby('country').resample(resample_time).agg(
        {
            'processed_stories_id':'count',
            'myth_total':'sum',
            'e_v2x_gender_5C': 'first', # Women political empowerment
        }
    ).reset_index().set_index('publish_date').sort_index()

    data = reg_df[reg_df['country']==country][['myth_total', 'processed_stories_id']]
    if data['processed_stories_id'].sum() < min_count:
        logger.info(f'DONE {country}')
        return None

    # custom peak means str format
    if isinstance(peaks, str):
        peak = datetime.strptime(peaks, "%Y-%m-%d")
    else:
        try:
            peak = datetime.strptime(peaks.loc[country, 'peak_date'], "%Y-%m-%d")
        except KeyError:
            logger.warning(f'{country} not found in peaks')
            return None

    if resample_time == 'D':
        # pre_period=['2014-10-17', '2017-10-17']
        # post_period=['2017-10-18', '2020-10-17']
        pre_period = ['2014-10-17', peak.strftime("%Y-%m-%d")]
        post_period = [
            (peak + relativedelta(days=+1)).strftime("%Y-%m-%d"),
            (peak + relativedelta(years=+1)).strftime("%Y-%m-%d")
        ]
    elif resample_time == 'M':
        start_date = '2014-10-31'
        if '2014-10-31' not in data.index:
            start_date = (data.index[0]).strftime('%Y-%m-%d')
        pre_period = [start_date, peak.strftime("%Y-%m-%d")]
        post_period = [
            last_day_of_month(peak + relativedelta(months=1)).strftime("%Y-%m-%d"),
            last_day_of_month(peak + relativedelta(months=13)).strftime("%Y-%m-%d")
        ]
        if post_period[1] not in data.index:
            post_period[1] = data.index[-1].strftime("%Y-%m-%d")
    elif resample_time == 'W':
        pre_period = ['2014-10-17', peak.strftime("%Y-%m-%d")]
        post_period = [
            (peak + relativedelta(week=+1)).strftime("%Y-%m-%d"),
            (peak + relativedelta(years=+1)).strftime("%Y-%m-%d")
        ]

    logger.info(pre_period)
    logger.info(post_period)
    if post_period[0] == post_period[1]:
        logger.info(f'No post period')
        return None

    normed_data, mu_sig = standardize(data)
    # normed_data = data

    obs_data = tfp.sts.regularize_series(normed_data['myth_total'].loc[:pre_period[1]].astype(np.float32))

    design_matrix = tf.reshape(pd.concat([
        normed_data['processed_stories_id'].loc[pre_period[0]: pre_period[1]],
        normed_data['processed_stories_id'].loc[post_period[0]: post_period[1]]
    ]).astype(np.float32), (-1,1))


    linear_level = tfp.sts.LocalLinearTrend(observed_time_series=obs_data, name='local')
    coverage_effect = tfp.sts.LinearRegression(design_matrix=design_matrix, name='coverage_effect')
    month_season = tfp.sts.Seasonal(num_seasons=4, num_steps_per_season=1, observed_time_series=obs_data, name='Month')
    year_season = tfp.sts.Seasonal(num_seasons=52, observed_time_series=obs_data, name='Year')
    residual_level = tfp.sts.Autoregressive(
        order=1,
        observed_time_series=obs_data, name='residual')


    model = tfp.sts.Sum([linear_level, coverage_effect, residual_level], observed_time_series=obs_data)

    ci = CausalImpact(
        normed_data,
        pre_period,
        post_period,
        model=model,
        model_args = {'fit_method': 'hmc'}
    )

    logger.info(f'DONE {country}')
    return {
        'p_value': ci.p_value,
        'summary': ci.summary(),
        'summary_data': ci.summary_data,
        'inferences': ci.inferences,
        'pre_data': ci.pre_data,
        'post_data': ci.post_data,
        '_mask': ci._mask
    }

@postprocess.command()
@click.argument('original_df_file')
@click.argument('outdir')
@click.option('--peaks', '-p', default = None)
@click.option('--resample', '-r', help='timeframe to resample. Can be W, M, or Y', default='W')
@click.option('--min_count', '-m', type=int, default=500)
def ci(original_df_file, outdir, peaks, resample, min_count):

    tf.get_logger().setLevel('ERROR')

    complete_df = pd.read_pickle(original_df_file)
    logger.info(f'Data loaded in from {original_df_file}')

    custom_peak=False
    if peaks and os.path.isfile(peaks):
        peaks_df = pd.read_csv(peaks).set_index('country')
    elif peaks and isinstance(str):
        peaks_df = peaks
        custom_peak=True
        logger.info(f'Custom peak is {peaks_df}')

    countries = list(complete_df['country'].unique())
    logger.info(f'Unique contries collected')

    # logger.info(f'Begin ProcessPoolExecutor')
    # with ProcessPoolExecutor() as executor:
        # results = executor.map(
        #     cimpact,
        #     repeat(complete_df),
        #     countries,
        #     repeat(peaks_df),
        #     repeat(min_count),
        #     repeat(resample)
        # )

    results = {}
    for country in countries:
        results[country] = cimpact(
            complete_df,
            country,
            peaks_df,
            min_count=min_count,
            resample_time=resample
        )

    # logger.info(f'End ProcessPoolExecutor')

    # results = [i for i in results]

    outfile = Path(outdir) / f'cimpact_results_{min_count}_{resample}{f"_{peaks_df}" if custom_peak else "_peaks"}.pkl'
    with open(outfile, 'wb') as f:
        pickle.dump(results, f) 
