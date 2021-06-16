import datetime
import subprocess
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat

import tqdm

params = {
    'range': (10,200),
    'interval': 1,
    'min_user': 10
}

for cluster in tqdm.tqdm(range(params['range'][0],params['range'][1]+1,params['interval'])):
    subprocess.run(
        [
            'Rscript',
            'bispectral_clustering.R',
            '/Users/hubert/Nextcloud/DPhil/DPhil_Studies/2021-04_Study_A_Diffusion/collection_results_2021_05_04_16_22/bispec_ready_counts.csv',
            '/Users/hubert/Nextcloud/DPhil/DPhil_Studies/2021-04_Study_A_Diffusion/collection_results_2021_05_04_16_22/bsc/',
            '--min_user',
            str(params['min_user']),
            '--ncluster',
            str(cluster)
        ],
        cwd='/Users/hubert/Nextcloud/DPhil/DPhil_Studies/2021-04_Study_A_Diffusion'
    )

# for cluster in tqdm.tqdm(range(params['range'][0],params['range'][1]+1,params['interval'])):
#     p=subprocess.Popen(
#         [
#             'Rscript',
#             'bispectral_clustering.R',
#             '/Users/hubert/Nextcloud/DPhil/DPhil_Studies/2021-04_Study_A_Diffusion/collection_results_2021_05_04_16_22/bispec_ready_counts.csv',
#             '/Users/hubert/Nextcloud/DPhil/DPhil_Studies/2021-04_Study_A_Diffusion/collection_results_2021_05_04_16_22/bsc/',
#             '--min_user',
#             str(params['min_user']),
#             '--ncluster',
#             str(cluster)
#         ],
#         cwd='/Users/hubert/Nextcloud/DPhil/DPhil_Studies/2021-04_Study_A_Diffusion'
#     )
# p.wait()

# def process_file(cluster_value, min_user):

#     start_time = datetime.datetime.now()
#     print('start cluster value {} at: {}'.format(cluster_value, start_time))

#     subprocess.run(
#         [
#             'Rscript',
#             'bispectral_clustering.R',
#             '/Users/hubert/Nextcloud/DPhil/DPhil_Studies/2021-04_Study_A_Diffusion/collection_results_2021_05_04_16_22/bispec_ready_counts.csv',
#             '/Users/hubert/Nextcloud/DPhil/DPhil_Studies/2021-04_Study_A_Diffusion/collection_results_2021_05_04_16_22/bsc/',
#             '--min_user',
#             str(min_user),
#             '--ncluster',
#             str(cluster_value)
#         ],
#         cwd='/Users/hubert/Nextcloud/DPhil/DPhil_Studies/2021-04_Study_A_Diffusion'
#     )

#     print('end cluster value {} at: {}'.format(cluster_value, datetime.datetime.now()-start_time))

# with ProcessPoolExecutor(max_workers=7) as executor:

#     _ = executor.map(
#         process_file,
#         range(params['range'][0],params['range'][1]+1,params['interval']),
#         repeat(params['min_user'])
#     )