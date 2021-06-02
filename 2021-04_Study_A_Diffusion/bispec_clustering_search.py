import subprocess
import tqdm

params = {
    'range': (10,200),
    'interval': 10,
    'min_user': 20
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