import numpy as np
import pandas as pd
import glob
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

all_sim_path = sorted(glob.glob('/home/hubert/DPhil_Studies/2021-04_Study_A_Diffusion/results/*/abm_sim_percent.obj', recursive=True))
print(all_sim_path)
all_sim = []
if len(all_sim_path) == 3:
    for i in all_sim_path:
        with open(i, 'rb') as f:
            x = pickle.load(f)
            all_sim.append(x)
    ref_diff = [i[0] for i in all_sim]
    all_sim = [i[1] for i in all_sim]
    # all_sim_df = pd.DataFrame.from_dict({
        # str(i): e[1] for i,e in enumerate(all_sim) 
    # }, orient='columns')
    all_sim_df = pd.DataFrame.from_dict({
        'Period':list(
            np.concatenate((
                np.ones(len(all_sim[0]), dtype=np.int32),
                np.ones(len(all_sim[1]), dtype=np.int32)+1,
                np.ones(len(all_sim[2]), dtype=np.int32)+2
            ))
        ),
        'Simulated Percent': list(np.concatenate(all_sim))},
        orient='columns'
    )
    all_sim_df.Period = all_sim_df.Period.astype(int)
    fig = plt.figure(figsize=(15,8))
    ax = sns.boxplot(
        y='Simulated Percent',
        x='Period',
        data=all_sim_df
    )
    ax.hlines(ref_diff[0], -0.3, 0.3, color='red')
    # ax2 = sns.boxplot(
    #     y='Simulated Percent',
    #     hue='Period',
    #     data=all_sim_df[all_sim_df['Period']==3.0],
    #     ax=axs[1]
    # )
    ax.hlines(ref_diff[1],  0.7, 1.3, color='red')
    ax.hlines(ref_diff[2],  1.7, 2.3, color='red')
    plt.savefig('/home/hubert/DPhil_Studies/2021-04_Study_A_Diffusion/results/abm_ref_sim_grouped_boxplot.png',bbox_inches='tight')