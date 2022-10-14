import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import click
import pandas as pd
import numpy as np

@click.command()
@click.argument('group_num')
def main(group_num):
    ABM_ready_filename = f'/home/hubert/DPhil_Studies/2021-04_Study_A_Diffusion/data/06_reporting/ABM_processed_df_group_{group_num}.obj'
    with open(ABM_ready_filename, 'rb') as f:
        df = pickle.load(f)
    print('finished loading in df')
    fig = plt.figure(figsize = (16,9))

    sns_df = df.groupby(['created_at','ht']).count()
    sns_df['tweet_id'] = np.log(sns_df['tweet_id'])

    # Draw line plot of size and total_bill with parameters and hue "day"
    sns.lineplot(
        x = "created_at", y = "tweet_id", data = sns_df, hue = "ht",
                style = "ht", palette = "hot", dashes = False,  legend="brief",)
 
    plt.title("Interactions per day per hashtag", fontsize = 20)
    fontsize=20
    plt.xlabel("Date", fontsize = fontsize)
    plt.ylabel("Interactions (log scale)", fontsize = fontsize)
    plot_save_path = f'/home/hubert/DPhil_Studies/2021-04_Study_A_Diffusion/results/0{group_num}_group/'
    plt.savefig(f'{plot_save_path}overall_interactions_{group_num}.png', transparent=True, bbox_inches='tight', dpi=300)
    plt.show();

if __name__=='__main__':
    main()