import os
import pandas as pd
import tqdm

# Step 1: Read .txt files and store in a dictionary
folder_path = '/data/inet-large-scale-twitter-diffusion/ball4321/data_c/TikTok/videos_transcribed/'  # Update with the path to your .txt files
text_data = {}
for filename in tqdm.tqdm(os.listdir(folder_path)):
    if filename.endswith('.txt'):
        file_path = os.path.join(folder_path, filename)
        with open(file_path, 'r', encoding='utf-8') as file:
            text_data[filename[:-4]] = file.read()  # Remove '.txt' from filename
print('data read in')

# Step 2: Read CSV file, excluding the 'position' column
csv_path = '/data/inet-large-scale-twitter-diffusion/ball4321/data_c/YT/videolist_search7071_2023_12_12-14_24_28.csv'  # Update with the path to your CSV file
metadata_df = pd.read_csv(csv_path).drop(columns=['position'])
print('csv read in')

# Step 3: Merge text data with metadata
# Convert text data to DataFrame
text_df = pd.DataFrame.from_dict(text_data, orient='index', columns=['text'])
text_df.reset_index(inplace=True)
text_df.rename(columns={'index': 'videoId'}, inplace=True)
print('merge complete')

# Merge with metadata
final_dataset = pd.merge(metadata_df, text_df, on='videoId', how='inner')

# Step 4: Your dataset is now ready for BERTopic or other NLP processing
#print(final_dataset.head())

final_dataset.to_parquet('/data/inet-large-scale-twitter-diffusion/ball4321/data_c/YT/audio_transcribed.parquet')
print('Save complete. Ending.')

# You can now use final_dataset for your NLP tasks

