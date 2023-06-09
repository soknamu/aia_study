import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import os

# Set directory where CSV files are located
csv_dir = './_data/spark/TEST_INPUT/'
csv_save = './_data/spark/TEST_INPUT/'
# Create empty list to hold dataframes
df_list = []

# Loop through all CSV files in directory
for file in os.listdir(csv_dir):
    if file.endswith('.csv'):
        # Read CSV file into dataframe
        df = pd.read_csv(os.path.join(csv_dir, file))
        
        # Do any necessary data cleaning or manipulation here
        # ...
        
        # Append dataframe to list
        df_list.append(df)

# Concatenate all dataframes in list into one dataframe
combined_df = pd.concat(df_list)

# Save combined dataframe to CSV file
combined_df.to_csv(csv_save + 'train_all.csv', index=False)
