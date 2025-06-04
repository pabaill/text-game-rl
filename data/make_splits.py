"""
Script for merging all CSV files in a directory into a single CSV file.
NOTE: this script is not used in the final implementation, which uses online training.
"""

import pandas as pd
import glob
import os

def merge_csv_files(directory, output_filename="merged_file.csv"):
    """Merges all CSV files in a directory into a single CSV file.

    Args:
        directory (str): The path to the directory containing the CSV files.
        output_filename (str, optional): The name of the output CSV file. 
            Defaults to "merged_file.csv".
    """
    all_filenames = glob.glob(os.path.join(directory, "*.csv"))
    all_df = []
    for f in all_filenames:
        df = pd.read_csv(f)
        df["game_name"] = f
        all_df.append(df)
    
    merged_df = pd.concat(all_df, ignore_index=True)
    merged_df.to_csv(output_filename, index=False)

if __name__ == '__main__':
    for directory in ['train', 'eval', 'test']:
        merge_csv_files(directory, output_filename=f"{directory}.csv")