import pandas as pd
import os

# Specify the folder containing the CSV files
folder_path = 'output'

# List all CSV files in the directory
csv_files = [file for file in os.listdir(folder_path) if file.endswith('.csv')]

# Initialize an empty list to hold dataframes
dataframes = []

# Read each CSV file and append to the list
for file in csv_files:
    file_path = os.path.join(folder_path, file)
    df = pd.read_csv(file_path)
    dataframes.append(df)

# Concatenate all dataframes into one
combined_df = pd.concat(dataframes, ignore_index=True)

# Save the combined dataframe to a new CSV file
combined_output_path = os.path.join(folder_path, 'csvFinal.csv')
combined_df.to_csv(combined_output_path, index=False)

print(f"All CSV files have been merged and saved to {combined_output_path}")
