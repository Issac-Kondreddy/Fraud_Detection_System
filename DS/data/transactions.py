import pandas as pd
import numpy as np

# Load the preprocessed dataset (replace with your actual file path)
input_file = '/Users/issackondreddy/Desktop/Projects/Fraud Detection System/DS/data/processed_creditcard.csv'

# Load the dataset into a DataFrame
df = pd.read_csv(input_file)

# Shuffle the entire dataset to randomize it
df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Sample 300 random rows from the shuffled dataset
df_sample = df_shuffled.sample(n=50, random_state=42)

# Remove the 'Class' column from the sample (since you want to predict it)
df_sample_no_class = df_sample.drop(columns=['Class'])

# Save the sample to a new CSV file
output_file = 'random_sample50_without_class.csv'
df_sample_no_class.to_csv(output_file, index=False)

print(f"Saved a random sample of 50 transactions without the 'Class' column to '{output_file}'")
