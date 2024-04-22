import pandas as pd
import numpy as np

def calculate_statistics(file_path):
    try:
        # Load the dataset
        df = pd.read_csv(file_path)

        # Ensure that 'Ground Truth Score' column exists
        if 'score' not in df.columns:
            print(f"Missing 'Ground Truth Score' in {file_path}")
            return

        # Calculate statistics
        mean_score = np.mean(df['score'])
        std_deviation = np.std(df['score'])
        min_score = np.min(df['score'])
        max_score = np.max(df['score'])
        median_score = np.median(df['score'])
        quantile_25 = np.percentile(df['score'], 25)
        quantile_75 = np.percentile(df['score'], 75)

        # Print the results
        print(f"Statistics for {file_path}:")
        print(f"Mean: {mean_score:.4f}")
        print(f"Standard Deviation: {std_deviation:.4f}")
        print(f"Min: {min_score:.4f}")
        print(f"Max: {max_score:.4f}")
        print(f"Median: {median_score:.4f}")
        print(f"25th Percentile: {quantile_25:.4f}")
        print(f"75th Percentile: {quantile_75:.4f}")
        print("-" * 40)

    except Exception as e:
        print(f"Failed to process {file_path}: {str(e)}")

# List of file paths
files = [
    'small_sbert_roberta_scores.csv',
    'medium_sbert_roberta_scores.csv',
    'large_sbert_roberta_scores.csv'
]

# Calculate statistics for each file
for file_path in files:
    calculate_statistics(file_path)
