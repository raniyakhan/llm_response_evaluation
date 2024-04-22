import pandas as pd
import matplotlib.pyplot as plt

def read_data(file_path):
    """Reads the dataset and returns the 'Ground Truth Score' as a pandas Series."""
    df = pd.read_csv(file_path)
    return df['Ground Truth Score']

def plot_detailed_comparison(small_scores, medium_scores, large_scores):
    """Generates detailed plots comparing ground truth scores from three datasets."""
    plt.figure(figsize=(14, 6))

    # Histograms
    plt.subplot(1, 3, 1)
    plt.hist(small_scores, bins=20, alpha=0.7, label='GPT-2 Small')
    plt.hist(medium_scores, bins=20, alpha=0.7, label='GPT-2 Medium')
    plt.hist(large_scores, bins=20, alpha=0.7, label='GPT-2 Large')
    plt.title('Histogram of Scores')
    plt.xlabel('Scores')
    plt.ylabel('Frequency')
    plt.legend()

    # Boxplot
    plt.subplot(1, 3, 3)
    plt.boxplot([small_scores, medium_scores, large_scores], labels=['GPT-2 Small', 'GPT-2 Medium', 'GPT-2 Large'])
    plt.title('Boxplot of Scores')
    plt.ylabel('Scores')

    plt.tight_layout()
    plt.show()

# Call the updated plot function



# Paths to the datasets
small_data_path = 'gpt2small_with_scores.csv'  # Change to your file path
medium_data_path = 'gpt2medium_withscores.csv'  # Change to your file path
large_data_path = 'gpt2large_withscores.csv'  # Change to your file path

# Read the scores from CSV files
small_scores = read_data(small_data_path)
medium_scores = read_data(medium_data_path)
large_scores = read_data(large_data_path)

# Plot the comparison
# plot_comparison(small_scores, medium_scores, large_scores)
plot_detailed_comparison(small_scores, medium_scores, large_scores)