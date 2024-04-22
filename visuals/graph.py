import pandas as pd
import matplotlib.pyplot as plt

def load_scores(file_path):
    """ Load scores from a CSV file into a Pandas DataFrame. """
    return pd.read_csv(file_path)['score']

def plot_histograms(small_scores, medium_scores, large_scores):
    """ Plot histograms for scores from small, medium, and large models. """
    plt.figure(figsize=(18, 5))

    # Histogram for small model scores
    plt.subplot(1, 3, 1)
    plt.hist(small_scores, bins=20, color='blue', alpha=0.7)
    plt.title('Histogram of Scores for Small Model')
    plt.xlabel('Scores')
    plt.ylabel('Frequency')

    # Histogram for medium model scores
    plt.subplot(1, 3, 2)
    plt.hist(medium_scores, bins=20, color='green', alpha=0.7)
    plt.title('Histogram of Scores for Medium Model')
    plt.xlabel('Scores')
    plt.ylabel('Frequency')

    # Histogram for large model scores
    plt.subplot(1, 3, 3)
    plt.hist(large_scores, bins=20, color='red', alpha=0.7)
    plt.title('Histogram of Scores for Large Model')
    plt.xlabel('Scores')
    plt.ylabel('Frequency')

    plt.tight_layout()
    plt.show()

def plot_boxplot(small_scores, medium_scores, large_scores):
    """ Plot a box plot to compare scores across small, medium, and large models. """
    data = [small_scores, medium_scores, large_scores]
    plt.figure(figsize=(8, 6))
    plt.boxplot(data, labels=['Small', 'Medium', 'Large'], notch=True, patch_artist=True)
    plt.title('Comparison of Ground Truth Scores Across Model Sizes')
    plt.ylabel('Scores')
    plt.grid(True)
    plt.show()

# Paths to the datasets
small_data_path = 'sbert_short_test.csv'  # Update path as needed
medium_data_path = 'sbert_medium_test.csv'  # Update path as needed
large_data_path = 'sbert_large_test.csv'  # Update path as needed

# Load the scores
small_scores = load_scores(small_data_path)
medium_scores = load_scores(medium_data_path)
large_scores = load_scores(large_data_path)

# Generate histograms
plot_histograms(small_scores, medium_scores, large_scores)

# Generate box plot
plot_boxplot(small_scores, medium_scores, large_scores)
