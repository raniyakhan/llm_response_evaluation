import pandas as pd
import matplotlib.pyplot as plt

# Load your datasets
df_small = pd.read_csv('gpt2large_withscores.csv')
df_medium = pd.read_csv('gpt2medium_withscores.csv')
df_large = pd.read_csv('gpt2small_with_scores.csv')

# Plotting
plt.figure(figsize=(15, 5))

# Histogram for GPT-2 Small
plt.subplot(1, 3, 1)
plt.hist(df_small['Ground Truth Score'], bins=20, alpha=0.7, label='Small')
plt.title('GPT-2 Small Ground Truth Scores')
plt.xlabel('Score')
plt.ylabel('Frequency')

# Histogram for GPT-2 Medium
plt.subplot(1, 3, 2)
plt.hist(df_medium['Ground Truth Score'], bins=20, alpha=0.7, color='orange', label='Medium')
plt.title('GPT-2 Medium Ground Truth Scores')
plt.xlabel('Score')
plt.ylabel('Frequency')

# Histogram for GPT-2 Large
plt.subplot(1, 3, 3)
plt.hist(df_large['Ground Truth Score'], bins=20, alpha=0.7, color='green', label='Large')
plt.title('GPT-2 Large Ground Truth Scores')
plt.xlabel('Score')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()
