import pandas as pd
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util

# Load Sentence Transformer model for coherence and relevance checks
sbert_model = SentenceTransformer('all-mpnet-base-v2')

# Load a pre-trained model for entailment checking (factual accuracy)
nli_model = pipeline('text-classification', model='roberta-large-mnli')

def check_factual_accuracy(hypothesis, reference):
    """ Check if the hypothesis is entailed by the reference. """
    result = nli_model(f"{reference} [SEP] {hypothesis}")
    if result[0]['label'] == 'ENTAILMENT' and result[0]['score'] > 0.7:
        return True
    return False

def calculate_similarity(text1, text2):
    """Calculate the semantic similarity between two texts using SBERT."""
    if pd.isna(text1) or pd.isna(text2):  # Check for NaN values
        return 0.0
    embeddings1 = sbert_model.encode(str(text1), convert_to_tensor=True)
    embeddings2 = sbert_model.encode(str(text2), convert_to_tensor=True)
    cosine_scores = util.pytorch_cos_sim(embeddings1, embeddings2)
    return cosine_scores.item()

def evaluate_responses(input_file_path, output_file_path):
    df = pd.read_csv(input_file_path)
    results = []

    for index, row in df.iterrows():
        llm_answer = row['LLM Answer']
        ground_truth = row['Ground Truth Answer']

        # Ensure data types are correct and handle NaN values
        llm_answer = '' if pd.isna(llm_answer) else str(llm_answer)
        ground_truth = '' if pd.isna(ground_truth) else str(ground_truth)

        # Factual Accuracy
        accuracy = check_factual_accuracy(llm_answer, ground_truth) if llm_answer and ground_truth else 0

        # Contextual Relevance
        relevance = calculate_similarity(llm_answer, ground_truth)

        # Coherence (Assumed as self-coherence in the response)
        coherence = calculate_similarity(llm_answer, llm_answer)

        # Combine the scores (adjust weights as necessary)
        score = 0.4 * accuracy + 0.3 * relevance + 0.3 * coherence
        results.append({'index': index, 'score': score})

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_file_path, index=False)
    print(f"Updated dataset with evaluation scores has been saved to {output_file_path}")

# User inputs for file paths
input_file_path = "gpt2_largeresponses.csv"
output_file_path = "sbert_large_test.csv"

# Evaluate and save results
evaluate_responses(input_file_path, output_file_path)
