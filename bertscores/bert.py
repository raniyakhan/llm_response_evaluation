import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import torch
from transformers import AutoModel, AutoTokenizer

# Load the model and tokenizer for BERT
model_name = 'bert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

def get_embedding(text):
    # Ensure the text is a non-empty string
    if not text or type(text) != str:
        return None
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).detach().numpy()

def calculate_bert_similarity(text1, text2):
    emb1 = get_embedding(text1)
    emb2 = get_embedding(text2)
    if emb1 is None or emb2 is None:
        return 0  # Return a similarity score of 0 if either text is invalid
    return cosine_similarity(emb1, emb2)[0][0]

# Load your dataset
input_csv = "gpt2_smallresponses.csv"  # Ensure this is the correct file path
df = pd.read_csv(input_csv)

# New DataFrame to store results
new_df = df.copy()

# Calculating new ground truth scores
for index, row in df.iterrows():
    llm_answer = str(row['LLM Answer'])
    ground_truth = str(row['Ground Truth Answer'])
    other_answers = str(row['Other Correct Answers']).split(';')
    
    # Calculate similarity with the ground truth answer
    gt_similarity = calculate_bert_similarity(llm_answer, ground_truth)
    
    # Calculate similarities with other correct answers
    similarities = [calculate_bert_similarity(llm_answer, ans) for ans in other_answers if ans.strip()]
    
    # Compute a weighted score
    weighted_score = 0.6 * gt_similarity + 0.4 * max(similarities) if similarities else gt_similarity
    
    # Store the weighted score in the new DataFrame, formatted to 7 decimal places
    new_df.at[index, 'Ground Truth Score'] = f"{weighted_score:.7f}"

# Save the new DataFrame to a CSV
output_csv = "gpt2_with_scores.csv"  # Specify the output file name
new_df.to_csv(output_csv, index=False)

print("Updated dataset with new ground truth scores has been saved.")
