import csv
from transformers import pipeline
from tqdm import tqdm

# Initialize the text generation model pipeline
model_name = 'gpt2-medium'  # SWITCH MODEL HERE
generator = pipeline('text-generation', model=model_name, framework="pt")  

def generate_answer(question):
    # Instruct the model to provide a one-sentence answer
    prompt = f"Answer the following question in one sentence: {question}"
    responses = generator(prompt, max_length=512, num_return_sequences=1, truncation=True)
    full_response = responses[0]['generated_text'].strip()

    # Replace all newlines and carriage returns in the response
    full_response = full_response.replace('\n', ' ').replace('\r', ' ')

    # Attempt to remove the prompt from the response if it's included
    try:
        # Split response at the first question mark, assuming it marks the end of the repeated question
        parts = full_response.split('?')
        # Take everything after the first question mark which should be the actual answer
        answer_part = parts[1] if len(parts) > 1 else parts[0]

        # Further refine to get the first complete sentence from the actual answer
        first_dot_index = answer_part.find('.')
        first_sentence = answer_part[:first_dot_index + 1] if first_dot_index != -1 else answer_part
    except Exception as e:
        # If any error occurs in processing, default to the full response
        print(f"Error processing response: {e}")
        first_sentence = full_response

    return first_sentence.strip()

def dummy_score(llm_answer, ground_truth):
    return 0

# CSV file paths
input_file_path = 'TruthfulQA.csv'  # Your input CSV file
output_file_path = 'gpt2_mediumresponses.csv'  # Output CSV file

# Process and generate the dataset
with open(input_file_path, mode='r', newline='', encoding='utf-8') as file:
    reader = csv.DictReader(file)
    rows = list(reader)  

    # Define the fields for the output CSV
    fieldnames = ['Question', 'LLM Answer', 'Ground Truth Answer', 'Other Correct Answers', 'Ground Truth Score']

    with open(output_file_path, mode='w', newline='', encoding='utf-8') as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for row in tqdm(rows, desc="Processing rows"):
            question = row['Question']
            ground_truth = row['Best Answer']
            other_correct_answers = row.get('Correct Answers', '')
            llm_answer = generate_answer(question)
            score = dummy_score(llm_answer, ground_truth)
            
            # Create a dictionary for the current row and write it to the CSV
            output_row = {
                'Question': question,
                'LLM Answer': llm_answer,
                'Ground Truth Answer': ground_truth,
                'Other Correct Answers': other_correct_answers,
                'Ground Truth Score': score
            }
            writer.writerow(output_row)

print("Dataset generation complete with LLM responses.")
