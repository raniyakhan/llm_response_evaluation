import json
import openai
from typing import List
from pydantic import BaseModel, Field

# Initialize the OpenAI client
openai.api_key = 'your-api-key'

class Propositions(BaseModel):
    propositions: List[str] = Field(description="Factual propositions extracted from the text")

class CorrectnessScore(BaseModel):
    score: bool
    reasoning: str = Field(description="The reasoning for the correctness score")

def extract_propositions(text: str) -> List[str]:
    """ Extract propositions using a language model """
    response = openai.ChatCompletion.create(
        model="gpt-4-turbo-preview",
        messages=[
            {"role": "user", "content": f"List all factual assertions from this response: {text}"}
        ]
    )
    # Assume response format adjustment
    return json.loads(response['choices'][0]['message']['content'])['propositions']

def evaluate_correctness(propositions: List[str], answers: List[str], ground_truth: str, ground_truth_weight: float) -> dict:
    """ Evaluate correctness of propositions against multiple correct answers with weighted ground truth """
    scores = []
    reasoning = []

    for proposition in propositions:
        truth_scores = []
        for answer in answers:
            response = openai.ChatCompletion.create(
                model="gpt-4-turbo-preview",
                messages=[
                    {
                        "role": "user",
                        "content": f"Is this statement correct based on the correct answers? Statement: {proposition}, Answer: {answer}"
                    }
                ]
            )
            result = json.loads(response['choices'][0]['message']['content'])
            truth_scores.append(result['score'])

        # Evaluate against ground truth
        response_gt = openai.ChatCompletion.create(
            model="gpt-4-turbo-preview",
            messages=[
                {
                    "role": "user",
                    "content": f"Is this statement correct based on the ground truth? Statement: {proposition}, Ground truth: {ground_truth}"
                }
            ]
        )
        result_gt = json.loads(response_gt['choices'][0]['message']['content'])
        # Weight the ground truth more
        combined_score = (result_gt['score'] * ground_truth_weight + max(truth_scores) * (1 - ground_truth_weight))
        scores.append(combined_score)
        reasoning.append(result_gt['reasoning'])

    average_score = sum(scores) / len(scores) if scores else None
    comment = " ".join(reasoning)
    return {"average_score": average_score, "reasoning": comment}

# Example usage
data = {
    "LLM Answer": "Your response here.",
    "Ground Truth Answer": "Your ground truth answer.",
    "Other Correct Answers": ["Other correct answer1", "Other correct answer2"]
}
propositions = extract_propositions(data['LLM Answer'])
correctness_info = evaluate_correctness(propositions, data['Other Correct Answers'], data['Ground Truth Answer'], ground_truth_weight=0.7)
print(correctness_info)
