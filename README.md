# Raniya Khan - Lynxius Technical Task

## Overview

This project aims to assess the accuracy and relevance of answers generated by Large Language Models (LLMs) to questions from the "TruthfulQA" dataset. The dataset is specifically curated to challenge LLMs with questions steeped in misconceptions, myths, and conspiracy theories. LLM responses were generated using GPT-2 variants (small, medium, and large) to compare response quality.

## Dataset and File Structure

### original_data/

### original_data/

Contains the original ["TruthfulQA.csv"](https://huggingface.co/datasets/truthful_qa) file from Hugging Face.

### bertscores/

Houses CSV files for each model variant (small, medium, large) with ground truth scores calculated using BERT embeddings.

### sbert+roberta_scores/

Includes files where scores are computed using SBERT for semantic similarity and RoBERTa for factual accuracy.

### visuals/

Scripts for generating statistical visuals like box plots and histograms to analyze score distributions.

### dataGenerator.py

Script to generate LLM responses for the dataset.

### future_iterations/

Proposes new methods for evaluating LLM responses using more advanced (and costly) APIs such as GPT-4 and some Langchain libraries.

--------------------

This was a fun learning experience! I know there's a lot of room for improvement in my work, but I'm willing to put the work in to make it happen.
