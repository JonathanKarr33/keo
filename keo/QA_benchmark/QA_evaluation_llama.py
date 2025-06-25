import pandas as pd
from tqdm import tqdm
import requests
import json

# Ollama API setup
OLLAMA_API_URL = 'http://localhost:11434'
MODEL_NAME = 'llama3.2'  # Adjust if your model name is different

# Helper function for Ollama completion requests
def ollama_completion(prompt, max_tokens=500, temperature=0.0):
    headers = {'Content-Type': 'application/json'}
    data = {
        'model': MODEL_NAME,
        'prompt': prompt,
        'temperature': temperature,
        'max_tokens': max_tokens,
        'stop': ['###']
    }

    try:
        response = requests.post(f'{OLLAMA_API_URL}/api/generate', headers=headers, data=json.dumps(data), stream=True)
        response.raise_for_status()  # Raise exception for HTTP errors

        # Process the streaming response
        output = ""
        for line in response.iter_lines():
            if line:  # Avoid empty lines
                try:
                    chunk = json.loads(line.decode('utf-8'))
                    output += chunk.get('response', '')
                except json.JSONDecodeError as e:
                    print(f"Failed to decode JSON chunk: {line}")
                    raise

        return output.strip()

    except requests.RequestException as e:
        print(f"Request failed: {e}")
        raise

# File paths
# input_file_path_list = ["VanillaGPT4o_QA_Answers.csv", "GPT4o_GraphRAG_spacy_QA_Answers.csv", "GPT4o_GraphRAG_loose_QA_Answers.csv", "GPT4o_GraphRAG_strict_QA_Answers.csv"]
# output_file_path_list = ["VanillaGPT4o_QA_Evaluation_llama.csv", "GPT4o_GraphRAG_spacy_QA_Evaluation_llama.csv", "GPT4o_GraphRAG_loose_QA_Evaluation_llama.csv", "GPT4o_GraphRAG_strict_QA_Evaluation_llama.csv"]
input_file_path_list = ["GPT4o_GraphRAG_spacy_QA_Answers.csv"]
output_file_path_list = ["GPT4o_GraphRAG_spacy_QA_Evaluation_llama.csv"]

for input_file_path, output_file_path in zip(input_file_path_list, output_file_path_list):
    # Load the QA answers dataset
    print(f"Loading answers data from {input_file_path}...")
    qa_data = pd.read_csv(input_file_path)

    # Initialize an empty list to store evaluated results
    evaluated_qa = []

    # Process each QA pair
    print("Evaluating answers with Llama3.2 (via Ollama) including Incident Description...")
    for idx, row in tqdm(qa_data.iterrows(), total=len(qa_data), desc="Evaluating QA"):
        c5 = row['c5']
        c119_text = row['c119_text']  # Incident Description
        question = row['question']
        answer = row['answer']

        # Prepare the prompt for evaluation
        prompt = f"""
You are a highly knowledgeable and objective aviation safety expert tasked with evaluating the quality of answers to aviation-related questions. 

For the given Incident Description, question, and answer, provide a rating on a scale of 0 to 5 (integer only):
- 0: The answer is completely unrelated to the question or incident description.
- 1: The answer is minimally relevant but mostly incorrect or incomplete.
- 2: The answer shows some relevance but has significant inaccuracies or lacks depth.
- 3: The answer is moderately relevant with minor issues or missing details.
- 4: The answer is relevant and mostly accurate with only minor improvements needed.
- 5: The answer is highly relevant, accurate, complete, and perfectly addresses the question and incident description.

**Incident Description:**
{c119_text}

**Question:**
{question}

**Answer:**
{answer}

Provide your rating along with a brief justification for your rating in the following format:
Rating: <0-5>
Justification: <Your reasoning>
"""

        # Query the Llama3.2 model via Ollama
        try:
            evaluation_output = ollama_completion(prompt)
            # Extract rating and justification
            rating = None
            justification = None
            for line in evaluation_output.split("\n"):
                if line.strip().startswith("Rating:"):
                    # Extract the integer rating
                    line_content = line.replace("Rating:", "").strip()
                    # Safely parse the rating
                    try:
                        rating = int(line_content)
                    except ValueError:
                        rating = -1
                if line.strip().startswith("Justification:"):
                    justification = line.replace("Justification:", "").strip()
        except Exception as e:
            rating = -1  # Indicate an error occurred during evaluation
            justification = f"Error processing evaluation: {str(e)}"

        # Append the evaluated results
        evaluated_qa.append({
            "c5": c5,
            "incident_description": c119_text,
            "question": question,
            "answer": answer,
            "rating": rating,
            "justification": justification
        })

    # Convert the results to a DataFrame
    output_df = pd.DataFrame(evaluated_qa)

    # Save the evaluated QA results to a CSV file
    print(f"Saving evaluated QA with ratings to {output_file_path}...")
    output_df.to_csv(output_file_path, index=False)

    print("Evaluation complete!")
