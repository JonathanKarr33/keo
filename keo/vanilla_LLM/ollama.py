import pandas as pd
import requests
import ast
import re
import nltk
import json
import os
from tqdm import tqdm
from utils.evaluate_cr import evaluate_cr
from utils.evaluate_re import evaluate_re
from utils.evaluate_ner import evaluate_ner

# Download NLTK tokenizer if not already available
nltk.download('punkt')

# Ollama API setup
OLLAMA_API_URL = 'http://localhost:11434'
MODEL_NAME = 'llama3.2'  # Adjust if your model name is different

# File paths
base_path = "OMIn_dataset/gold_standard/raw"
ner_path = f"{base_path}/ner.csv"
cr_path = f"{base_path}/cr.csv"
re_path = f"{base_path}/re.csv"

# Directory to save Llama results
results_dir = 'keo/vanilla_LLM/llama32_3B_results'

# Create the results directory if it doesn't exist
os.makedirs(results_dir, exist_ok=True)

# Load the datasets
ner_df = pd.read_csv(ner_path)
cr_df = pd.read_csv(cr_path)
re_df = pd.read_csv(re_path)

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

# Function to parse list columns from strings
def parse_list_column(column):
    parsed = []
    for item in column:
        try:
            parsed_item = ast.literal_eval(item)
            parsed.append(parsed_item)
        except:
            parsed.append([])
    return parsed

def perform_ner(ner_df):
    # Adjust the DataFrame to match the raw data structure
    ner_df.rename(columns={'c5_unique_id': 'id', 'c119_text': 'sample'}, inplace=True)
    ner_df['entities'] = ner_df['GS'].apply(ast.literal_eval)

    samples_df = ner_df[['id', 'sample']].drop_duplicates()
    results = []

    print("Running NER...")
    for index, row in tqdm(samples_df.iterrows(), total=len(samples_df), desc="NER Progress"):
        sample_id = row['id']
        sample_text = row['sample']
        
        # Improved prompt with few-shot examples
        prompt = (
            f"You are an AI assistant that performs Named Entity Recognition (NER). "
            f"Identify all entities in the following text, and provide their types:\n\n"
            f"Examples:\n"
            f"Text: \"ACFT WAS TAXIING FOR TAKE OFF WHEN IT LOST CONTROL, RAN INTO A DITCH, AND STRUCK A TREE. OTHER CIRCUMSTANCES AE UNK\"\n"
            f"Output: Entities: [\"ACFT\"], Types: [\"VEHICLE\"]\n"
            f"Text: \"AFTER TAKEOFF, ENGINE QUIT. WING FUEL TANK SUMPS WERE NOT DRAINED DURING PREFLIGHT BECAUSE THEY WERE FROZEN.\"\n"
            f"Output: Entities: [\"TAKEOFF\", \"ENGINE\", \"WING FUEL TANK SUMPS\", \"PREFLIGHT\"], Types: [\"PHASE\", \"SYSTEM\", \"SYSTEM\", \"PROCEDURE\"]\n\n"
            f"Now, identify the entities and their types for the following text:\n\n"
            f"Text: \"{sample_text}\"\n"
            f"Output: Entities:"
        )

        # Call the Ollama API
        gpt_output = ollama_completion(prompt)

        # Extract and process the output
        try:
            # Regex to find the entities and types in the output
            entities_match = re.search(r'Entities:\s*\[(.*?)\]', gpt_output)
            entity_types_match = re.search(r'Types:\s*\[(.*?)\]', gpt_output)

            # Extract entities and entity types if they exist, otherwise set as empty list
            entities = ast.literal_eval(f"[{entities_match.group(1)}]") if entities_match else []
            entity_types = ast.literal_eval(f"[{entity_types_match.group(1)}]") if entity_types_match else []

            results.append({'id': sample_id, 'sample': sample_text, 'entities_predicted': entities, 'entity_types_predicted': entity_types})
        except Exception as e:
            print(f"Error processing output for sample {sample_id}: {e}")
            results.append({'id': sample_id, 'sample': sample_text, 'entities_predicted': [], 'entity_types_predicted': []})

    # Save results to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(results_dir, 'ner_results.csv'), index=False)
    print(f"NER results saved to {os.path.join(results_dir, 'ner_results.csv')}")

    # Compare results with gold standard
    print("Evaluating ner...")
    evaluate_ner(ner_path, os.path.join(results_dir, 'ner_results.csv'), os.path.join(results_dir, 'ner_score.txt'))

# Function to safely parse strings to Python literals
def safe_literal_eval(val):
    try:
        return ast.literal_eval(val)
    except (ValueError, SyntaxError):
        return []

def perform_cr(cr_df):
    # Parse coreferences from the 'coreferences' column
    cr_df['coreferences_parsed'] = cr_df['coreferences'].apply(safe_literal_eval)
    cr_df['coreferences_readable'] = cr_df['coreferences_human_readable'].apply(safe_literal_eval)

    # Get unique samples
    samples_df = cr_df[['c5', 'c119_text']].drop_duplicates()
    results = []
    print("Performing CR...")

    for _, row in tqdm(samples_df.iterrows(), total=len(samples_df), desc="CR Progress"):
        sample_id = row['c5']
        sample_text = row['c119_text']

        # Prepare the prompt for CR with few-shot examples
        prompt = (
            "You are an AI assistant that performs Coreference Resolution (CR). "
            "Identify clusters of expressions that refer to the same entity in the following text:\n\n"
            "Examples:\n"
            "Text: \"ACFT WAS TAXIING FOR TAKE OFF WHEN IT LOST CONTROL, RAN INTO A DITCH, AND STRUCK A TREE. OTHER CIRCUMSTANCES AE UNK\"\n"
            "Output:\n"
            "Cluster 1: [\"ACFT\", \"IT\"]\n"
            "Text: \"AFTER TAKEOFF, ENGINE QUIT. WING FUEL TANK SUMPS WERE NOT DRAINED DURING PREFLIGHT BECAUSE THEY WERE FROZEN.\"\n"
            "Output:\n"
            "Cluster 1: [\"WING FUEL TANK SUMPS\", \"THEY\"]\n\n"
            "Now, perform coreference resolution on the following text:\n"
            f"Text: \"{sample_text}\"\n"
            "Output:")

        # Call the Ollama API
        gpt_output = ollama_completion(prompt)
        try:
            # Regex to find clusters in the output
            cluster_matches = re.findall(r'Cluster\s*\d+:\s*\[(.*?)\]', gpt_output)
            clusters = [re.findall(r'"(.*?)"', cluster) for cluster in cluster_matches]

            # Generate coreferences in the required format
            tokens = list(nltk.word_tokenize(sample_text))
            coreferences_human_readable = [mention for cluster in clusters for mention in cluster]
            coreferences = []

            for cluster in clusters:
                cluster_indices = []
                for mention in cluster:
                    mention_tokens = nltk.word_tokenize(mention)
                    start_index = None
                    for i in range(len(tokens) - len(mention_tokens) + 1):
                        if tokens[i:i + len(mention_tokens)] == mention_tokens:
                            start_index = i
                            break
                    if start_index is not None:
                        end_index = start_index + len(mention_tokens) - 1
                        cluster_indices.append([start_index, end_index])
                coreferences.append(cluster_indices)

            results.append({
                'c5': sample_id,
                'c119_text': sample_text,
                'coreferences_human_readable': coreferences_human_readable,
                'coreferences': coreferences
            })
        except Exception as e:
            print(f"Error processing output for sample {sample_id}: {e}")
            results.append({
                'c5': sample_id,
                'c119_text': sample_text,
                'coreferences_human_readable': [],
                'coreferences': []
            })

    # Save Llama results to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(results_dir, 'cr_results.csv'), index=False)
    print(f"CR results saved to {os.path.join(results_dir, 'cr_results.csv')}")

    print("Evaluating CR...")
    evaluate_cr(cr_path, os.path.join(results_dir, 'cr_results.csv'), os.path.join(results_dir, 'cr_score.txt'))

def parse_re_output(gpt_output):
    """
    Parse the GPT output to extract relation triples with improved handling for malformed data.
    """
    # Normalize newlines and clean up whitespace
    gpt_output = gpt_output.strip().replace("\r", "").replace("\n\n", "\n").strip()

    # Regular expression to match complete triples
    triple_pattern = re.compile(
        r"Subject:\s*(.*?)\nSubject Type:\s*(.*?)\nRelation:\s*(.*?)\nObject:\s*(.*?)\nObject Type:\s*(.*?)(?:\n|$)",
        re.DOTALL
    )
    
    # Find all matches
    matches = triple_pattern.findall(gpt_output)
    results = []

    # Handle malformed or incomplete triples
    lines = gpt_output.split("\n")
    current_triple = {"Subject": None, "Subject Type": None, "Relation": None, "Object": None, "Object Type": None}
    keys = list(current_triple.keys())
    key_idx = 0

    for line in lines:
        # Detect and assign parts of triples
        for key in keys[key_idx:]:
            if line.startswith(f"{key}:"):
                current_triple[key] = line.split(f"{key}:")[1].strip()
                key_idx += 1
                break

        # If a complete triple is detected, add it to results
        if all(current_triple.values()):
            results.append(current_triple)
            current_triple = {key: None for key in keys}  # Reset for next triple
            key_idx = 0

    # Combine results from regex matches and reconstructed triples
    for match in matches:
        results.append({
            "Subject": match[0].strip(),
            "Subject Type": match[1].strip(),
            "Relation": match[2].strip(),
            "Object": match[3].strip(),
            "Object Type": match[4].strip(),
        })

    return results


def perform_re(re_df):
    """
    Perform Relation Extraction (RE) using GPT/Ollama API and save the results.

    Parameters:
    - re_df (DataFrame): DataFrame containing input samples for RE.
    - results_dir (str): Directory to save the results.
    """
    # Replace NaN values with an empty string to avoid AttributeError
    re_df['entity1, relation_type, entity2'] = re_df['entity1, relation_type, entity2'].fillna('')

    # Prepare entity-relation pairs
    re_df['relations'] = re_df['entity1, relation_type, entity2'].apply(
        lambda x: x.split('\n') if isinstance(x, str) else []
    )

    samples_df = re_df[['c5_unique_id', 'c119_text']].drop_duplicates()
    results = []
    print("Performing RE...")

    for _, row in tqdm(samples_df.iterrows(), total=len(samples_df), desc="RE Progress"):
        sample_id = row['c5_unique_id']
        sample_text = row['c119_text']

        # Prepare the prompt for RE with few-shot examples
        prompt = (
            "You are an AI assistant that performs Relation Extraction (RE). "
            "Identify subject, relation, and object triples along with their types from the following text:\n\n"
            "Here are some possible relations to choose from: has effect, has cause, followed by, part of, instance of, facet of, used by, location, time period, owned by, owner of, influenced by, maintained by, conflict, event distance, designed by, located in the administrative territorial entity, located in or next to body of water\n"
            "Examples:\n"
            "Text: \"ACFT WAS TAXIING FOR TAKE OFF WHEN IT LOST CONTROL, RAN INTO A DITCH, AND STRUCK A TREE. OTHER CIRCUMSTANCES AE UNK\"\n"
            "Output:\n"
            "Subject: lost control\nSubject Type: action\nRelation: has effect\nObject: ran into a ditch\nObject Type: event\n"
            "Subject: lost control\nSubject Type: action\nRelation: has effect\nObject: struck a tree\nObject Type: event\n"
            "Remember always follow the format above and do not add any other information. Now, extract all relations from the following text:\n"
            f"Text: \"{sample_text}\"\n"
            "Output:"
        )

        # Call the Ollama API
        gpt_output = ollama_completion(prompt)
        try:
            # Parse the GPT output
            parsed_triples = parse_re_output(gpt_output)
            for triple in parsed_triples:
                results.append({
                    'c5_unique_id': sample_id,
                    'sample': sample_text,
                    'subject': triple["Subject"],
                    'subject_type': triple["Subject Type"],
                    'relation': triple["Relation"],
                    'object': triple["Object"],
                    'object_type': triple["Object Type"]
                })
        except Exception as e:
            print(f"Error processing output for sample {sample_id}: {e}")

    # Save results to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(results_dir, 're_results.csv'), index=False)
    print(f"RE results saved to {os.path.join(results_dir, 're_results.csv')}")

    print("Evaluating RE...")
    evaluate_re(re_path, os.path.join(results_dir, 're_results.csv'), os.path.join(results_dir, 're_score.txt'))

if __name__ == "__main__":
    # Run the tasks
    # perform_ner(ner_df)
    # perform_cr(cr_df)
    perform_re(re_df)
