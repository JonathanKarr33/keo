import pandas as pd
import requests
import ast
import re
import nltk
import json
import os
from tqdm import tqdm

# Download NLTK tokenizer if not already available
nltk.download('punkt')
nltk.download('punkt_tab')

# Ollama API setup
OLLAMA_API_URL = 'http://localhost:11434'
MODEL_NAME = 'llama3.1'  # Adjust if your model name is different

# File paths
base_path = "OMIn_dataset/gold_standard/processed"
ner_path = f"{base_path}/ner.csv"
nel_path = f"{base_path}/nel.csv"
cr_path = f"{base_path}/cr.csv"
re_path = f"{base_path}/re.csv"

# Directory to save Llama results
results_dir = 'keo/vanilla_LLM/llama31_results'

# Create the results directory if it doesn't exist
os.makedirs(results_dir, exist_ok=True)

# Load the datasets
ner_df = pd.read_csv(ner_path)
nel_df = pd.read_csv(nel_path)
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
    samples_df = ner_df[['id', 'sample']].drop_duplicates()
    results = []

    print("Running NER...")
    for index, row in tqdm(samples_df.iterrows(), total=len(samples_df), desc="NER Progress"):
        sample_id = row['id']
        sample_text = row['sample']
        prompt = f"""You are an AI assistant that performs Named Entity Recognition (NER).
Identify all entities in the following text:

"{sample_text}"

List the entities you find, one per line.
"""

        gpt_output = ollama_completion(prompt)
        entities = [line.strip() for line in gpt_output.split('\n') if line.strip()]
        results.append({'id': sample_id, 'entities_predicted': entities})

    # Save results to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(results_dir, 'ner_results.csv'), index=False)
    print(f"NER results saved to {os.path.join(results_dir, 'ner_results.csv')}")

    # Compare results with gold standard
    evaluate_ner(results, ner_df)

def evaluate_ner(results, ner_df):
    gold_standard = ner_df.groupby('id')['entities'].apply(list).reset_index()
    comparison_df = pd.merge(pd.DataFrame(results), gold_standard, on='id')
    total_TP = total_FP = total_FN = 0

    for _, row in comparison_df.iterrows():
        true_entities = set(map(str.lower, row['entities']))
        pred_entities = set(map(str.lower, row['entities_predicted']))
        TP = len(true_entities & pred_entities)
        FP = len(pred_entities - true_entities)
        FN = len(true_entities - pred_entities)
        total_TP += TP
        total_FP += FP
        total_FN += FN

    precision = total_TP / (total_TP + total_FP) if total_TP + total_FP else 0
    recall = total_TP / (total_TP + total_FN) if total_TP + total_FN else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
    print(f"NER Results: Precision={precision:.4f}, Recall={recall:.4f}, F1-score={f1:.4f}")

def perform_nel(nel_df):
    nel_df['entity_list'] = parse_list_column(nel_df['entity'])
    nel_df['qid_list'] = parse_list_column(nel_df['qid'])
    samples_df = nel_df[['id', 'sample']].drop_duplicates()
    results = []

    print("Running NEL...")
    for _, row in tqdm(samples_df.iterrows(), total=len(samples_df), desc="NEL Progress"):
        sample_id = row['id']
        sample_text = row['sample']
        prompt = f"""You are an AI assistant that performs Named Entity Linking (NEL).
Identify entities in the following text and link them to Wikidata QIDs:

"{sample_text}"

Provide the output in the format:
Entity: <entity_name>
QID: <QID>
"""

        gpt_output = ollama_completion(prompt)
        matches = re.findall(r'Entity:\s*(.+?)\s*QID:\s*(Q\d+)', gpt_output, re.IGNORECASE)
        entities_predicted = [match[0].strip() for match in matches]
        qids_predicted = [match[1].strip() for match in matches]
        results.append({'id': sample_id, 'entities_predicted': entities_predicted, 'qids_predicted': qids_predicted})

    # Save results to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(results_dir, 'nel_results.csv'), index=False)
    print(f"NEL results saved to {os.path.join(results_dir, 'nel_results.csv')}")

    # Compare results with gold standard
    evaluate_nel(results, nel_df)

def evaluate_nel(results, nel_df):
    gold_standard = nel_df.groupby('id').agg({
        'entity_list': lambda x: [item for sublist in x for item in sublist],
        'qid_list': lambda x: [item for sublist in x for item in sublist]
    }).reset_index()
    comparison_df = pd.merge(pd.DataFrame(results), gold_standard, on='id')
    total_TP = total_FP = total_FN = 0

    for _, row in comparison_df.iterrows():
        true_pairs = set(zip(map(str.lower, row['entity_list']), row['qid_list']))
        pred_pairs = set(zip(map(str.lower, row['entities_predicted']), row['qids_predicted']))
        TP = len(true_pairs & pred_pairs)
        FP = len(pred_pairs - true_pairs)
        FN = len(true_pairs - pred_pairs)
        total_TP += TP
        total_FP += FP
        total_FN += FN

    precision = total_TP / (total_TP + total_FP) if total_TP + total_FP else 0
    recall = total_TP / (total_TP + total_FN) if total_TP + total_FN else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
    print(f"NEL Results: Precision={precision:.4f}, Recall={recall:.4f}, F1-score={f1:.4f}")

def perform_cr(cr_df):
    cr_df['coreferences_parsed'] = cr_df['coreferences'].apply(ast.literal_eval)
    samples_df = cr_df[['id', 'sample']].drop_duplicates()
    results = []

    print("Running CR...")
    for _, row in tqdm(samples_df.iterrows(), total=len(samples_df), desc="CR Progress"):
        sample_id = row['id']
        sample_text = row['sample']
        prompt = f"""You are an AI assistant that performs Coreference Resolution (CR).
Perform coreference resolution on the following text. Identify clusters of expressions that refer to the same entity:

"{sample_text}"

Provide the clusters in the format:
Cluster 1: ["mention1", "mention2"]
"""

        gpt_output = ollama_completion(prompt)
        cluster_matches = re.findall(r'Cluster\s*\d+:\s*\[(.*?)\]', gpt_output)
        clusters = [re.findall(r'"(.*?)"', cluster) for cluster in cluster_matches]
        results.append({'id': sample_id, 'clusters_predicted': clusters})

    # Save results to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(results_dir, 'cr_results.csv'), index=False)
    print(f"CR results saved to {os.path.join(results_dir, 'cr_results.csv')}")

    # Evaluation can be added similarly
    # evaluate_cr(results, cr_df)  # Implement evaluate_cr function if needed

def perform_re(re_df):
    samples_df = re_df[['id', 'sample']].drop_duplicates()
    results = []

    print("Running RE...")
    for _, row in tqdm(samples_df.iterrows(), total=len(samples_df), desc="RE Progress"):
        sample_id = row['id']
        sample_text = row['sample']
        prompt = f"""You are an AI assistant that performs Relation Extraction (RE).
Extract all relations from the following text. Identify subject, relation, and object triples:

"{sample_text}"

Provide the output in the format:
Subject: <subject>
Relation: <relation>
Object: <object>
"""

        gpt_output = ollama_completion(prompt)
        triple_matches = re.findall(r'Subject:\s*(.*?)\s*Relation:\s*(.*?)\s*Object:\s*(.*?)(?:\n|$)', gpt_output, re.DOTALL)
        triples = [(match[0].strip(), match[1].strip(), match[2].strip()) for match in triple_matches]
        results.append({'id': sample_id, 'triples_predicted': triples})

    # Save results to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(results_dir, 're_results.csv'), index=False)
    print(f"RE results saved to {os.path.join(results_dir, 're_results.csv')}")

    # Evaluation can be added similarly
    # evaluate_re(results, re_df)  # Implement evaluate_re function if needed

# Run the tasks
perform_ner(ner_df)
perform_nel(nel_df)
perform_cr(cr_df)
perform_re(re_df)
