import pandas as pd
import openai
import ast
import re
import nltk
import os
from tqdm import tqdm

# Download the NLTK data required for tokenization
nltk.download('punkt')

# Set your OpenAI API key
openai.api_key = 'Your_Key'  # Replace with your OpenAI API key

# Load the CSV files
base_path = "OMIn_dataset/gold_standard/processed"
ner_path = f"{base_path}/ner.csv"
nel_path = f"{base_path}/nel.csv"
cr_path = f"{base_path}/cr.csv"
re_path = f"{base_path}/re.csv"

# Directory to save GPT-4 results
results_dir = 'keo/vanilla_LLM/gpt4_results'

# Create the results directory if it doesn't exist
os.makedirs(results_dir, exist_ok=True)

# Read the datasets
ner_df = pd.read_csv(ner_path)
nel_df = pd.read_csv(nel_path)
cr_df = pd.read_csv(cr_path)
re_df = pd.read_csv(re_path)

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
    # Get unique samples
    samples_df = ner_df[['id', 'sample']].drop_duplicates()
    results = []
    print("Performing NER...")

    for _, row in tqdm(samples_df.iterrows(), total=len(samples_df), desc="NER Progress"):
        sample_id = row['id']
        sample_text = row['sample']

        # Prepare the prompt for NER
        prompt = f"Identify all entities in the following text:\n\n\"{sample_text}\"\n\nList the entities you find."

        # Call the OpenAI GPT-4 API
        response = openai.ChatCompletion.create(
            model="gpt-4-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0,
        )

        # Extract and process GPT-4 output
        gpt_output = response['choices'][0]['message']['content'].strip()
        entities = [line.strip() for line in gpt_output.split('\n') if line.strip()]
        results.append({'id': sample_id, 'entities_predicted': entities})

    # Save GPT-4 results to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(results_dir, 'ner_results.csv'), index=False)
    print(f"NER results saved to {os.path.join(results_dir, 'ner_results.csv')}")

    # Prepare gold standard data
    gold_standard = ner_df.groupby('id')['entities'].apply(list).reset_index()
    comparison_df = pd.merge(results_df, gold_standard, on='id')

    print("Evaluating NER...")

    # Evaluate performance
    total_TP = total_FP = total_FN = 0
    for _, row in tqdm(comparison_df.iterrows(), total=len(comparison_df), desc="NER Evaluation Progress"):
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
    # Parse the entity and qid columns
    nel_df['entity_list'] = parse_list_column(nel_df['entity'])
    nel_df['qid_list'] = parse_list_column(nel_df['qid'])
    samples_df = nel_df[['id', 'sample']].drop_duplicates()
    results = []
    print("Performing NEL...")

    for _, row in tqdm(samples_df.iterrows(), total=len(samples_df), desc="NEL Progress"):
        sample_id = row['id']
        sample_text = row['sample']

        # Prepare the prompt for NEL
        prompt = (
            f"Identify entities in the following text and link them to Wikidata QIDs:\n\n"
            f"\"{sample_text}\"\n\n"
            f"Provide the output in the format:\nEntity: <entity_name>\nQID: <QID>"
        )

        # Call the OpenAI GPT-4 API
        response = openai.ChatCompletion.create(
            model="gpt-4-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0,
        )

        # Extract and process GPT-4 output
        gpt_output = response['choices'][0]['message']['content'].strip()
        matches = re.findall(r'Entity:\s*(.+?)\s*QID:\s*(Q\d+)', gpt_output, re.IGNORECASE)
        entities_predicted = [match[0].strip() for match in matches]
        qids_predicted = [match[1].strip() for match in matches]
        results.append({
            'id': sample_id,
            'entities_predicted': entities_predicted,
            'qids_predicted': qids_predicted
        })

    # Save GPT-4 results to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(results_dir, 'nel_results.csv'), index=False)
    print(f"NEL results saved to {os.path.join(results_dir, 'nel_results.csv')}")

    # Prepare gold standard data
    gold_standard = nel_df.groupby('id').agg({
        'entity_list': lambda x: [item for sublist in x for item in sublist],
        'qid_list': lambda x: [item for sublist in x for item in sublist]
    }).reset_index()
    comparison_df = pd.merge(results_df, gold_standard, on='id')

    print("Evaluating NEL...")

    # Evaluate performance
    total_TP = total_FP = total_FN = 0
    for _, row in tqdm(comparison_df.iterrows(), total=len(comparison_df), desc="NEL Evaluation Progress"):
        true_pairs = set(zip(map(str.lower, filter(None, row['entity_list'])), filter(None, row['qid_list'])))
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
    # Parse coreferences and convert lists to tuples to make them hashable
    cr_df['coreferences_parsed'] = cr_df['coreferences'].apply(ast.literal_eval).apply(lambda x: [tuple(y) for y in x])

    # Perform drop_duplicates on relevant columns
    samples_df = cr_df[['id', 'sample']].drop_duplicates()

    results = []
    print("Performing CR...")

    # Iterate over each unique sample with a progress bar
    for _, row in tqdm(samples_df.iterrows(), total=len(samples_df), desc="CR Progress"):
        sample_id = row['id']
        sample_text = row['sample']

        # Prepare the prompt for CR
        prompt = (
            f"Perform coreference resolution on the following text. Identify clusters of expressions that refer to the same entity:\n\n"
            f"\"{sample_text}\"\n\n"
            f"Provide the clusters in the format:\nCluster 1: [\"mention1\", \"mention2\"]"
        )

        # Call the GPT-4-turbo API
        response = openai.ChatCompletion.create(
            model="gpt-4-turbo",  # Use the cheaper GPT-4 model
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0,
        )

        # Extract and process GPT-4 output
        gpt_output = response['choices'][0]['message']['content'].strip()
        cluster_matches = re.findall(r'Cluster\s*\d+:\s*\[(.*?)\]', gpt_output)
        clusters = [re.findall(r'"(.*?)"', cluster) for cluster in cluster_matches]
        results.append({'id': sample_id, 'clusters_predicted': clusters})

    # Save GPT-4 results to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(results_dir, 'cr_results.csv'), index=False)
    print(f"CR results saved to {os.path.join(results_dir, 'cr_results.csv')}")

    print("Evaluating CR...")

    # Evaluation logic
    total_TP = total_FP = total_FN = 0
    for _, row in tqdm(samples_df.iterrows(), total=len(samples_df), desc="CR Evaluation Progress"):
        sample_text = row['sample']
        tokens = nltk.word_tokenize(sample_text)
        gold_coreferences = cr_df[cr_df['id'] == row['id']]['coreferences_parsed'].iloc[0]

        # Extract gold standard mentions
        gold_clusters = []
        for cluster in gold_coreferences:
            mentions = [' '.join(tokens[start:end + 1]) for start, end in cluster]
            gold_clusters.append(mentions)

        # Retrieve predicted clusters for the current sample
        predicted_row = next((item for item in results if item['id'] == row['id']), {})
        predicted_clusters = predicted_row.get('clusters_predicted', [])

        # Flatten the clusters and convert to lowercase for comparison
        gold_mentions = set(map(str.lower, [m for cluster in gold_clusters for m in cluster]))
        pred_mentions = set(map(str.lower, [m for cluster in predicted_clusters for m in cluster]))

        # Calculate True Positives, False Positives, and False Negatives
        TP = len(gold_mentions & pred_mentions)
        FP = len(pred_mentions - gold_mentions)
        FN = len(gold_mentions - pred_mentions)
        total_TP += TP
        total_FP += FP
        total_FN += FN

    # Calculate Precision, Recall, and F1-score
    precision = total_TP / (total_TP + total_FP) if total_TP + total_FP else 0
    recall = total_TP / (total_TP + total_FN) if total_TP + total_FN else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0

    # Print the evaluation results
    print(f"CR Results: Precision={precision:.4f}, Recall={recall:.4f}, F1-score={f1:.4f}")

def perform_re(re_df):
    # Get samples from ner_df
    ner_samples = ner_df[['id', 'sample']].drop_duplicates()
    re_df = pd.merge(re_df, ner_samples, on='id')
    samples_df = re_df[['id', 'sample']].drop_duplicates()
    results = []
    print("Performing RE...")

    for _, row in tqdm(samples_df.iterrows(), total=len(samples_df), desc="RE Progress"):
        sample_id = row['id']
        sample_text = row['sample']

        # Prepare the prompt for RE
        prompt = (
            f"Extract all relations from the following text. Identify subject, relation, and object triples:\n\n"
            f"\"{sample_text}\"\n\n"
            f"Provide the output in the format:\nSubject: <subject>\nRelation: <relation>\nObject: <object>"
        )

        # Call the OpenAI GPT-4 API
        response = openai.ChatCompletion.create(
            model="gpt-4-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0,
        )

        # Extract and process GPT-4 output
        gpt_output = response['choices'][0]['message']['content'].strip()
        triple_matches = re.findall(
            r'Subject:\s*(.*?)\s*Relation:\s*(.*?)\s*Object:\s*(.*?)(?:\n|$)', gpt_output, re.DOTALL
        )
        triples = [(match[0].strip(), match[1].strip(), match[2].strip()) for match in triple_matches]
        results.append({'id': sample_id, 'triples_predicted': triples})

    # Save GPT-4 results to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(results_dir, 're_results.csv'), index=False)
    print(f"RE results saved to {os.path.join(results_dir, 're_results.csv')}")

    # Prepare gold standard data
    gold_standard = re_df.groupby('id').apply(
        lambda x: list(zip(x['subject'], x['relation'], x['object']))
    ).reset_index().rename(columns={0: 'triples_gold'})
    comparison_df = pd.merge(pd.DataFrame(results), gold_standard, on='id')

    print("Evaluating RE...")

    # Evaluate performance
    total_TP = total_FP = total_FN = 0
    for _, row in tqdm(comparison_df.iterrows(), total=len(comparison_df), desc="RE Evaluation Progress"):
        gold_triples = set(map(lambda t: tuple(map(str.lower, t)), row['triples_gold']))
        pred_triples = set(map(lambda t: tuple(map(str.lower, t)), row['triples_predicted']))
        TP = len(gold_triples & pred_triples)
        FP = len(pred_triples - gold_triples)
        FN = len(gold_triples - pred_triples)
        total_TP += TP
        total_FP += FP
        total_FN += FN

    precision = total_TP / (total_TP + total_FP) if total_TP + total_FP else 0
    recall = total_TP / (total_TP + total_FN) if total_TP + total_FN else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
    print(f"RE Results: Precision={precision:.4f}, Recall={recall:.4f}, F1-score={f1:.4f}")

# Run the tasks
perform_ner(ner_df)
perform_nel(nel_df)
perform_cr(cr_df)
perform_re(re_df)
