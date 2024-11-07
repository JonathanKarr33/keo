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
openai.api_key = "Your_OpenAI_Key"  # Replace with your OpenAI API key

# Load the CSV files from the 'raw' directory
base_path = "OMIn_dataset/gold_standard/raw"
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

        # Call the OpenAI GPT-4o API
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0,
        )

        # Extract and process GPT-4 output
        gpt_output = response['choices'][0]['message']['content'].strip()

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
    evaluate_ner(results, ner_df)

# Evaluation function for NER
def evaluate_ner(results, ner_df):
    # Prepare gold standard data
    gold_standard = ner_df.groupby('id').agg({
        'entities': lambda x: [entity for sublist in x for entity in sublist],
        'GS TYPE': lambda x: [etype for sublist in x for etype in ast.literal_eval(sublist)]
    }).reset_index()

    comparison_df = pd.merge(pd.DataFrame(results), gold_standard, on='id', how='left')
    total_TP_entities = total_FP_entities = total_FN_entities = 0
    total_TP_types = total_FP_types = total_FN_types = 0

    for _, row in comparison_df.iterrows():
        true_entities = set(map(str.lower, row['entities']))
        pred_entities = set(map(str.lower, row['entities_predicted']))
        TP_entities = len(true_entities & pred_entities)
        FP_entities = len(pred_entities - true_entities)
        FN_entities = len(true_entities - pred_entities)
        total_TP_entities += TP_entities
        total_FP_entities += FP_entities
        total_FN_entities += FN_entities

        true_types = set(map(str.lower, row['GS TYPE']))
        pred_types = set(map(str.lower, row['entity_types_predicted']))
        TP_types = len(true_types & pred_types)
        FP_types = len(pred_types - true_types)
        FN_types = len(true_types - pred_types)
        total_TP_types += TP_types
        total_FP_types += FP_types
        total_FN_types += FN_types

    # Calculating metrics for entities
    precision_entities = total_TP_entities / (total_TP_entities + total_FP_entities) if (total_TP_entities + total_FP_entities) else 0
    recall_entities = total_TP_entities / (total_TP_entities + total_FN_entities) if (total_TP_entities + total_FN_entities) else 0
    f1_entities = 2 * precision_entities * recall_entities / (precision_entities + recall_entities) if (precision_entities + recall_entities) else 0

    # Calculating metrics for entity types
    precision_types = total_TP_types / (total_TP_types + total_FP_types) if (total_TP_types + total_FP_types) else 0
    recall_types = total_TP_types / (total_TP_types + total_FN_types) if (total_TP_types + total_FN_types) else 0
    f1_types = 2 * precision_types * recall_types / (precision_types + recall_types) if (precision_types + recall_types) else 0

    # Outputting the results
    print(f"NER Results (Entities): Precision={precision_entities:.4f}, Recall={recall_entities:.4f}, F1-score={f1_entities:.4f}")
    print(f"NER Results (Entity Types): Precision={precision_types:.4f}, Recall={recall_types:.4f}, F1-score={f1_types:.4f}")

def perform_nel(nel_df):
    # Prepare entity-QID pairs
    nel_df['entities'] = nel_df[['primary_ent', 'secondary_ent', 'tertiary_ent']].values.tolist()
    nel_df['qids'] = nel_df[['primary_qid', 'secondary_qid', 'tertiary_qid']].values.tolist()

    # Clean up NaN values
    nel_df['entities'] = nel_df['entities'].apply(lambda x: [e for e in x if pd.notna(e)])
    nel_df['qids'] = nel_df['qids'].apply(lambda x: [q for q in x if pd.notna(q)])

    samples_df = nel_df[['id', 'sample']].drop_duplicates()
    results = []

    print("Performing NEL...")
    for _, row in tqdm(samples_df.iterrows(), total=len(samples_df), desc="NEL Progress"):
        sample_id = row['id']
        sample_text = row['sample']

        # Prepare the prompt for NEL with few-shot examples
        prompt = (
            f"You are an AI assistant that performs Named Entity Linking (NEL). "
            f"Identify the entities in the following text and link them to the appropriate Wikidata QIDs:\n\n"
            f"Examples:\n"
            f"Text: \"ACFT WAS TAXIING FOR TAKE OFF WHEN IT LOST CONTROL, RAN INTO A DITCH, AND STRUCK A TREE. OTHER CIRCUMSTANCES AE UNK\"\n"
            f"Output:\n"
            f"Entity: ACFT\nQID: Q11436\n\n"
            f"Text: \"AFTER TAKEOFF, ENGINE QUIT. WING FUEL TANK SUMPS WERE NOT DRAINED DURING PREFLIGHT BECAUSE THEY WERE FROZEN.\"\n"
            f"Output:\n"
            f"Entity: TAKEOFF\nQID: Q854248\n"
            f"Entity: ENGINE\nQID: Q1\n\n"
            f"Now, identify the entities and link them to their respective Wikidata QIDs for the following text:\n\n"
            f"Text: \"{sample_text}\"\n"
            f"Output:"
        )

        # Call the OpenAI GPT-4o API
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0,
        )

        # Extract and process GPT-4o output
        gpt_output = response['choices'][0]['message']['content'].strip()
        try:
            # Regex to find entities and their QIDs in the output
            matches = re.findall(r'Entity:\s*(.+?)\s*QID:\s*(Q\d+)', gpt_output, re.IGNORECASE)
            entities_predicted = [match[0].strip() for match in matches]
            qids_predicted = [match[1].strip() for match in matches]

            results.append({
                'id': sample_id,
                'entities_predicted': entities_predicted,
                'qids_predicted': qids_predicted
            })
        except Exception as e:
            print(f"Error processing output for sample {sample_id}: {e}")
            results.append({
                'id': sample_id,
                'sample': sample_text,
                'entities_predicted': [],
                'qids_predicted': []
            })

    # Save results to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(results_dir, 'nel_results.csv'), index=False)
    print(f"NEL results saved to {os.path.join(results_dir, 'nel_results.csv')}")

    # Prepare gold standard data
    gold_standard = nel_df.groupby('id').agg({
        'entities': lambda x: [item for sublist in x for item in sublist],
        'qids': lambda x: [item for sublist in x for item in sublist]
    }).reset_index()
    comparison_df = pd.merge(results_df, gold_standard, on='id')

    print("Evaluating NEL...")

    # Evaluate performance
    total_TP = total_FP = total_FN = 0
    for _, row in tqdm(comparison_df.iterrows(), total=len(comparison_df), desc="NEL Evaluation Progress"):
        true_pairs = set(zip(map(str.lower, row['entities']), row['qids']))
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

def safe_literal_eval(val):
    try:
        return ast.literal_eval(val)
    except (ValueError, SyntaxError):
        return []  # Default value if parsing fails
        
# Updated function to perform CR using GPT-4o
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

        # Call the GPT-4o API
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0,
        )

        # Extract and process GPT-4o output
        gpt_output = response['choices'][0]['message']['content'].strip()
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

    # Save GPT-4 results to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(results_dir, 'cr_results.csv'), index=False)
    print(f"CR results saved to {os.path.join(results_dir, 'cr_results.csv')}")

    print("Evaluating CR...")

    # Evaluation logic
    total_TP = total_FP = total_FN = 0
    for _, row in tqdm(samples_df.iterrows(), total=len(samples_df), desc="CR Evaluation Progress"):
        sample_id = row['c5']
        sample_text = row['c119_text']

        # Retrieve gold coreferences
        gold_coreferences = cr_df[cr_df['c5'] == sample_id]['coreferences'].iloc[0]
        gold_coreferences_parsed = safe_literal_eval(gold_coreferences)

        # Retrieve predicted clusters
        predicted_row = next((item for item in results if item['c5'] == sample_id), {})
        predicted_coreferences = predicted_row.get('coreferences', [])

        # Calculate True Positives, False Positives, and False Negatives
        gold_pairs = set(tuple(pair) for cluster in gold_coreferences_parsed for pair in cluster)
        predicted_pairs = set(tuple(pair) for cluster in predicted_coreferences for pair in cluster)

        TP = len(gold_pairs & predicted_pairs)
        FP = len(predicted_pairs - gold_pairs)
        FN = len(gold_pairs - predicted_pairs)
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
    # Replace NaN values with an empty string to avoid AttributeError
    re_df['entity1, relation_type, entity2'] = re_df['entity1, relation_type, entity2'].fillna('')

    # Prepare entity-relation pairs
    re_df['relations'] = re_df['entity1, relation_type, entity2'].apply(lambda x: x.split('\n') if isinstance(x, str) else [])

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
            "Examples:\n"
            "Text: \"ACFT WAS TAXIING FOR TAKE OFF WHEN IT LOST CONTROL, RAN INTO A DITCH, AND STRUCK A TREE. OTHER CIRCUMSTANCES AE UNK\"\n"
            "Output:\n"
            "Subject: lost control\nSubject Type: action\nRelation: has effect\nRelation Type: causation\nObject: ran into a ditch\nObject Type: event\n"
            "Subject: lost control\nSubject Type: action\nRelation: has effect\nRelation Type: causation\nObject: struck a tree\nObject Type: event\n"
            "Remember always follow the format above and do not add any other information. Now, extract all relations from the following text:\n"
            f"Text: \"{sample_text}\"\n"
            "Output:")

        # Call the GPT-4o API
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0,
        )

        # Extract and process GPT-4o output
        gpt_output = response['choices'][0]['message']['content'].strip()
        try:
            # Extract and process GPT-4o output
            triple_matches = re.findall(
                r'Subject:\s*(.*?)\nSubject Type:\s*(.*?)\nRelation:\s*(.*?)\nRelation Type:\s*(.*?)\nObject:\s*(.*?)\nObject Type:\s*(.*?)(?:\n|$)',
                gpt_output, re.DOTALL
            )
            for match in triple_matches:
                results.append({
                    'c5_unique_id': sample_id,
                    'sample': sample_text,
                    'subject': match[0].strip(),
                    'subject_type': match[1].strip(),
                    'relation': match[2].strip(),
                    'relation_type': match[3].strip(),
                    'object': match[4].strip(),
                    'object_type': match[5].strip()
                })
        except Exception as e:
            print(f"Error processing output for sample {sample_id}: {e}")

    # Save results to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(results_dir, 're_results.csv'), index=False)
    print(f"RE results saved to {os.path.join(results_dir, 're_results.csv')}")

    print("Evaluating RE...")

    # Prepare gold standard data
    gold_standard = re_df.groupby('c5_unique_id').agg({
        'relations': lambda x: [item for sublist in x for item in sublist]
    }).reset_index()
    comparison_df = pd.merge(results_df, gold_standard, on='c5_unique_id', how='left')

    # Evaluate performance
    total_TP = total_FP = total_FN = 0
    for _, row in tqdm(comparison_df.iterrows(), total=len(comparison_df), desc="RE Evaluation Progress"):
        gold_triples = set(map(lambda t: tuple(map(str.lower, t.split(','))), row['relations']))
        pred_triples = set(
            (row['subject'].lower(), row['relation'].lower(), row['object'].lower())
            for _, row in results_df[results_df['c5_unique_id'] == row['c5_unique_id']].iterrows()
        )
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
