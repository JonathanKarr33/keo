import pandas as pd
from tqdm import tqdm

import re

def parse_gold_relations(relation_string):
    """
    Parse the gold standard relations from the string into structured triples.
    Handles inconsistent whitespace, single relations, and multiple relations.
    """
    relations = []
    #print(f"Raw relation string:\n{relation_string}\n")  # Debug: Print raw input

    if isinstance(relation_string, str):
        # Split by large spaces (more than 2 consecutive spaces) or newlines
        relation_lines = re.split(r'\s{2,}', relation_string.strip())

        for line in relation_lines:
            # Normalize each line
            line = " ".join(line.split())  # Collapse any remaining extra spaces

            # Split into parts by comma
            parts = [part.strip() for part in line.split(',')]

            if len(parts) == 3:
                relations.append((parts[0], parts[1], parts[2]))
            elif len(parts) == 1:
                # Handle single relation case
                single_parts = parts[0].split()
                if len(single_parts) == 3:
                    relations.append((single_parts[0], single_parts[1], single_parts[2]))

    # print(f"Extracted relations:\n{relations}\n")  # Debug: Final parsed relations
    return relations

def evaluate_re(gold_file, prediction_file, output_file):
    """
    Evaluate Relation Extraction (RE) results against the gold standard.

    Parameters:
    - gold_file (str): Path to the gold standard CSV file.
    - prediction_file (str): Path to the predictions CSV file.
    """
    # Load gold standard and predictions
    gold_df = pd.read_csv(gold_file)
    pred_df = pd.read_csv(prediction_file)

    total_TP = 0
    total_FP = 0
    total_FN = 0

    # Preprocess gold standard to extract relations
    gold_relations = {}
    for _, row in gold_df.iterrows():
        c5_id = row['c5_unique_id']
        gold_relations.setdefault(c5_id, []).extend(parse_gold_relations(row['entity1, relation_type, entity2']))
    

    # Iterate through predictions
    for _, pred_row in tqdm(pred_df.iterrows(), total=len(pred_df), desc="Evaluating Predictions"):
        pred_c5_id = pred_row['c5_unique_id']
        pred_subject = pred_row['subject'].strip()
        pred_relation = pred_row['relation'].strip()
        pred_object = pred_row['object'].strip()

        # Check against gold relations for the same c5 ID
        if pred_c5_id in gold_relations:
            matching_gold_relations = [
                (subj, rel, obj) for subj, rel, obj in gold_relations[pred_c5_id]
                if subj.lower() == pred_subject.lower() and rel.lower() == pred_relation.lower() and obj.lower() == pred_object.lower()
            ]

            if matching_gold_relations:
                total_TP += 1
            else:
                total_FP += 1
        else:
            total_FP += 1

    # Calculate FN
    for c5_id, gold_rels in gold_relations.items():
        for subj, rel, obj in gold_rels:
            matching_predictions = pred_df[
                (pred_df['c5_unique_id'] == c5_id) &
                (pred_df['subject'].str.lower() == subj.lower()) &
                (pred_df['relation'].str.lower() == rel.lower()) &
                (pred_df['object'].str.lower() == obj.lower())
            ]
            if matching_predictions.empty:
                total_FN += 1

    # Calculate metrics
    precision = total_TP / (total_TP + total_FP) if total_TP + total_FP > 0 else 0
    recall = total_TP / (total_TP + total_FN) if total_TP + total_FN > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

    # Print results
    result = f"RE Results: Precision={precision:.4f}, Recall={recall:.4f}, F1-score={f1:.4f}"
    print(result)

    # Write the results to the output file
    with open(output_file, 'w') as f:
        f.write(result)

if __name__ == "__main__":
    # File paths
    gold_file = "/home/kuangshiai/Desktop/24Fall-ND-Courses/LLM/keo/OMIn_dataset/gold_standard/raw/re.csv"  # Path to gold standard file
    prediction_file = "/home/kuangshiai/Desktop/24Fall-ND-Courses/LLM/keo/keo/vanilla_LLM/llama31_results/re_results.csv"  # Path to prediction file
    output_file = "keo/vanilla_LLM/llama31_results/re_score.txt"

    evaluate_re(gold_file, prediction_file, output_file)
