import pandas as pd
import ast

def evaluate_ner(ground_truth_file, predictions_file):
    """
    Evaluate NER results using ground truth and predictions CSV files.

    Parameters:
    - ground_truth_file (str): Path to the ground truth CSV file.
    - predictions_file (str): Path to the predictions CSV file.
    """
    # Load the ground truth and predictions data
    ner_df = pd.read_csv(ground_truth_file)
    results = pd.read_csv(predictions_file)

    # Prepare the ground truth data
    gold_standard = ner_df.groupby('c5_unique_id').agg({
        'GS': lambda x: [entity for sublist in x for entity in ast.literal_eval(sublist)],
        'GS TYPE': lambda x: [etype for sublist in x for etype in ast.literal_eval(sublist)]
    }).reset_index().rename(columns={'c5_unique_id': 'id', 'GS': 'entities', 'GS TYPE': 'GS TYPE'})

    # Merge predictions with the ground truth
    comparison_df = pd.merge(results, gold_standard, on='id', how='left')

    # Initialize counters for entities and entity types
    total_TP_entities = total_FP_entities = total_FN_entities = 0
    total_TP_types = total_FP_types = total_FN_types = 0

    # Calculate true positives, false positives, and false negatives
    for _, row in comparison_df.iterrows():
        # Entities
        true_entities = set(map(str.lower, row['entities']))
        pred_entities = set(map(str.lower, ast.literal_eval(row['entities_predicted'])))
        TP_entities = len(true_entities & pred_entities)
        FP_entities = len(pred_entities - true_entities)
        FN_entities = len(true_entities - pred_entities)
        total_TP_entities += TP_entities
        total_FP_entities += FP_entities
        total_FN_entities += FN_entities

        # Entity types
        true_types = set(map(str.lower, row['GS TYPE']))
        pred_types = set(map(str.lower, ast.literal_eval(row['entity_types_predicted'])))
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

if __name__ == "__main__":
    # Test the script with example files
    ground_truth_file = "/home/kuangshiai/Desktop/24Fall-ND-Courses/LLM/keo/OMIn_dataset/gold_standard/raw/ner.csv"
    predictions_file = "keo/vanilla_LLM/gpt4_results/ner_results.csv"

    evaluate_ner(ground_truth_file, predictions_file)