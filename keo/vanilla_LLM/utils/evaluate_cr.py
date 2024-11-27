import pandas as pd
import ast

def safe_literal_eval(value):
    try:
        return ast.literal_eval(value)
    except Exception:
        return []

def evaluate_cr(ground_truth_file, predictions_file):
    """
    Evaluate Coreference Resolution (CR) results.

    Parameters:
    - ground_truth_file (str): Path to the ground truth CSV file.
    - predictions_file (str): Path to the predictions CSV file.
    """
    gt_df = pd.read_csv(ground_truth_file)
    pred_df = pd.read_csv(predictions_file)

    total_TP = total_FP = total_FN = 0

    for _, row in gt_df.iterrows():
        sample_id = row['c5']
        gold_coreferences = safe_literal_eval(row['coreferences'])

        predicted_row = pred_df[pred_df['c5'] == sample_id]
        if predicted_row.empty:
            predicted_coreferences = []
        else:
            predicted_coreferences = safe_literal_eval(predicted_row.iloc[0]['coreferences'])

        gold_pairs = set(tuple(pair) for cluster in gold_coreferences for pair in cluster)
        predicted_pairs = set(tuple(pair) for cluster in predicted_coreferences for pair in cluster)

        TP = len(gold_pairs & predicted_pairs)
        FP = len(predicted_pairs - gold_pairs)
        FN = len(gold_pairs - predicted_pairs)
        total_TP += TP
        total_FP += FP
        total_FN += FN

    precision = total_TP / (total_TP + total_FP) if total_TP + total_FP else 0
    recall = total_TP / (total_TP + total_FN) if total_TP + total_FN else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0

    print(f"CR Results: Precision={precision:.4f}, Recall={recall:.4f}, F1-score={f1:.4f}")

if __name__ == "__main__":
    # Test the script with example files
    ground_truth_file = "/home/kuangshiai/Desktop/24Fall-ND-Courses/LLM/keo/OMIn_dataset/gold_standard/raw/cr.csv"
    predictions_file = "keo/vanilla_LLM/gpt4_results/cr_results.csv"

    evaluate_cr(ground_truth_file, predictions_file)