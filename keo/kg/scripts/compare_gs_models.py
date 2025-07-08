import pandas as pd
import re
from collections import Counter
import argparse
from difflib import SequenceMatcher

def parse_triplets(triplet_string):
    """Parse triplet string into list of (entity1, relation, entity2) tuples"""
    if pd.isna(triplet_string) or triplet_string == "":
        return []
    
    # Split by semicolon and clean up
    triplets = []
    for triplet in triplet_string.split(';'):
        triplet = triplet.strip()
        if not triplet:
            continue
            
        # Extract entities and relation using regex
        # Pattern: <entity1, relation, entity2> or entity1, relation, entity2
        match = re.search(r'<?([^,]+),\s*([^,]+),\s*([^>]+)>?', triplet)
        if match:
            entity1 = match.group(1).strip()
            relation = match.group(2).strip()
            entity2 = match.group(3).strip()
            triplets.append((entity1, relation, entity2))
    
    return triplets

def compute_prf1(pred, gold):
    pred_set = set(pred)
    gold_set = set(gold)
    tp = len(pred_set & gold_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1, len(gold_set), len(pred_set)

def soft_entity_match(e1, e2, threshold=0.8):
    return SequenceMatcher(None, e1.lower(), e2.lower()).ratio() >= threshold

def soft_triplet_match(pred_triplet, gold_triplet):
    # Relation must match exactly, entities must be similar
    return (
        pred_triplet[1].strip().lower() == gold_triplet[1].strip().lower() and
        soft_entity_match(pred_triplet[0], gold_triplet[0]) and
        soft_entity_match(pred_triplet[2], gold_triplet[2])
    )

def per_component_prf1(pred, gold, idx):
    pred_set = set([t[idx] for t in pred])
    gold_set = set([t[idx] for t in gold])
    tp = len(pred_set & gold_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1

def macro_f1(pred_rows, gold_rows):
    # pred_rows and gold_rows are lists of lists of triplets
    f1s = []
    for pred, gold in zip(pred_rows, gold_rows):
        _, _, f1, _, _ = compute_prf1(pred, gold)
        f1s.append(f1)
    return sum(f1s) / len(f1s) if f1s else 0.0

def compute_soft_f1(pred, gold):
    matched_pred = set()
    matched_gold = set()
    for i, p in enumerate(pred):
        for j, g in enumerate(gold):
            if j not in matched_gold and soft_triplet_match(p, g):
                matched_pred.add(i)
                matched_gold.add(j)
                break
    tp = len(matched_pred)
    fp = len(pred) - tp
    fn = len(gold) - tp
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1, fn, fp

def compare_annotations(model_output_file):
    """Compare strict vs loose gold standards and model outputs using clean triplets columns"""
    import os
    os.makedirs('output', exist_ok=True)
    # Load data
    strict_gs = pd.read_csv('output/re_gs_strict.csv')
    loose_gs = pd.read_csv('output/re_gs_loose.csv')
    model_output = pd.read_csv(model_output_file)
    
    # Parse triplets per row for macro F1 and error analysis
    strict_gs_rows = [parse_triplets(x) for x in strict_gs['entity1, relation_type, entity2']]
    loose_gs_rows = [parse_triplets(x) for x in loose_gs['entity1, relation_type, entity2']]
    model_triplets_rows = {col: [parse_triplets(x) for x in model_output[col]] for col in model_output.columns if col.endswith('_triplets_clean')}
    
    # Parse triplets (micro, pooled)
    strict_triplets = [t for row in strict_gs_rows for t in row]
    loose_triplets = [t for row in loose_gs_rows for t in row]
    model_triplets = {col: [t for row in rows for t in row] for col, rows in model_triplets_rows.items()}
    
    # Prepare detailed stats
    detailed_rows = []
    for gs_type, gold, gold_rows in [('strict', strict_triplets, strict_gs_rows), ('loose', loose_triplets, loose_gs_rows)]:
        for col, pred in model_triplets.items():
            model_name = col.replace('_triplets_clean','')
            pred_rows = model_triplets_rows[col]
            # Micro F1
            precision, recall, f1, support_gold, support_pred = compute_prf1(pred, gold)
            # Macro F1
            macro = macro_f1(pred_rows, gold_rows)
            # Per-component F1
            p_e1, r_e1, f1_e1 = per_component_prf1(pred, gold, 0)
            p_rel, r_rel, f1_rel = per_component_prf1(pred, gold, 1)
            p_e2, r_e2, f1_e2 = per_component_prf1(pred, gold, 2)
            # Soft F1
            soft_p, soft_r, soft_f1, fn_soft, fp_soft = compute_soft_f1(pred, gold)
            # Error analysis (micro, strict)
            fp = [t for t in pred if t not in gold]
            fn = [t for t in gold if t not in pred]
            # Error analysis (micro, soft)
            soft_fp = [t for i, t in enumerate(pred) if not any(soft_triplet_match(t, g) for g in gold)]
            soft_fn = [t for j, t in enumerate(gold) if not any(soft_triplet_match(p, gold[j]) for p in pred)]
            # Save stats (do not save error lists)
            detailed_rows.append({
                'model': model_name,
                'gs_type': gs_type,
                'precision_micro': precision,
                'recall_micro': recall,
                'f1_micro': f1,
                'precision_macro': macro,
                'recall_macro': macro, # macro F1 is symmetric, for F1 only
                'f1_macro': macro,
                'precision_entity1': p_e1,
                'recall_entity1': r_e1,
                'f1_entity1': f1_e1,
                'precision_relation': p_rel,
                'recall_relation': r_rel,
                'f1_relation': f1_rel,
                'precision_entity2': p_e2,
                'recall_entity2': r_e2,
                'f1_entity2': f1_e2,
                'precision_soft': soft_p,
                'recall_soft': soft_r,
                'f1_soft': soft_f1,
                'support_gold': support_gold,
                'support_pred': support_pred,
                'false_negatives': len(fn),
                'false_positives': len(fp),
                'soft_false_negatives': len(soft_fn),
                'soft_false_positives': len(soft_fp)
            })
    detailed_df = pd.DataFrame(detailed_rows)
    detailed_df.to_csv('output/compare_gs_detailed_stats.csv', index=False)
    print('\nDetailed stats saved to output/compare_gs_detailed_stats.csv')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare gold standard and model triplet outputs.")
    parser.add_argument('--model_output', type=str, default='output/100_kg_llm_triplets_gemma3_phi4mini.csv', help='Model output CSV file to compare (default: clean 100-row LLM triplets)')
    args = parser.parse_args()
    compare_annotations(args.model_output) 