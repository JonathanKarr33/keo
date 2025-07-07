import pandas as pd
import re
from collections import Counter
import argparse

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

def compare_annotations(model_output_file):
    """Compare strict vs loose gold standards and model outputs using clean triplets columns"""
    
    # Load data
    strict_gs = pd.read_csv('output/re_gs_strict.csv')
    loose_gs = pd.read_csv('output/re_gs_loose.csv')
    model_output = pd.read_csv(model_output_file)
    
    print("=== GOLD STANDARD COMPARISON ===")
    print(f"Strict GS rows with annotations: {strict_gs['entity1, relation_type, entity2'].notna().sum()}/{len(strict_gs)}")
    print(f"Loose GS rows with annotations: {loose_gs['entity1, relation_type, entity2'].notna().sum()}/{len(loose_gs)}")
    
    # Parse triplets
    strict_triplets = []
    loose_triplets = []
    # Only use columns ending with _triplets_clean
    model_triplets = {col: [] for col in model_output.columns if col.endswith('_triplets_clean')}
    
    for idx, row in strict_gs.iterrows():
        # Parse strict GS
        strict_parsed = parse_triplets(row['entity1, relation_type, entity2'])
        strict_triplets.extend(strict_parsed)
        
        # Parse loose GS
        loose_parsed = parse_triplets(loose_gs.iloc[idx]['entity1, relation_type, entity2'])
        loose_triplets.extend(loose_parsed)
        
        # Parse model outputs
        if idx < len(model_output):
            for col in model_triplets:
                parsed = parse_triplets(model_output.iloc[idx][col])
                model_triplets[col].extend(parsed)
    
    print(f"\n=== TRIPLET COUNTS ===")
    print(f"Strict GS triplets: {len(strict_triplets)}")
    print(f"Loose GS triplets: {len(loose_triplets)}")
    for col in model_triplets:
        print(f"{col.replace('_triplets_clean','').replace('_',' ').title()} triplets: {len(model_triplets[col])}")
    
    # Analyze relation types
    def get_relation_types(triplets):
        return Counter([t[1] for t in triplets])
    
    print(f"\n=== RELATION TYPE ANALYSIS ===")
    print("Strict GS relations:", dict(get_relation_types(strict_triplets)))
    print("Loose GS relations:", dict(get_relation_types(loose_triplets)))
    for col in model_triplets:
        print(f"{col.replace('_triplets_clean','').replace('_',' ').title()} relations:", dict(get_relation_types(model_triplets[col])))
    
    # Sample comparison for first few incidents
    print(f"\n=== SAMPLE COMPARISON (First 3 incidents) ===")
    for i in range(min(3, len(strict_gs))):
        print(f"\nIncident {i+1}: {strict_gs.iloc[i]['c5_unique_id']}")
        print(f"Text: {strict_gs.iloc[i]['c119_text'][:100]}...")
        
        strict_sample = parse_triplets(strict_gs.iloc[i]['entity1, relation_type, entity2'])
        loose_sample = parse_triplets(loose_gs.iloc[i]['entity1, relation_type, entity2'])
        
        model_samples = {}
        if i < len(model_output):
            for col in model_triplets:
                model_samples[col] = parse_triplets(model_output.iloc[i][col])
        
        print(f"  Strict GS: {len(strict_sample)} triplets")
        print(f"  Loose GS: {len(loose_sample)} triplets")
        for col in model_triplets:
            print(f"  {col.replace('_triplets_clean','').replace('_',' ').title()}: {len(model_samples.get(col, []))} triplets")
        
        if strict_sample:
            print(f"  Strict sample: {strict_sample[0] if strict_sample else 'None'}")
        if loose_sample:
            print(f"  Loose sample: {loose_sample[0] if loose_sample else 'None'}")
        for col in model_triplets:
            if model_samples.get(col):
                print(f"  {col.replace('_triplets_clean','').replace('_',' ').title()} sample: {model_samples[col][0]}")

    # Compute and save F1 summary
    summary_rows = []
    for gs_type, gold in [('strict', strict_triplets), ('loose', loose_triplets)]:
        for col, pred in model_triplets.items():
            model_name = col.replace('_triplets_clean','')
            precision, recall, f1, support_gold, support_pred = compute_prf1(pred, gold)
            summary_rows.append({
                'model': model_name,
                'gs_type': gs_type,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'support_gold': support_gold,
                'support_pred': support_pred
            })
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv('output/compare_gs_f1_summary.csv', index=False)
    print('\nF1 summary saved to output/compare_gs_f1_summary.csv')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare gold standard and model triplet outputs.")
    parser.add_argument('--model_output', type=str, default='output/100_kg_llm_triplets_gemma3_phi4mini.csv', help='Model output CSV file to compare (default: clean 100-row LLM triplets)')
    args = parser.parse_args()
    compare_annotations(args.model_output) 