import os
import csv
import argparse
import re
from collections import defaultdict, Counter
from sentence_transformers import SentenceTransformer, util
import numpy as np
import spacy

# --- SBERT-based entity normalization ---

def extract_entities_from_triplets(triplet_str):
    # Returns a set of all entities in a triplet string
    entities = set()
    for match in re.findall(r'<([^,]+),\s*([^,]+),\s*([^>]+)>', triplet_str or ""):
        e1, _, e2 = match
        entities.add(e1.strip())
        entities.add(e2.strip())
    return entities

def remove_stopwords(text, stopwords):
    return ' '.join([w for w in text.split() if w.lower() not in stopwords])

def process_csv_sbert(input_csv, output_csv, replacements_csv, model, sim_threshold=0.7, spacy_stopwords=None):
    with open(input_csv, newline='', encoding='utf-8') as fin:
        reader = csv.DictReader(fin)
        fieldnames = reader.fieldnames
        rows = list(reader)
    # Only process columns ending with '_triplets_clean'
    triplet_cols = [col for col in fieldnames if col.endswith('_triplets_clean')]
    # Also keep non-triplet columns (e.g., c5, c119) for output
    non_triplet_cols = [col for col in fieldnames if not col.endswith('_triplets') and not col.endswith('_triplets_clean')]
    output_fieldnames = non_triplet_cols + triplet_cols
    # Step 1: Extract all unique entity mentions from all triplets
    all_entities = set()
    for row in rows:
        for col in triplet_cols:
            if row[col]:
                all_entities.update(extract_entities_from_triplets(row[col]))
    all_entities = sorted(all_entities)
    print(f"Total unique entity mentions found: {len(all_entities)}")
    # Always create stopword-removed versions
    all_entities_nostop = [remove_stopwords(e, spacy_stopwords) for e in all_entities]
    # Step 2: Encode all canonical nodes (entities)
    entity_embeddings = model.encode(all_entities, convert_to_tensor=True, show_progress_bar=False)
    entity_embeddings_nostop = model.encode(all_entities_nostop, convert_to_tensor=True, show_progress_bar=False)
    replacements = []
    def replace_entity(entity):
        candidates = [(entity, entity_embeddings, all_entities)]
        entity_nostop = remove_stopwords(entity, spacy_stopwords)
        candidates.append((entity_nostop, entity_embeddings_nostop, all_entities))
        best_node = entity
        best_sim = 0
        for ent_text, emb_matrix, ent_list in candidates:
            entity_emb = model.encode(ent_text, convert_to_tensor=True)
            sims = util.cos_sim(entity_emb, emb_matrix)[0].cpu().numpy()
            best_idx = int(np.argmax(sims))
            sim = sims[best_idx]
            node = ent_list[best_idx]
            if sim > sim_threshold and node != entity:
                if sim > best_sim:
                    best_node = node
                    best_sim = sim
        if best_node != entity:
            return best_node, best_sim
        return entity, None
    def fix_triplet_string_sbert(triplet_str, row_idx, col):
        def fix_triplet(match):
            e1, rel, e2 = match.group(1), match.group(2), match.group(3)
            e1_fixed, sim1 = replace_entity(e1.strip())
            e2_fixed, sim2 = replace_entity(e2.strip())
            if e1 != e1_fixed:
                replacements.append({'existing_node': e1, 'match': e1_fixed, 'new_row': row_idx, 'final_node': e1_fixed, 'similarity': sim1})
            if e2 != e2_fixed:
                replacements.append({'existing_node': e2, 'match': e2_fixed, 'new_row': row_idx, 'final_node': e2_fixed, 'similarity': sim2})
            return f"<{e1_fixed}, {rel.strip().upper()}, {e2_fixed}>"
        return re.sub(r'<([^,]+),\s*([^,]+),\s*([^>]+)>', fix_triplet, triplet_str or "")
    for i, row in enumerate(rows):
        for col in triplet_cols:
            if row[col]:
                row[col] = fix_triplet_string_sbert(row[col], i, col)
    # Only output non-triplet and clean triplet columns
    with open(output_csv, 'w', newline='', encoding='utf-8') as fout:
        writer = csv.DictWriter(fout, fieldnames=output_fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row[col] for col in output_fieldnames})
    print(f"Wrote: {output_csv}")
    # Step 6: Write replacements mapping
    if replacements:
        with open(replacements_csv, 'w', newline='', encoding='utf-8') as fout:
            writer = csv.DictWriter(fout, fieldnames=['existing_node', 'match', 'new_row', 'final_node', 'similarity'])
            writer.writeheader()
            for r in replacements:
                writer.writerow(r)
        print(f"Wrote: {replacements_csv} ({len(replacements)} replacements)")
    else:
        print("No entity replacements were made.")

def find_csvs_to_fix(root_dir):
    files = []
    for dirpath, _, filenames in os.walk(root_dir):
        parent = os.path.basename(os.path.dirname(dirpath))
        # Accept any parent directory ending with _with_nodes_batches
        if not parent.endswith("_with_nodes_batches"):
            continue
        for f in filenames:
            # Accept any file ending with _withprevnodes_<batch>.csv or starting with llm_with_existing_nodes_
            if (
                f.endswith('.csv')
                and (
                    '_withprevnodes_' in f
                    or f.startswith('llm_with_existing_nodes_')
                )
                and not f.endswith('_with_entity_mentions_fixed.csv')
            ):
                fixed_name = f.replace('.csv', '_with_entity_mentions_fixed.csv')
                if fixed_name not in filenames:
                    files.append(os.path.join(dirpath, f))
    return files

def main():
    parser = argparse.ArgumentParser(
        description="""
        Fix entity mentions in triplet columns of batch CSVs using SBERT semantic similarity.
        Processes all CSVs in output/kg_llm/ subdirs whose parent ends with 'with_node_batches' or 'with_nodes_batches', skipping any already fixed.
        Outputs both a fixed CSV and a CSV mapping all replacements for each input file.
        """
    )
    parser.add_argument('--input', default='output/kg_llm/', help='Root directory to search for batch CSVs (default: output/kg_llm/)')
    parser.add_argument('--output-dir', default=None, help='Directory to write fixed CSVs (default: same as input)')
    parser.add_argument('--sbert-model', default='all-MiniLM-L6-v2', help='SBERT model to use (default: all-MiniLM-L6-v2)')
    parser.add_argument('--sim-threshold', type=float, default=0.7, help='Similarity threshold for replacement (default: 0.7)')
    parser.add_argument('--force', action='store_true', help='Force overwrite and reprocess files even if _with_entity_mentions_fixed.csv exists')
    args = parser.parse_args()

    model = SentenceTransformer(args.sbert_model)
    nlp = spacy.blank('en')
    spacy_stopwords = nlp.Defaults.stop_words

    if os.path.isdir(args.input):
        files = find_csvs_to_fix(args.input)
    else:
        files = [args.input]

    for f in files:
        base = os.path.basename(f)
        out_dir = args.output_dir or os.path.dirname(f)
        # Shorten output filenames
        out_path = os.path.join(out_dir, base.replace('.csv', '_fixed.csv'))
        replacements_path = os.path.join(out_dir, base.replace('.csv', '_replacements.csv'))
        # Skip if already processed, unless --force is set
        if os.path.exists(out_path) and not args.force:
            print(f"Skipping {f} (already fixed)")
            continue
        process_csv_sbert(f, out_path, replacements_path, model, sim_threshold=args.sim_threshold, spacy_stopwords=spacy_stopwords)

if __name__ == "__main__":
    main() 