import pandas as pd
import networkx as nx
import os
import matplotlib.pyplot as plt
import openai
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

# Parse triplet string into tuples (entity1, relation, entity2)
def parse_triplets(triplet_string):
    if pd.isna(triplet_string) or not triplet_string.strip():
        return []
    triplets = []
    for triplet in triplet_string.split('\n'):
        triplet = triplet.strip()
        if triplet.startswith('<') and triplet.endswith('>'):
            parts = [p.strip() for p in triplet[1:-1].split(',')]
            if len(parts) == 3:
                triplets.append(tuple(parts))
    return triplets

def build_kg(input_csv, model, subset=None, start=0, output_prefix='../output/kg_llm', n_gold_standard_matched=None):
    """
    input_csv: path to the LLM triplet CSV (must have c5 column)
    model: model shortname (e.g., 'gemma3_4b_it', 'gpt4o')
    subset: number of rows to use (None for all)
    start: row index to start from (0 means start at the first row)
    output_prefix: prefix for output files
    n_gold_standard_matched: if set, use only this many rows that match the gold standard (by c5)
    """
    # Handle GPT-4o model differently
    if model == 'gpt4o':
        col = f"{model}_triplets_clean"
    else:
        col = f"{model}_triplets_clean"
    
    df = pd.read_csv(input_csv)

    original_total_rows = len(df)  # Save before filtering

    # Gold standard matching (by c5)
    matched_c5 = []
    unmatched_c5 = []
    gs_matched_count = None
    unique_gs_matches = None
    gs_folder = None  # Will be used for output folder naming
    if n_gold_standard_matched is not None:
        gold_standard_csv = '../../OMIn_dataset/data/FAA_data/FAA_sample_100.csv'
        gs_df = pd.read_csv(gold_standard_csv)
        gs_c5_set = set(gs_df['c5'])
        df['is_gs'] = df['c5'].isin(gs_c5_set)
        matched = df[df['is_gs']]
        unmatched = df[~df['is_gs']]
        # Select up to n_gold_standard_matched GS rows
        matched = matched.iloc[start:start+n_gold_standard_matched] if n_gold_standard_matched is not None else matched.iloc[start:]
        gs_matched_count = len(matched)
        unique_gs_matches = matched['c5'].nunique()
        matched_c5 = matched['c5'].tolist()
        unmatched_c5 = unmatched['c5'].tolist()
        # If subset is set, fill up to subset with non-GS rows
        if subset is not None:
            n_needed = max(0, subset - len(matched))
            non_gs_rows = unmatched.iloc[start:start+n_needed]
            df = pd.concat([matched, non_gs_rows], axis=0)
            filtered_total_rows = len(df)
        else:
            df = matched.copy()
            filtered_total_rows = len(df)
    elif subset is not None:
        df = df.iloc[start:start+subset]
        filtered_total_rows = len(df)
    else:
        df = df.iloc[start:]
        filtered_total_rows = len(df)

    # Set output folder with gold standard and total rows info
    if n_gold_standard_matched is not None:
        if subset is not None:
            gs_folder = f"gs{gs_matched_count}_total{filtered_total_rows}"
        else:
            gs_folder = f"gs{gs_matched_count}_total{original_total_rows}"
    else:
        gs_folder = f"total{filtered_total_rows}"
    model_folder = os.path.join(output_prefix, gs_folder, model)
    os.makedirs(model_folder, exist_ok=True)

    # Build the knowledge graph
    G = nx.DiGraph()
    for idx, row in df.iterrows():
        triplet_str = row.get(col, '')
        triplets = parse_triplets(triplet_str)
        c5_value = row.get('c5', '')
        for (e1, rel, e2) in triplets:
            if not G.has_node(e1):
                G.add_node(e1, count=0, c5_sources=set())
            if not G.has_node(e2):
                G.add_node(e2, count=0, c5_sources=set())
            G.nodes[e1]['count'] += 1
            G.nodes[e2]['count'] += 1
            G.nodes[e1]['c5_sources'].add(c5_value)
            G.nodes[e2]['c5_sources'].add(c5_value)
            if G.has_edge(e1, e2):
                G[e1][e2]['weight'] += 1
                G[e1][e2]['relations'].add(rel)
                G[e1][e2]['rows'].add(idx)
                G[e1][e2]['c5_sources'].add(c5_value)
            else:
                G.add_edge(e1, e2, weight=1, relations={rel}, rows={idx}, c5_sources={c5_value})

    # Convert c5_sources, relations, and rows from set/int to string for GML compatibility
    for n, d in G.nodes(data=True):
        if isinstance(d.get('c5_sources', None), set):
            d['c5_sources'] = ';'.join(sorted(d['c5_sources']))
    for u, v, data in G.edges(data=True):
        if isinstance(data.get('c5_sources', None), set):
            data['c5_sources'] = ';'.join(sorted(data['c5_sources']))
        if isinstance(data.get('relations', None), set):
            data['relations'] = ';'.join(sorted(data['relations']))
        rows_val = data.get('rows', None)
        if isinstance(rows_val, set):
            data['rows'] = ','.join(str(x) for x in sorted(rows_val))
        elif isinstance(rows_val, int):
            data['rows'] = str(rows_val)
        elif isinstance(rows_val, list):
            data['rows'] = ','.join(str(x) for x in rows_val)

    # Output files
    gml_path = os.path.join(model_folder, f"kg_{model}.gml")
    nx.write_gml(G, gml_path)
    print(f"KG saved to {gml_path}")

    nodes_df = pd.DataFrame([
        {'c5_sources': d.get('c5_sources', ''), 'node': n, 'count': d.get('count', 0)} for n, d in G.nodes(data=True)
    ])
    nodes_csv = os.path.join(model_folder, f"kg_{model}_nodes.csv")
    nodes_df.to_csv(nodes_csv, index=False)
    print(f"Nodes saved to {nodes_csv}")

    edges_df = pd.DataFrame([
        {'c5_sources': d.get('c5_sources', ''), 'entity1': u, 'entity2': v, 'weight': d['weight'], 'relations': d['relations'], 'rows': d['rows']} for u, v, d in G.edges(data=True)
    ])
    edges_csv = os.path.join(model_folder, f"kg_{model}_edges.csv")
    edges_df.to_csv(edges_csv, index=False)
    print(f"Edges saved to {edges_csv}")

    # PNG visualization
    png_path = os.path.join(model_folder, f"kg_{model}.png")
    plt.figure(figsize=(12, 12))
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, with_labels=True, node_size=300, font_size=8, node_color="lightblue", edge_color="gray", arrows=True)
    plt.title(f"Knowledge Graph: {model}")
    plt.tight_layout()
    plt.savefig(png_path, format="PNG", dpi=1000)  # Increased DPI for higher resolution
    plt.close()
    print(f"Graph visualization saved to {png_path}")

    # Output summary info
    summary_lines = []
    if n_gold_standard_matched is not None:
        summary_lines.append(f"Rows matched to gold standard: {gs_matched_count}")
        summary_lines.append(f"Unique GS matches in filtered LLM: {unique_gs_matches}")
        if n_gold_standard_matched == 100 and unique_gs_matches != 100:
            summary_lines.append(f"WARNING: Only {unique_gs_matches} unique GS matches found, expected 100!")
    summary_lines.append(f"Total rows used for KG: {filtered_total_rows}")
    summary_lines.append(f"Number of nodes in KG: {G.number_of_nodes()}")
    summary_lines.append(f"Number of edges in KG: {G.number_of_edges()}")
    summary_txt = os.path.join(model_folder, f"kg_{model}_summary.txt")
    with open(summary_txt, 'w') as f:
        for line in summary_lines:
            print(line)
            f.write(line + '\n')
    print(f"Summary saved to {summary_txt}")


# =====================
# CONFIGURATION SECTION
# =====================
input_csv = 'output/100_kg_llm_triplets_gpt4o.csv'  # Path to the LLM triplet CSV
model = 'gpt4o'  # Model shortname (e.g., 'gemma3_4b_it', 'phi4mini_instruct', 'gpt4o')
subset = 10  # Number of rows to use (None for all, or e.g., 100)
start = 0  # Row index to start from (0 means start at the first row)
output_prefix = 'output/kg_llm'  # Prefix for output files (subfolder for each model will be created)
n_gold_standard_matched = 10  # Set to an integer to use only that many rows that match the gold standard (None-100)

if __name__ == "__main__":
    # Check for GPT-4o model and verify OpenAI API key
    if model == 'gpt4o':
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set. Please set it with your OpenAI API key.")
        print("✓ OpenAI API key verified for GPT-4o")
    
    os.makedirs(output_prefix, exist_ok=True)
    build_kg(input_csv, model, subset, start, output_prefix, n_gold_standard_matched)
    print("\n=== c5-based matching complete ===")
    print("✓ All output files include c5_sources tracking") 