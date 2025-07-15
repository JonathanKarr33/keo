import os
import glob
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import re
from collections import defaultdict

def parse_triplets(triplet_string):
    if pd.isna(triplet_string) or not triplet_string.strip():
        return []
    relations = [
        'OWNED BY', 'INSTANCE OF', 'FOLLOWED BY', 'HAS CAUSE', 'FOLLOWS',
        'EVENT DISTANCE', 'HAS EFFECT', 'LOCATION', 'USED BY', 'INFLUENCED BY',
        'TIME PERIOD', 'PART OF', 'MAINTAINED BY', 'DESIGNED BY'
    ]
    rel_pattern = '|'.join([re.escape(r) for r in relations])
    # Regex: <entity1, RELATION, entity2> where RELATION is from the set
    pattern = re.compile(rf'<(.+?),\s*({rel_pattern}),\s*(.+?)>', re.IGNORECASE)
    triplets = []
    for match in pattern.finditer(triplet_string):
        e1, rel, e2 = match.groups()
        triplets.append((e1.strip(), rel.strip().upper(), e2.strip()))
    return triplets

def process_triplet_csv(csv_path):
    df = pd.read_csv(csv_path)
    G = nx.DiGraph()
    node_c5 = defaultdict(set)
    node_count = defaultdict(int)
    edge_c5 = defaultdict(set)
    edge_rel = defaultdict(set)
    edge_rows = defaultdict(set)
    edge_weight = defaultdict(int)
    for idx, row in df.iterrows():
        c5 = str(row.get('c5', ''))
        triplet_str = row.get('gpt4o_triplets_clean', '')
        triplets = parse_triplets(triplet_str)
        for e1, rel, e2 in triplets:
            G.add_node(e1)
            G.add_node(e2)
            G.add_edge(e1, e2)
            node_c5[e1].add(c5)
            node_c5[e2].add(c5)
            node_count[e1] += 1
            node_count[e2] += 1
            edge_c5[(e1, e2)].add(c5)
            edge_rel[(e1, e2)].add(rel)
            edge_rows[(e1, e2)].add(idx)
            edge_weight[(e1, e2)] += 1
    # Save GML
    base = os.path.splitext(csv_path)[0]
    gml_path = base + '.gml'
    nx.write_gml(G, gml_path)
    print(f"Saved GML: {gml_path}")
    # Save PNG
    png_path = base + '.png'
    plt.figure(figsize=(12, 12))
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, with_labels=True, node_size=300, font_size=8, node_color="lightblue", edge_color="gray", arrows=True)
    edge_labels = { (u, v): ';'.join(sorted(edge_rel[(u, v)])) for u, v in G.edges() }
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=6)
    plt.title(f"KG: {os.path.basename(base)}")
    plt.tight_layout()
    plt.savefig(png_path, format="PNG", dpi=1000)
    plt.close()
    print(f"Saved PNG: {png_path}")
    # Save nodes CSV
    nodes_csv = base + '_nodes.csv'
    with open(nodes_csv, 'w') as f:
        f.write('c5_sources,node,count\n')
        for node in G.nodes():
            c5s = ';'.join(sorted(node_c5[node]))
            count = node_count[node]
            f.write(f'{c5s},{node},{count}\n')
    print(f"Saved nodes CSV: {nodes_csv}")
    # Save edges CSV
    edges_csv = base + '_edges.csv'
    with open(edges_csv, 'w') as f:
        f.write('c5_sources,entity1,entity2,weight,relations,rows\n')
        for u, v in G.edges():
            c5s = ';'.join(sorted(edge_c5[(u, v)]))
            weight = edge_weight[(u, v)]
            rels = ';'.join(sorted(edge_rel[(u, v)]))
            rows = ','.join(str(x) for x in sorted(edge_rows[(u, v)]))
            f.write(f'{c5s},{u},{v},{weight},{rels},{rows}\n')
    print(f"Saved edges CSV: {edges_csv}")
    # Save summary
    summary_txt = base + '_summary.txt'
    with open(summary_txt, 'w') as f:
        f.write(f'File: {csv_path}\n')
        f.write(f'Number of nodes: {G.number_of_nodes()}\n')
        f.write(f'Number of edges: {G.number_of_edges()}\n')
        f.write(f'Total rows used: {len(df)}\n')
    print(f"Saved summary: {summary_txt}")

def main():
    root = os.path.join('output', 'kg_llm')
    pattern = os.path.join(root, '**', '*_with_entity_mentions_fixed.csv')
    files = glob.glob(pattern, recursive=True)
    print(f"Found {len(files)} files.")
    for f in files:
        print(f"Processing: {f}")
        process_triplet_csv(f)

if __name__ == '__main__':
    main()