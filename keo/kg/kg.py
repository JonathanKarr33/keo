import pandas as pd
import networkx as nx
import ast
import matplotlib.pyplot as plt

# Load the CSV files
base_path = "../../OMIn_dataset/gold_standard/raw"
ner_path = f"{base_path}/ner.csv"
cr_path = f"{base_path}/cr.csv"
nel_path = f"{base_path}/nel.csv"
re_paths = {
    "strict": "re_gs_strict.csv",
    "loose": "re_gs_loose.csv"
}

# Read the CSVs into pandas DataFrames
ner_df = pd.read_csv(ner_path, delimiter=',', encoding='utf-8')
cr_df = pd.read_csv(cr_path, delimiter=',', encoding='utf-8')
nel_df = pd.read_csv(nel_path, delimiter=',', encoding='utf-8')

# Dictionary to store relationship counts for strict and loose graphs
relationship_counts = {"strict": {}, "loose": {}}

# Process for both strict and loose graphs
for strict_re_gs, re_path in re_paths.items():
    print(f"Processing {strict_re_gs.upper()} Knowledge Graph...")

    re_df = pd.read_csv(re_path, delimiter=',', encoding='utf-8')

    # Create a directed graph
    G = nx.DiGraph()

    # Step 1: Process Entities and Types from ner_df for nodes
    for _, row in ner_df.iterrows():
        incident_id = row['c5_unique_id']
        entities = ast.literal_eval(row['GS'])
        types = ast.literal_eval(row['GS TYPE'])
        
        if len(entities) == len(types):
            for entity, entity_type in zip(entities, types):
                if entity:
                    if G.has_node(entity):
                        # Ensure incident_ids is a list before appending
                        if 'incident_ids' not in G.nodes[entity]:
                            G.nodes[entity]['incident_ids'] = []  # Initialize incident_ids as an empty list
                        G.nodes[entity]['incident_ids'].append(str(incident_id))
                    else:
                        G.add_node(entity, type=entity_type, incident_ids=[str(incident_id)])

    # Step 2: Add coreference relationships from cr_df
    for _, row in cr_df.iterrows():
        incident_id = row['c5']
        coreferences = row['coreferences_human_readable']
        
        if pd.notnull(coreferences) and coreferences:
            coreferences = coreferences.strip()[1:-1].split(', ')
            for i in range(len(coreferences)):
                for j in range(i + 1, len(coreferences)):
                    entity1, entity2 = coreferences[i].strip(), coreferences[j].strip()
                    if not G.has_edge(entity1, entity2):
                        G.add_edge(entity1, entity2, relation='coreference', incident_ids=[str(incident_id)])
                    else:
                        # Ensure incident_ids is a list before appending
                        if 'incident_ids' not in G[entity1][entity2]:
                            G[entity1][entity2]['incident_ids'] = []  # Initialize as an empty list
                        G[entity1][entity2]['incident_ids'].append(str(incident_id))

    # Step 3: Add NEL and its QIDs from nel.csv
    for _, row in nel_df.iterrows():
        incident_id = row['id']
        primary_ent = row['primary_ent']
        primary_qid = str(row['primary_qid']).strip()
        secondary_ent = str(row['secondary_ent']).strip()
        secondary_qid = str(row['secondary_qid']).strip()
        tertiary_ent = str(row['tertiary_ent']).strip()
        tertiary_qid = str(row['tertiary_qid']).strip()

        # Handle 'nan' values by setting them to None
        if primary_qid.lower() == 'nan':
            primary_qid = None
        if secondary_ent.lower() == 'nan':
            secondary_ent = None
        if secondary_qid.lower() == 'nan':
            secondary_qid = None
        if tertiary_ent.lower() == 'nan':
            tertiary_ent = None
        if tertiary_qid.lower() == 'nan':
            tertiary_qid = None

        # Add primary entity as a node with its QID if available
        if primary_ent:
            if G.has_node(primary_ent):
                G.nodes[primary_ent]['incident_ids'].append(str(incident_id))
            else:
                G.add_node(primary_ent, type='Primary', incident_ids=[str(incident_id)], qid=primary_qid)

        # Add secondary entity as a node with its QID if available
        if secondary_ent:
            if G.has_node(secondary_ent):
                G.nodes[secondary_ent]['incident_ids'].append(str(incident_id))
            else:
                G.add_node(secondary_ent, type='Secondary', incident_ids=[str(incident_id)], qid=secondary_qid)

        # Add tertiary entity as a node with its QID if available
        if tertiary_ent:
            if G.has_node(tertiary_ent):
                G.nodes[tertiary_ent]['incident_ids'].append(str(incident_id))
            else:
                G.add_node(tertiary_ent, type='Tertiary', incident_ids=[str(incident_id)], qid=tertiary_qid)

        # Create edges for relationships in NEL
        if primary_ent and secondary_ent:
            if G.has_edge(primary_ent, secondary_ent):
                G[primary_ent][secondary_ent]['incident_ids'].append(str(incident_id))
            else:
                G.add_edge(primary_ent, secondary_ent, relation='secondary_of', incident_ids=[str(incident_id)])
        if primary_ent and tertiary_ent:
            if G.has_edge(primary_ent, tertiary_ent):
                G[primary_ent][tertiary_ent]['incident_ids'].append(str(incident_id))
            else:
                G.add_edge(primary_ent, tertiary_ent, relation='tertiary_of', incident_ids=[str(incident_id)])

    # Print number of nodes and edges after step 3
    print(f"Nodes after processing nel_df: {len(G.nodes)}")
    print(f"Edges after processing nel_df: {len(G.edges)}")
    # Step 4: Add RE edges
    valid_relations = [
        "HAS EFFECT", "HAS CAUSE", "PART OF", "INSTANCE OF", "FOLLOWED BY", 
        "LOCATION", "TIME PERIOD", "MAINTAINED BY", "USED BY",
        "OWNED BY", "FACET OF", "FOLLOWS", "INFLUENCED BY", 
        "EVENT DISTANCE", "DESIGNED BY", "CONFLICT", 
        "OWNER OF"
    ]

    for _, row in re_df.iterrows():
        incident_id = row['c5_unique_id']
        relations = row['entity1, relation_type, entity2']
        
        if isinstance(relations, str):
            for group in relations.split(';'):
                relation_triples = [r.strip() for r in group.split(',')]
                if len(relation_triples) >= 3:
                    entity1, relation, entity2 = None, None, None
                    for i, item in enumerate(relation_triples):
                        if item in valid_relations:
                            relation = item
                            entity1 = ", ".join(relation_triples[:i])
                            entity2 = ", ".join(relation_triples[i + 1:])
                            break
                    
                    if strict_re_gs:
                        if G.has_node(entity1) and G.has_node(entity2):
                            if not G.has_edge(entity1, entity2):
                                G.add_edge(entity1, entity2, relation=relation, incident_ids=[str(incident_id)])
                    else:
                        if not G.has_node(entity1):
                            G.add_node(entity1, type="re", incident_ids=[str(incident_id)])
                        if not G.has_node(entity2):
                            G.add_node(entity2, type="re", incident_ids=[str(incident_id)])
                        if not G.has_edge(entity1, entity2):
                            G.add_edge(entity1, entity2, relation=relation, incident_ids=[str(incident_id)])
                    
                    # Count the relationships
                    if relation not in relationship_counts[strict_re_gs]:
                        relationship_counts[strict_re_gs][relation] = 0
                    relationship_counts[strict_re_gs][relation] += 1

    # Step 6: Ensure all attributes are strings before saving the GML
    def convert_to_string(graph):
        # Convert node attributes to strings
        for node, data in graph.nodes(data=True):
            for key, value in data.items():
                if value is None:
                    graph.nodes[node][key] = ""  # Convert None to empty string
                elif isinstance(value, list):
                    graph.nodes[node][key] = [str(v) if v is not None else "" for v in value]
                else:
                    graph.nodes[node][key] = str(value)  # Ensure all values are strings
        
        # Convert edge attributes to strings
        for u, v, data in graph.edges(data=True):
            for key, value in data.items():
                if value is None:
                    graph[u][v][key] = ""  # Convert None to empty string
                elif isinstance(value, list):
                    graph[u][v][key] = [str(v) if v is not None else "" for v in value]
                else:
                    graph[u][v][key] = str(value)  # Ensure all values are strings

    # Convert attributes of the graph
    convert_to_string(G)
    # Export GML
    nx.write_gml(G, f"knowledge_graph_{strict_re_gs}.gml")
    print(f"{strict_re_gs.upper()} GML saved.")
    
    # Export Nodes and Edges CSV
    nodes_df = pd.DataFrame([{
        'node': node,
        'type': data.get('type', ''),
        'incident_ids': ', '.join(sorted(set(data.get('incident_ids', [])))),
    } for node, data in G.nodes(data=True)])
    nodes_df.to_csv(f"knowledge_graph_nodes_{strict_re_gs}.csv", index=False)
    
    edges_df = pd.DataFrame([{
        'entity1': u,
        'entity2': v,
        'relation': data.get('relation', ''),
        'incident_ids': ', '.join(sorted(set(data.get('incident_ids', [])))),
    } for u, v, data in G.edges(data=True)])
    edges_df.to_csv(f"knowledge_graph_edges_{strict_re_gs}.csv", index=False)

    print(f"{strict_re_gs.upper()} Nodes and Edges saved.")

# Output the relationship counts to a CSV
relationship_counts_df = pd.DataFrame([
    {'type': rel_type, 'strict': relationship_counts['strict'].get(rel_type, 0), 'loose': relationship_counts['loose'].get(rel_type, 0)}
    for rel_type in set(relationship_counts['strict'].keys()).union(relationship_counts['loose'].keys())
])
relationship_counts_df.to_csv('relationship_counts.csv', index=False)

# Visualization and Saving as PNG
# Draw the graph (simple visualization, you might want to use different layouts for better readability)
for strict_re_gs in re_paths:
    G = nx.read_gml(f"knowledge_graph_{strict_re_gs}.gml")
    
    plt.figure(figsize=(10, 10))
    nx.draw(G, with_labels=True, node_size=50, font_size=10)
    plt.title(f"Knowledge Graph: {strict_re_gs.upper()}")
    
    # Save the figure as a PNG
    plt.savefig(f"knowledge_graph_{strict_re_gs}.png", format='png')
    plt.close()  # Close the plot to avoid display and memory issues
    print(f"Graph visualization for {strict_re_gs.upper()} saved as PNG.")
