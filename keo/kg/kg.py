import pandas as pd
import networkx as nx
import ast
import matplotlib.pyplot as plt

# Load the CSV files
base_path = "../../OMIn_dataset/gold_standard/raw"
ner_path = f"{base_path}/ner.csv"
re_path = f"{base_path}/re.csv"
cr_path = f"{base_path}/cr.csv"
nel_path = f"{base_path}/nel.csv"

# Read the CSVs into pandas DataFrames
ner_df = pd.read_csv(ner_path, delimiter=',', encoding='utf-8')
re_df = pd.read_csv(re_path, delimiter=',', encoding='utf-8')
cr_df = pd.read_csv(cr_path, delimiter=',', encoding='utf-8')
nel_df = pd.read_csv(nel_path, delimiter=',', encoding='utf-8')

# Create a directed graph for the Knowledge Graph
G = nx.DiGraph()

# Create a dictionary to hold sentences for each incident
incident_sentences = {}

# Step 1: Process Entities and Types from ner_df for nodes
for _, row in ner_df.iterrows():
    incident_id = row['c5_unique_id']
    entities = ast.literal_eval(row['GS'])  # Convert string representation to list
    types = ast.literal_eval(row['GS TYPE'])  # Convert string representation to list
    
    # Ensure GS and GS TYPE have the same length
    if len(entities) == len(types):
        # Add entities as nodes with their types as attributes
        for entity, entity_type in zip(entities, types):
            if entity:
                if G.has_node(entity):
                    G.nodes[entity]['incident_ids'].append(str(incident_id))  # Append if node exists
                else:
                    G.add_node(entity, type=entity_type, incident_ids=[str(incident_id)])
    else:
        print(f"Warning: Mismatched lengths for GS and GS TYPE in incident {incident_id}")

# Print number of nodes and edges after step 1
print(f"Nodes after processing ner_df: {len(G.nodes)}")
print(f"Edges after processing ner_df: {len(G.edges)}")

# Step 2: Add coreference relationships from cr_df
for _, row in cr_df.iterrows():
    incident_id = row['c5']
    coreferences = row['coreferences_human_readable']
    
    # Check if coreferences is not null or empty
    if pd.notnull(coreferences) and coreferences:
        coreferences = coreferences.strip()[1:-1].split(', ')

        # Iterate through the coreferences
        for i in range(len(coreferences)):
            for j in range(i + 1, len(coreferences)):
                entity1 = coreferences[i].strip()  # First entity
                entity2 = coreferences[j].strip()  # Second entity
                
                # Add the nodes to the graph if they don't exist
                if G.has_node(entity1):
                    G.nodes[entity1]['incident_ids'].append(str(incident_id))  # Add incident_id if node exists
                else:
                    G.add_node(entity1, incident_ids=[str(incident_id)])

                if G.has_node(entity2):
                    G.nodes[entity2]['incident_ids'].append(str(incident_id))  # Add incident_id if node exists
                else:
                    G.add_node(entity2, type="coreference", incident_ids=[str(incident_id)])

                # Add an edge between the coreferenced entities
                if not G.has_edge(entity1, entity2):
                    G.add_edge(entity1, entity2, relation='coreference', incident_ids=[str(incident_id)])
                else:
                    G[entity1][entity2]['incident_ids'].append(str(incident_id))  # Append to existing edge

# Print number of nodes and edges after step 2
print(f"Nodes after processing cr_df: {len(G.nodes)}")
print(f"Edges after processing cr_df: {len(G.edges)}")

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

# Step 4: Create Edges from re_df and populate incident_sentences
for _, row in re_df.iterrows():
    incident_id = row['c5_unique_id']
    # Store the c119_text corresponding to this incident ID
    incident_sentences[incident_id] = row['c119_text'].strip()  # Strip trailing whitespaces

    relations = row['entity1, relation_type, entity2']
    
    if isinstance(relations, str):
        # Split by commas and strip extra spaces
        relation_triples = [r.strip() for r in relations.split(',')]
        if len(relation_triples) == 3:
            entity1, relation, entity2 = relation_triples
            # Create an edge if not already present
            if G.has_node(entity1) and G.has_node(entity2):
                if not G.has_edge(entity1, entity2):
                    G.add_edge(entity1, entity2, relation=relation, incident_ids=[str(incident_id)])
                else:
                    G[entity1][entity2]['incident_ids'].append(str(incident_id))

# Print number of nodes and edges after step 4
print(f"Nodes after processing re_df: {len(G.nodes)}")
print(f"Edges after processing re_df: {len(G.edges)}")

# Step 5: Create the combined_knowledge_graph.csv and nodes and edges csv
combined_data = []

# For each unique incident_id, collect the nodes associated with that incident
for incident_id, c119_text in incident_sentences.items():
    nodes_for_incident = []

    # Collect nodes that have this incident_id
    for node, data in G.nodes(data=True):
        if str(incident_id) in data.get('incident_ids', []):
            nodes_for_incident.append(node)

    # Append the combined data
    combined_data.append({
        'incident_id': incident_id,
        'c119_text': c119_text,
        'nodes': ', '.join(nodes_for_incident)
    })

# Create DataFrame for combined knowledge graph
combined_df = pd.DataFrame(combined_data)
combined_df.to_csv('combined_knowledge_graph.csv', index=False)

# Export Nodes and Edges
nodes_data = [{
    'node': node,
    'type': data.get('type', ''),
    'incident_ids': ', '.join(sorted(set(data.get('incident_ids', [])))),
    'c119_text': '; '.join(
        sorted(set(incident_sentences.get(i, '').strip() for i in data.get('incident_ids', [])))
    )
} for node, data in G.nodes(data=True)]

nodes_df = pd.DataFrame(nodes_data)
nodes_df.to_csv('knowledge_graph_nodes.csv', index=False)

edges_data = [{
    'entity1': u,
    'entity2': v,
    'relation': data.get('relation', ''),
    'incident_ids': ', '.join(sorted(set(data.get('incident_ids', [])))),
    'c119_text': '; '.join(
        sorted(set(incident_sentences.get(i, '').strip() for i in data.get('incident_ids', [])))
    )
} for u, v, data in G.edges(data=True)]

edges_df = pd.DataFrame(edges_data)
edges_df.to_csv('knowledge_graph_edges.csv', index=False)



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

# Now, write the graph to a GML file
nx.write_gml(G, "knowledge_graph.gml")
print("GML file saved successfully.")

# Visualization
plt.figure(figsize=(12, 12))
pos = nx.spring_layout(G, k=0.3)
nx.draw(G, pos, with_labels=True, node_size=50, font_size=8)
plt.savefig("knowledge_graph.png", format="PNG")
print("Knowledge graph visualization and GML file saved.")
