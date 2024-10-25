import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# Load the CSV files
base_path = "../../OMIn_dataset/gold_standard/processed"
ner_path = f"{base_path}/ner.csv"
nel_path = f"{base_path}/nel.csv"
cr_path = f"{base_path}/cr.csv"
re_path = f"{base_path}/re.csv"

# Read the CSVs into pandas DataFrames
ner_df = pd.read_csv(ner_path, delimiter=',', encoding='utf-8')
nel_df = pd.read_csv(nel_path, delimiter=',', encoding='utf-8')
cr_df = pd.read_csv(cr_path, delimiter=',', encoding='utf-8')
re_df = pd.read_csv(re_path, delimiter=',', encoding='utf-8')

# Create a directed graph for the Knowledge Graph
G = nx.DiGraph()

# Step 1: Process Entities and QIDs from nel_df for nodes
for _, row in nel_df.iterrows():
    incident_id = row['id']
    entities = eval(row['entity'])  # Convert the string representation to a list
    qids = eval(row['qid'])         # Convert the string representation to a list
    
    # Add nodes with QID as an attribute where available
    for entity, qid in zip(entities, qids):
        if entity:  # Check for non-empty entity strings
            #TODO: all should have incident ids
            G.add_node(entity, qid=str(qid) if qid else "", incident_id=str(incident_id))

# Step 2: Process Coreferences for merging nodes in cr_df
for _, row in cr_df.iterrows():
    incident_id = row['id']
    coreferences = eval(row['coreferences'])  # Convert the string to a list of coreference pairs
    
    # Merge coreferential entities
    for coref in coreferences:
        if len(coref) > 1:
            primary = coref[0]  # The primary mention (position tuple)
            for secondary in coref[1:]:
                # Extract coreferential text spans (sample[start:end])
                primary_text = row['sample'][primary[0]:primary[1]] if len(primary) == 2 else ''
                secondary_text = row['sample'][secondary[0]:secondary[1]] if len(secondary) == 2 else ''
                if primary_text and secondary_text and primary_text != secondary_text:
                    # Create an edge indicating the coreferential relationship
                    G.add_edge(primary_text, secondary_text, relation="coreferential")

# Step 3: Extract Relationships from ner_df\
#TODO: Probably will remove
for _, row in ner_df.iterrows():
    incident_id = row['id']
    entities = [row['entities']] if isinstance(row['entities'], str) else row['entities']  # Handle single entities
    
    # Create a relationship between each entity in the same incident sample
    for i in range(len(entities) - 1):
        entity1 = entities[i]
        entity2 = entities[i + 1]
        if entity1 and entity2:  # Ensure non-empty strings
            G.add_edge(entity1, entity2, relation="relatedTo", incident_id=str(incident_id))

# Step 4: Create Edges from re_df
#TODO update file with new GS
for _, row in re_df.iterrows():
    incident_id = row['id']
    subject = row['subject']
    relation = row['relation']
    object_ = row['object']
    
    # Add edges based on the relationship information
    if subject and object_:
        G.add_edge(subject, object_, relation=relation, incident_id=str(incident_id))

# Display the number of nodes and edges to understand the graph's size
print(f"Number of nodes: {len(G.nodes)}")
print(f"Number of edges: {len(G.edges)}")

# Ensure all attributes are strings before saving
for node, data in G.nodes(data=True):
    for k, v in data.items():
        if v is None:
            G.nodes[node][k] = ""

for u, v, data in G.edges(data=True):
    for key in data:
        data[key] = str(data[key]) if data[key] is not None else ""

# Save the graph to a GML file
nx.write_gml(G, "knowledge_graph.gml")

# Save the graph nodes to a CSV file
nodes_data = []
for node, data in G.nodes(data=True):
    nodes_data.append([node, data.get('qid', ''), data.get('incident_id', '')])

nodes_df = pd.DataFrame(nodes_data, columns=['Node', 'QID', 'Incident_ID'])
nodes_df.to_csv("knowledge_graph_nodes.csv", index=False)

# Save the graph edges to a CSV file
edges_data = []
for u, v, data in G.edges(data=True):
    edges_data.append([u, v, data.get('relation', ''), data.get('incident_id', '')])

edges_df = pd.DataFrame(edges_data, columns=['Source', 'Target', 'Relation', 'Incident_ID'])
edges_df.to_csv("knowledge_graph_edges.csv", index=False)

# Visualization
plt.figure(figsize=(12, 12))
pos = nx.spring_layout(G, k=0.15)
nx.draw(G, pos, with_labels=True, node_size=20, font_size=8, arrows=True)
plt.title("Knowledge Graph")
plt.show()
