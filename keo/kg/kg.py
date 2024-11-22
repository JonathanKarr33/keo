import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import ast

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
                G.add_node(entity, type=entity_type, incident_id=str(incident_id))
    else:
        #TODO: Fix csv mismatched lengths
        print(f"Warning: Mismatched lengths for GS and GS TYPE in incident {incident_id}")
print("Named Entity Recognition nodes were added.")
print(f"Number of nodes: {len(G.nodes)}")

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
                if not G.has_node(entity1):
                    G.add_node(entity1, incident_id=str(incident_id))
                
                if not G.has_node(entity2):
                    G.add_node(entity2, type="coreference", incident_id=str(incident_id))

                # Add an edge between the coreferenced entities
                G.add_edge(entity1, entity2, relation='coreference', incident_id=str(incident_id))
print("Coreference relationships processed and nodes added as needed:")
print(f"Number of nodes: {len(G.nodes)}")
print(f"Number of edges: {len(G.edges)}")

# Step 3: add NEL and its QIDs from nel.csv
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
        G.add_node(primary_ent, type='Primary', incident_id=str(incident_id), qid=primary_qid)

    # Add secondary entity as a node with its QID if available
    if secondary_ent:
        G.add_node(secondary_ent, type='Secondary', incident_id=str(incident_id), qid=secondary_qid)
    
    # Add tertiary entity as a node with its QID if available
    if tertiary_ent:
        G.add_node(tertiary_ent, type='Tertiary', incident_id=str(incident_id), qid=tertiary_qid)

    # Create edges for relationships in NEL
    if primary_ent and secondary_ent:
        G.add_edge(primary_ent, secondary_ent, relation='secondary_of', incident_id=str(incident_id))
    if primary_ent and tertiary_ent:
        G.add_edge(primary_ent, tertiary_ent, relation='tertiary_of', incident_id=str(incident_id))

print("Named entity linking with QIDs was added:")
print(f"Number of nodes: {len(G.nodes)}")
print(f"Number of edges: {len(G.edges)}")

# Step 4: Create Edges from re_df and populate incident_sentences
for _, row in re_df.iterrows():
    incident_id = row['c5_unique_id']
    # Store the c119_text corresponding to this incident ID
    incident_sentences[incident_id] = row['c119_text']
    
    relations = row['entity1, relation_type, entity2']
    
    if isinstance(relations, str):
        # Split by commas and strip extra spaces
        relation_triples = [r.strip() for r in relations.split(',')]
        
        # Process the relation triples as (subject, relation, object)
        for i in range(0, len(relation_triples), 3):
            try:
                subject = relation_triples[i]
                relation = relation_triples[i + 1]
                object_ = relation_triples[i + 2]
                
                # Add edges based on the relationship information, ensuring both nodes exist
                if subject and object_:
                    if G.has_node(subject) and G.has_node(object_):
                        G.add_edge(subject, object_, relation=relation, incident_id=str(incident_id))
            except IndexError:
                # Skip any incomplete triples
                continue

print("RE processing done:")
print(f"Number of nodes: {len(G.nodes)}")
print(f"Number of edges: {len(G.edges)}")

# Ensure all attributes are strings before saving
for node, data in G.nodes(data=True):
    for k, v in data.items():
        G.nodes[node][k] = str(v) if v else ""

for u, v, data in G.edges(data=True):
    for key in data:
        data[key] = str(data[key]) if data[key] else ""

# Prepare the nodes DataFrame including c119_text
nodes_df = pd.DataFrame([
    (node, data.get('type', ''), data.get('qid', ''), data.get('incident_id', ''), incident_sentences.get(data.get('incident_id', ''), '')) 
    for node, data in G.nodes(data=True)
], columns=['Node', 'Type', 'QID', 'Incident_ID', 'c119_text'])

# Save the nodes DataFrame to a CSV file
nodes_df.to_csv("knowledge_graph_nodes.csv", index=False)

# Prepare the edges DataFrame including c119_text
edges_df = pd.DataFrame([
    (u, v, data.get('relation', ''), data.get('incident_id', ''), incident_sentences.get(data.get('incident_id', ''), '')) 
    for u, v, data in G.edges(data=True)
], columns=['Source', 'Target', 'Relation', 'Incident_ID', 'c119_text'])

# Save the edges DataFrame to a CSV file
edges_df.to_csv("knowledge_graph_edges.csv", index=False)

# Save the graph to a GML file
nx.write_gml(G, "knowledge_graph.gml")

# Visualization
plt.figure(figsize=(12, 12))
pos = nx.spring_layout(G, k=0.15)
nx.draw(G, pos, with_labels=True, node_size=20, font_size=8, arrows=True)
plt.title("Knowledge Graph")
plt.show()
