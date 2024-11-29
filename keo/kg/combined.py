import pandas as pd

# Load the CSV
file_path = 'knowledge_graph_nodes.csv'
knowledge_graph_nodes = pd.read_csv(file_path)

# Step 1: Inspect the data to verify structure
print("Original data structure:")
print(knowledge_graph_nodes.head())

# Ensure required columns exist
if 'incident_ids' not in knowledge_graph_nodes.columns or 'c119_text' not in knowledge_graph_nodes.columns or 'node' not in knowledge_graph_nodes.columns:
    raise ValueError("Missing one or more required columns: 'incident_ids', 'c119_text', 'node'")

# Step 2: Split 'incident_ids' and 'c119_text' into individual rows
try:
    knowledge_graph_expanded = knowledge_graph_nodes.assign(
        incident_ids=knowledge_graph_nodes['incident_ids'].str.split(','),
        c119_text=knowledge_graph_nodes['c119_text'].str.split(';')
    ).explode('incident_ids').explode('c119_text')
except Exception as e:
    raise ValueError(f"Error during split and explode: {e}")

# Step 3: Clean up whitespace
knowledge_graph_expanded['incident_ids'] = knowledge_graph_expanded['incident_ids'].str.strip()
knowledge_graph_expanded['c119_text'] = knowledge_graph_expanded['c119_text'].str.strip()

print("Expanded data structure:")
print(knowledge_graph_expanded.head())

# Step 4: Validate exploding process
if knowledge_graph_expanded.isnull().any().any():
    print("Warning: Null values found after exploding data.")

# Step 5: Group by 'incident_ids' and 'c119_text' and aggregate all nodes
try:
    combined_knowledge_graph = knowledge_graph_expanded.groupby(
        ['incident_ids', 'c119_text'], as_index=False
    ).agg({
        'node': lambda x: sorted(set(x))  # Collect unique nodes for each group
    })
except Exception as e:
    raise ValueError(f"Error during grouping and aggregation: {e}")

# Step 6: Debug specific cases
debug_sentence = "TIRED TAXI WITH TIEDOWN CHAINS ATTACHE. ROLLED OVER. PILOT FAILED NOTE RAMP PERSON TIED SKIDS DUE HIGH WIND."
debug_rows = combined_knowledge_graph[combined_knowledge_graph['c119_text'] == debug_sentence]
print("Debugging specific sentence:")
print(debug_rows)

# Step 7: Save the results
output_file_path = 'combined_knowledge_graph.csv'
try:
    combined_knowledge_graph.to_csv(output_file_path, index=False)
    print(f"File saved successfully: {output_file_path}")
except Exception as e:
    raise ValueError(f"Error saving the file: {e}")
