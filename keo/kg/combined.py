import pandas as pd

# Load the CSV file into a DataFrame
file_path = 'knowledge_graph_nodes.csv'
data = pd.read_csv(file_path)

# Splitting incident_ids and expanding into multiple rows
data_expanded = data.assign(incident_ids=data['incident_ids'].str.split(',')).explode('incident_ids')

# Cleaning up whitespace in incident_ids
data_expanded['incident_ids'] = data_expanded['incident_ids'].str.strip()

# Grouping nodes by incident_ids while retaining the c119_text
result = data_expanded.groupby('incident_ids').agg({
    'c119_text': 'first',  # The first c119_text per incident_id
    'node': lambda x: ', '.join(x)  # Combine all associated nodes as a string
}).reset_index()

# Renaming columns for clarity
result.rename(columns={
    'incident_ids': 'id',
    'c119_text': 'sentence',
    'node': 'nodes'
}, inplace=True)

# Saving the processed data to a CSV file
output_path = 'combined_knowledge_graph.csv'
result.to_csv(output_path, index=False)

print(f"The output file has been saved to: {output_path}")

print(f"Output saved to {output_path}")
