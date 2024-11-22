import pandas as pd

# Load the CSV file into a DataFrame
file_path = 'knowledge_graph_nodes.csv'
data = pd.read_csv(file_path)
print("Data loaded successfully:")
print(data.head())

# Create a dictionary to hold sentences and their associated nodes
sentence_nodes = {}

# Iterate through the DataFrame and populate the dictionary
for _, row in data.iterrows():
    sentence = row['c119_text']
    node = row['Node']
    
    if sentence in sentence_nodes:
        sentence_nodes[sentence].append(node)
    else:
        sentence_nodes[sentence] = [node]

# Create a new DataFrame to hold the output
output_data = {
    'Sentence': [],
    'Nodes': []
}

# Populate the output DataFrame
for sentence, nodes in sentence_nodes.items():
    output_data['Sentence'].append(sentence)
    output_data['Nodes'].append(', '.join(nodes))

# Convert the output data to a DataFrame
output_df = pd.DataFrame(output_data)

# Define the output file path
output_file_path = 'combined_knowledge_graph.csv'

# Save the DataFrame to a CSV file
output_df.to_csv(output_file_path, index=False)

print(f"Output saved to {output_file_path}")
