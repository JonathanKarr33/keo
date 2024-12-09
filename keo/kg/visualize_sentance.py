import networkx as nx
import matplotlib.pyplot as plt

# Load the graph from the GML file
def load_graph_from_gml(file_path):
    return nx.read_gml(file_path)

# Extract subgraph based on a specific incident ID (both strict and loose behave the same)
def extract_subgraph_by_incident(graph, incident_id):
    nodes_to_include = set()

    # Loop through all nodes and include those that are directly related to the incident_id
    for node, data in graph.nodes(data=True):
        if 'incident_ids' in data and incident_id in data['incident_ids']:
            nodes_to_include.add(node)

    # Create a subgraph with the selected nodes
    subgraph = graph.subgraph(nodes_to_include).copy()
    
    return subgraph

# Visualize and save the subgraph as a PNG file
def visualize_and_save_subgraph(subgraph, output_file, incident_id):
    plt.figure(figsize=(9, 9))
    pos = nx.spring_layout(subgraph, seed=42)  # Position nodes using spring layout
    
    # Draw nodes and edges
    nx.draw(subgraph, pos, with_labels=True, node_size=3000, node_color="lightblue", font_size=10, font_weight='bold', edge_color='gray')

    # Add edge labels if they exist (assuming the edge has 'edge_type' attribute)
    edge_labels = {}
    for u, v, data in subgraph.edges(data=True):
        if 'edge_type' in data:
            edge_labels[(u, v)] = data['edge_type']
    
    if edge_labels:
        nx.draw_networkx_edge_labels(subgraph, pos, edge_labels=edge_labels, font_size=10, font_weight='bold', font_color='red')
    
    plt.title(f"Subgraph for Incident ID '{incident_id}'", fontsize=18)
    
    # Save the plot as PNG
    plt.savefig(output_file, format="PNG")
    plt.close()

# Main function to run the process
def main():
    # Paths to the GML files
    gml_file_strict = 'knowledge_graph_strict.gml'
    gml_file_loose = 'knowledge_graph_loose.gml'
    
    # Load the graphs (both behave the same way)
    graph_strict = load_graph_from_gml(gml_file_strict)
    graph_loose = load_graph_from_gml(gml_file_loose)
    
    # Incident ID to filter
    incident_id = '19800217031649I'
    
    # Define output paths
    strict_output_file = f'subgraph_strict_{incident_id}.png'
    loose_output_file = f'subgraph_loose_{incident_id}.png'
    
    # Extract the subgraph related to the incident ID (same extraction logic for both files)
    strict_subgraph = extract_subgraph_by_incident(graph_strict, incident_id)
    visualize_and_save_subgraph(strict_subgraph, strict_output_file, incident_id)
    print(f"Subgraph for incident ID '{incident_id}' from strict file saved as {strict_output_file}")
    
    loose_subgraph = extract_subgraph_by_incident(graph_loose, incident_id)
    visualize_and_save_subgraph(loose_subgraph, loose_output_file, incident_id)
    print(f"Subgraph for incident ID '{incident_id}' from loose file saved as {loose_output_file}")

if __name__ == "__main__":
    main()
