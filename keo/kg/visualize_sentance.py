import networkx as nx
import matplotlib.pyplot as plt

# Load the graph from the GML file
def load_graph_from_gml(file_path):
    return nx.read_gml(file_path)

# Extract subgraph based on a specific incident ID and type of extraction (strict or loose)
def extract_subgraph_by_incident(graph, incident_id, extraction_type="strict"):
    nodes_to_include = set()
    edges_to_include = []

    for node in graph.nodes(data=True):
        # If the node's incident_ids contains the incident_id, include the node
        if 'incident_ids' in node[1]:
            if incident_id in node[1]['incident_ids']:
                nodes_to_include.add(node[0])
                # Collect edges connected to this node
                for neighbor in graph.neighbors(node[0]):
                    edges_to_include.append((node[0], neighbor))
    
    if extraction_type == "loose":
        # For loose extraction, include neighbors of nodes already marked
        loose_nodes_to_include = set(nodes_to_include)
        for node in nodes_to_include:
            for neighbor in graph.neighbors(node):
                loose_nodes_to_include.add(neighbor)
        
        nodes_to_include = loose_nodes_to_include
    
    # Create a subgraph with the selected nodes and edges
    subgraph = graph.subgraph(nodes_to_include).copy()
    subgraph.add_edges_from(edges_to_include)
    
    return subgraph

# Visualize and save the subgraph as a PNG file
def visualize_and_save_subgraph(subgraph, output_file, incident_id, extraction_type):
    plt.figure(figsize=(12, 12))
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
    
    plt.title(f"Subgraph for Incident ID '{incident_id}' ({extraction_type.capitalize()} Extraction)", fontsize=16)
    
    # Save the plot as PNG
    plt.savefig(output_file, format="PNG")
    plt.close()

# Main function to run the process
def main():
    # Paths to the GML files
    gml_file_strict = 'knowledge_graph_strict.gml'
    gml_file_loose = 'knowledge_graph_loose.gml'
    
    # Load the graphs
    graph_strict = load_graph_from_gml(gml_file_strict)
    graph_loose = load_graph_from_gml(gml_file_loose)
    
    # Incident ID to filter
    incident_id = '19800217031649I'
    
    # Define output paths
    strict_output_file = f'subgraph_strict_{incident_id}.png'
    loose_output_file = f'subgraph_loose_{incident_id}.png'
    
    # Extract the subgraph related to the incident ID (strict) from the strict GML file
    strict_subgraph = extract_subgraph_by_incident(graph_strict, incident_id, extraction_type="strict")
    visualize_and_save_subgraph(strict_subgraph, strict_output_file, incident_id, extraction_type="strict")
    print(f"Strict subgraph for incident ID '{incident_id}' saved as {strict_output_file}")
    
    # Extract the subgraph related to the incident ID (loose) from the loose GML file
    loose_subgraph = extract_subgraph_by_incident(graph_loose, incident_id, extraction_type="loose")
    visualize_and_save_subgraph(loose_subgraph, loose_output_file, incident_id, extraction_type="loose")
    print(f"Loose subgraph for incident ID '{incident_id}' saved as {loose_output_file}")

if __name__ == "__main__":
    main()
