import pandas as pd
from tqdm import tqdm
import os
import sys

# Set up the module path for imports
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

# Import the updated GraphRAG module
from graph_rag.KEO_GraphRAG import GraphRetriever, load_aviation_graph

# File paths
qa_file_path = "GPT4o_Generated_QA.csv"
faa_data_file_path = "../../OMIn_dataset/data/FAA_data/FAA_sample_100.csv"
output_file_path_list = ["GPT4o_GraphRAG_strict_QA_Answers.csv"]
graph_file_path_list = ["../kg/knowledge_graph_strict.gml"]
openai_api_key = os.getenv("OPENAI_API_KEY")  # Load from environment variable

for graph_file_path, output_file_path in zip(graph_file_path_list, output_file_path_list):
    # Initialize the pipeline
    print("Initializing the pipeline...")

    # Load the knowledge graph
    print(f"Loading knowledge graph from {graph_file_path}...")
    graph = load_aviation_graph(graph_file_path)
    if not graph:
        raise RuntimeError("Failed to load the knowledge graph.")

    # Initialize the GraphRetriever
    print("Initializing the GraphRetriever...")
    retriever = GraphRetriever(graph=graph, openai_api_key=openai_api_key)

    # Generate embeddings (using cache if available)
    print("Generating embeddings...")
    retriever.generate_embeddings()

    # Load the QA dataset
    print(f"Loading QA data from {qa_file_path}...")
    qa_data = pd.read_csv(qa_file_path)

    # Initialize an empty list to store results
    processed_qa = []

    # Process each QA pair
    print("Processing QA pairs...")
    for idx, row in tqdm(qa_data.iterrows(), total=len(qa_data), desc="QA Processing"):
        c5 = row['c5']
        c119_text = row['c119_text']
        question = row['Question']

        # Query the graph to get an answer
        try:
            results = retriever.query(question, k=8)
            # Generate a structured answer
            answer = retriever.generate_structured_answer(query=question, results=results)
        except Exception as e:
            answer = f"Error processing question: {str(e)}"

        # Append the results
        processed_qa.append({
            "c5": c5,
            "c119_text": c119_text,
            "question": question,
            "answer": answer
        })

    # Convert the results to a DataFrame
    output_df = pd.DataFrame(processed_qa)

    # Save the processed QA with answers to a CSV file
    print(f"Saving processed QA with answers to {output_file_path}...")
    output_df.to_csv(output_file_path, index=False)

    print("Processing complete!")
