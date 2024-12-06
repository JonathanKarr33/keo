import pandas as pd
from tqdm import tqdm
import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)
from graph_rag.KEO_GraphRAG_spacy import run_analysis_pipeline, query_graph

# File paths
qa_file_path = "GPT4o_Generated_QA.csv"
faa_data_file_path = "../../OMIn_dataset/data/FAA_data/FAA_sample_100.csv"
output_file_path = "GraphRAG_QA_Answers.csv"

# Initialize the pipeline
print("Initializing the pipeline...")
pipeline_results = run_analysis_pipeline(
    csv_path=faa_data_file_path,
    openai_api_key="Your_OpenAI_API_Key",  # Replace with your OpenAI API key
    cache_dir="graph_cache"
)
retriever = pipeline_results['retriever']

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
        answer = query_graph(retriever, query=question, k=8, threshold=0.3)
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
