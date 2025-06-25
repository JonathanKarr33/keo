import pandas as pd
from tqdm import tqdm
from openai import OpenAI
import os

# File paths
qa_file_path = "GPT4o_Generated_QA.csv"
output_file_path = "VanillaGPT4o_QA_Answers.csv"

# OpenAI API Key
openai_api_key = os.getenv("OPENAI_API_KEY")  # Load from environment variable
client = OpenAI(api_key=openai_api_key)

# Load the QA dataset
print(f"Loading QA data from {qa_file_path}...")
qa_data = pd.read_csv(qa_file_path)

# Initialize an empty list to store results
processed_qa = []

# Process each QA pair
print("Processing QA pairs with vanilla GPT-4o knowledge-based answers...")
for idx, row in tqdm(qa_data.iterrows(), total=len(qa_data), desc="QA Processing"):
    c5 = row['c5']
    question = row['Question']

    # Prepare the prompt for GPT-4o
    prompt = f"""
You are an aviation safety expert. Answer the following question based on your knowledge and expertise.

Question:
{question}

Answer:
"""
    # Query GPT-4o to get an answer
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an assistant specializing in aviation safety QA."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0
    )

        # Extract and process GPT-4 output
        answer = response.choices[0].message.content.strip()
    except Exception as e:
        answer = f"Error processing question: {str(e)}"

    # Append the results
    processed_qa.append({
        "c5": c5,
        "c119_text": row['c119_text'],
        "question": question,
        "answer": answer
    })

# Convert the results to a DataFrame
output_df = pd.DataFrame(processed_qa)

# Save the processed QA with answers to a CSV file
print(f"Saving processed QA with answers to {output_file_path}...")
output_df.to_csv(output_file_path, index=False)

print("Processing complete!")