import pandas as pd
from tqdm import tqdm
from openai import OpenAI

# File paths
input_file_path_list = ["GraphRAG_QA_Answers.csv"]
output_file_path_list = ["GraphRAG_QA_Evaluation.csv"]

# OpenAI API Key
openai_api_key = "Your_OpenAI_API_Key"  # Replace with your OpenAI API key
client = OpenAI(api_key=openai_api_key)

for input_file_path, output_file_path in zip(input_file_path_list, output_file_path_list):
    # Load the QA answers dataset
    print(f"Loading answers data from {input_file_path}...")
    qa_data = pd.read_csv(input_file_path)

    # Initialize an empty list to store evaluated results
    evaluated_qa = []

    # Process each QA pair
    print("Evaluating answers with GPT-4 including Incident Description...")
    for idx, row in tqdm(qa_data.iterrows(), total=len(qa_data), desc="Evaluating QA"):
        c5 = row['c5']
        c119_text = row['c119_text']  # Incident Description
        question = row['question']
        answer = row['answer']

        # Prepare the prompt for evaluation
        prompt = f"""
    You are a highly knowledgeable and objective aviation safety expert tasked with evaluating the quality of answers to aviation-related questions. 

    For the given Incident Description, question, and answer, provide a rating on a scale of 0 to 5 (integer only):
    - 0: The answer is completely unrelated to the question or incident description.
    - 1: The answer is minimally relevant but mostly incorrect or incomplete.
    - 2: The answer shows some relevance but has significant inaccuracies or lacks depth.
    - 3: The answer is moderately relevant with minor issues or missing details.
    - 4: The answer is relevant and mostly accurate with only minor improvements needed.
    - 5: The answer is highly relevant, accurate, complete, and perfectly addresses the question and incident description.

    **Incident Description:**
    {c119_text}

    **Question:**
    {question}

    **Answer:**
    {answer}

    Provide your rating along with a brief justification for your rating in the following format:
    Rating: <0-5>
    Justification: <Your reasoning>
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
            evaluation_output = response.choices[0].message.content.strip()

            # Extract rating and justification
            rating = None
            justification = None
            for line in evaluation_output.split("\n"):
                if line.startswith("Rating:"):
                    rating = int(line.replace("Rating:", "").strip())
                if line.startswith("Justification:"):
                    justification = line.replace("Justification:", "").strip()
        except Exception as e:
            rating = -1  # Indicate an error occurred during evaluation
            justification = f"Error processing evaluation: {str(e)}"

        # Append the evaluated results
        evaluated_qa.append({
            "c5": c5,
            "incident_description": c119_text,
            "question": question,
            "answer": answer,
            "rating": rating,
            "justification": justification
        })

    # Convert the results to a DataFrame
    output_df = pd.DataFrame(evaluated_qa)

    # Save the evaluated QA results to a CSV file
    print(f"Saving evaluated QA with ratings to {output_file_path}...")
    output_df.to_csv(output_file_path, index=False)

    print("Evaluation complete!")
