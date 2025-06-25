import pandas as pd
import os
from openai import OpenAI  # Assuming you are using OpenAI API for the LLM interaction
from tqdm import tqdm  # Import the tqdm library

# Read the CSV file
file_path = "../../OMIn_dataset/data/FAA_data/FAA_sample_100.csv"
faa_data = pd.read_csv(file_path)

# Extract the "c119" column
c119_content = faa_data["c119"]

# Initialize an empty list to store the generated QA questions
generated_qa = []

# Few-shot examples
few_shot_examples = """
What is the most common cause of engine failure?
Describe incidents involving engine problems.
How do weather conditions affect incidents?
What are the main factors in landing incidents?
What types of pilot errors are reported?
What safety issues are most common?
"""

# OpenAI API Key (replace with your API key)
openai_api_key = os.getenv("OPENAI_API_KEY")  # Load from environment variable
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY environment variable not set. Please set it with your OpenAI API key.")
client = OpenAI(api_key=openai_api_key)

# Process each row in "c119" with a tqdm progress bar
for idx, content in tqdm(enumerate(c119_content), total=len(c119_content), desc="Processing rows"):
    # Create a prompt for the LLM agent
    prompt = f"""
Using the following incident information from aviation reports:
"{content}"

Generate 5 QA questions about aviation safety based on this content. Use these as examples for the style and structure of the questions, separate each question with a new line, and do not use numbers or bullet points.:
{few_shot_examples}
"""
    try:
        # Call the OpenAI API
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an assistant specializing in generating QA questions for aviation safety incidents."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0
        )

        # Extract and process GPT-4 output
        gpt_output = response.choices[0].message.content.strip()
        qa_questions = gpt_output.split("\n")
        for question in qa_questions:
            if len(question) > 0:
                generated_qa.append({"c5": faa_data["c5"][idx], "c119_text": content, "Question": question})

    except Exception as e:
        print("Error processing" + faa_data["c5"][idx] + f": {e}")
        continue

# Convert the generated QA questions into a DataFrame
qa_df = pd.DataFrame(generated_qa)

# Save the QA questions to a new CSV file
output_file_path = "GPT4o_Generated_QA.csv"
qa_df.to_csv(output_file_path, index=False)

print(f"Generated QA questions have been saved to {output_file_path}")
