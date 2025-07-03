import csv
import os
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import argparse

# Model configs
SMALL_MODELS = [
    ("gemma3_4b_it", "google/gemma-3-4b-it"),
    ("phi4mini_instruct", "microsoft/Phi-4-mini-instruct"),
]
LARGE_MODELS = [
    ("gemma3_12b_it", "google/gemma-3-12b-it"),
    ("phi4_12b", "microsoft/phi-4"),
    ("mistral_small_instruct_2409", "mistralai/Mistral-Small-Instruct-2409"),
]

# Prompt template
PROMPT_TEMPLATE = '''Instruction:
Extract informative triplets directly from the text following the examples. 
Format each triplet as: <entity1, relation, entity2>
Do not add any extra words, line breaks, or explanatory notes.
Focus on extracting factual relationships from the text.

Use only these relation types:
- OWNED BY
- INSTANCE OF 
- FOLLOWED BY
- HAS CAUSE
- FOLLOWS
- EVENT DISTANCE
- HAS EFFECT
- LOCATION
- USED BY
- INFLUENCED BY
- TIME PERIOD
- PART OF
- MAINTAINED BY
- DESIGNED BY

Example:
Text: The Wright brothers designed the first successful airplane in 1903 in Kitty Hawk.
Triplets:
<first successful airplane, DESIGNED BY, Wright brothers>
<first successful airplane, TIME PERIOD, 1903>
<first successful airplane, LOCATION, Kitty Hawk>

Target Text: {text}
Triplets:
'''

# Helper to load model and tokenizer
def load_model_and_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32, device_map="auto", trust_remote_code=True)
    return tokenizer, model

def generate_triplets(prompt, tokenizer, model, max_new_tokens=256, temperature=0.1):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs, 
            max_new_tokens=max_new_tokens, 
            do_sample=temperature > 0,  # Only sample if temperature > 0
            temperature=temperature,
            pad_token_id=tokenizer.pad_token_id
        )
    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    if 'Triplets:' in decoded:
        return decoded.split('Triplets:')[-1].strip()
    else:
        return decoded.strip()

def main():
    parser = argparse.ArgumentParser(description="Extract triplets using selected LLMs.")
    parser.add_argument('--size', choices=['small', 'large'], default='small', help='Model size set to use: small (default) or large')
    args = parser.parse_args()

    if args.size == 'small':
        models = SMALL_MODELS
        output_csv = "re_gs_strict_kg_gemma3_phi4mini.csv"
    else:
        models = LARGE_MODELS
        output_csv = "re_gs_strict_kg_gemma12b_phi12b_mistral.csv"
        print("WARNING: Large models may require significant GPU memory. If you encounter OOM errors, run each model separately and merge results.")

    input_csv = "re_gs_strict.csv"

    # Load models
    model_objs = {}
    for shortname, modelname in models:
        print(f"Loading {modelname}...")
        model_objs[shortname] = load_model_and_tokenizer(modelname)

    # Read input CSV
    rows = []
    with open(input_csv, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    # Prepare output
    fieldnames = ["index", "c5_unique_id", "c119_text"] + [f"{shortname}_triplets" for shortname, _ in models]
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in tqdm(rows, desc="Processing rows"):
            text = row["c119_text"]
            prompt = PROMPT_TEMPLATE.format(text=text)
            triplet_outputs = {}
            for shortname, _ in models:
                tokenizer, model = model_objs[shortname]
                triplet_outputs[f"{shortname}_triplets"] = generate_triplets(prompt, tokenizer, model)
            writer.writerow({
                "index": row["index"],
                "c5_unique_id": row["c5_unique_id"],
                "c119_text": text,
                **triplet_outputs
            })
            f.flush()  # Ensure the file is saved after each row

if __name__ == "__main__":
    """
    Usage:
      python generate_kg_triplets.py [--size small|large]

    Options:
      --size small   Use small models (default):
                       - Gemma 3 4B It (google/gemma-3-4b-it)
                       - Phi 4 mini Instruct (microsoft/Phi-4-mini-instruct)
      --size large   Use large models:
                       - Gemma 12B (google/gemma-3-12b-it)
                       - Phi 12B (microsoft/phi-4)
                       - Mistral-Small-Instruct-2409 (mistralai/Mistral-Small-Instruct-2409)

    Notes:
      - Output CSV columns will match the models used.
      - Large models require significant GPU memory. If you encounter OOM errors, run each model separately and merge results.
    """
    main() 