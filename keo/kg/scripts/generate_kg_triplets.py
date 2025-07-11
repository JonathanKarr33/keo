import csv
import os
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, AutoProcessor
import torch
import argparse
import re
import warnings

try:
    from transformers import Gemma3ForConditionalGeneration
except ImportError:
    Gemma3ForConditionalGeneration = None

os.environ["TORCHDYNAMO_DISABLE"] = "1"

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
TEXT: THE WRIGHT BROTHERS DESIGNED THE FIRST SUCCESSFUL AIRPLANE IN 1903 IN KITTY HAWK.
Triplets:
<FIRST SUCCESSFUL AIRPLANE, DESIGNED BY, WRIGHT BROTHERS>
<FIRST SUCCESSFUL AIRPLANE, TIME PERIOD, 1903>
<FIRST SUCCESSFUL AIRPLANE, LOCATION, KITTY HAWK>

Target Text: {text}
Triplets:
'''

# Helper to load model and tokenizer/processor
def load_model_and_tokenizer(model_name, shortname):
    if shortname.startswith("gemma3"):
        if Gemma3ForConditionalGeneration is None:
            raise ImportError("transformers >=4.50.0 required for Gemma3 support.")
        processor = AutoProcessor.from_pretrained(model_name, use_fast=True)
        model = Gemma3ForConditionalGeneration.from_pretrained(
            model_name, device_map="auto"
        ).eval()
        return processor, model
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32, device_map="auto", trust_remote_code=True
        )
        return tokenizer, model

def extract_triplets_only(text):
    # Find all lines that match the triplet pattern
    triplet_lines = re.findall(r'<[^>]+>', text)
    return '\n'.join(triplet_lines)

def generate_triplets(prompt, tokenizer_or_processor, model, shortname, max_new_tokens=256, temperature=0.1):
    if shortname.startswith("gemma3"):
        # Gemma expects multimodal format (see Hugging Face docs)
        messages = [
            {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
            {"role": "user", "content": [{"type": "text", "text": prompt}]}
        ]
        inputs = tokenizer_or_processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt"
        ).to(model.device)
        # Remove unsupported keys if present
        for k in ['top_p', 'top_k', 'temperature']:
            if k in inputs:
                del inputs[k]
    elif shortname.startswith("phi4"):
        # Phi expects string content (see Hugging Face docs)
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        inputs = tokenizer_or_processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt"
        ).to(model.device)
    else:
        inputs = tokenizer_or_processor(prompt, return_tensors="pt").to(model.device)
    
    if shortname.startswith("gemma3"):
        input_len = inputs["input_ids"].shape[-1]
        with torch.inference_mode():
            print(f"[INFO] Calling generate() for model: {shortname}")
            try:
                generation = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
            except Exception as e:
                print(f"[ERROR][{shortname}] {str(e)}")
                if "generation flags are not valid" in str(e).lower():
                    warnings.warn(f"[WARNING][{shortname}] {str(e)}")
                raise
            generation = generation[0][input_len:]
        decoded = tokenizer_or_processor.decode(generation, skip_special_tokens=True)
        if 'Triplets:' in decoded:
            triplet_text = decoded.split('Triplets:')[-1].strip()
        else:
            triplet_text = decoded.strip()
        triplet_text = extract_triplets_only(triplet_text)
        return triplet_text
    elif shortname.startswith("phi4"):
        input_len = inputs["input_ids"].shape[-1]
        with torch.inference_mode():
            print(f"[INFO] Calling generate() for model: {shortname}")
            try:
                generation = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
            except Exception as e:
                print(f"[ERROR][{shortname}] {str(e)}")
                if "generation flags are not valid" in str(e).lower():
                    warnings.warn(f"[WARNING][{shortname}] {str(e)}")
                raise
            generation = generation[0][input_len:]
        decoded = tokenizer_or_processor.decode(generation, skip_special_tokens=True)
        if 'Triplets:' in decoded:
            triplet_text = decoded.split('Triplets:')[-1].strip()
        else:
            triplet_text = decoded.strip()
        triplet_text = extract_triplets_only(triplet_text)
        return triplet_text
    else:
        with torch.no_grad():
            print(f"[INFO] Calling generate() for model: {shortname}")
            try:
                output = model.generate(
                    **inputs, 
                    max_new_tokens=max_new_tokens, 
                    do_sample=False,  # Greedy decoding, but temperature is set to 0.1
                    temperature=temperature,
                    pad_token_id=tokenizer_or_processor.pad_token_id
                )
            except Exception as e:
                print(f"[ERROR][{shortname}] {str(e)}")
                if "generation flags are not valid" in str(e).lower():
                    warnings.warn(f"[WARNING][{shortname}] {str(e)}")
                raise
        decoded = tokenizer_or_processor.decode(output[0], skip_special_tokens=True)
        if 'Triplets:' in decoded:
            triplet_text = decoded.split('Triplets:')[-1].strip()
        else:
            triplet_text = decoded.strip()
        triplet_text = extract_triplets_only(triplet_text)
        return triplet_text

# --- GPT-4o integration ---
def generate_triplets_with_gpt4o(text, client, max_tokens=256):
    prompt = PROMPT_TEMPLATE.format(text=text)
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that extracts knowledge graph triplets from text."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=0.1
        )
        response_text = response.choices[0].message.content.strip()
        triplets = extract_triplets_only(response_text)
        return triplets
    except Exception as e:
        print(f"Error generating triplets: {str(e)}")
        return ""
# --- end GPT-4o integration ---

def main():
    parser = argparse.ArgumentParser(description="Extract triplets using selected LLMs.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--size', choices=['small', 'large'], default='small', help='Model size set to use: small (default) or large')
    group.add_argument('--gpt4o', action='store_true', help='Use OpenAI GPT-4o API for triplet extraction')
    parser.add_argument('--gpt4o-test-n', nargs='?', type=int, default=None, help='For --gpt4o: number of rows to process. If omitted, defaults to 100. If flag is present with no value, defaults to 10. If a value is given, uses that value. Cannot be used with --all.')
    parser.add_argument('--all', action='store_true', help='If set, process all rows from the CSV file; otherwise, use the default 100-row CSV')
    args = parser.parse_args()

    if args.gpt4o:
        # Only import openai/dotenv if needed
        import importlib
        openai = importlib.import_module('openai')
        dotenv = importlib.import_module('dotenv')
        os_path = os.path
        dotenv.load_dotenv(os_path.join(os_path.dirname(os_path.dirname(__file__)), '.env'))
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set. Please create a .env file in the kg directory with your OpenAI API key.")
        print("✓ OpenAI API key verified")
        client = openai.OpenAI(api_key=openai_api_key)
        output_csv = "output/all_kg_llm_triplets_gpt4o.csv" if args.all else "output/100_kg_llm_triplets_gpt4o.csv"
    elif args.size == 'small':
        models = SMALL_MODELS
        output_csv = "output/all_kg_llm_triplets_gemma3_phi4mini.csv" if args.all else "output/100_kg_llm_triplets_gemma3_phi4mini.csv"
    else:
        models = LARGE_MODELS
        output_csv = "output/all_kg_llm_triplets_gemma12b_phi12b_mistral.csv" if args.all else "output/100_kg_llm_triplets_gemma12b_phi12b_mistral.csv"
        print("WARNING: Large models may require significant GPU memory. If you encounter OOM errors, run each model separately and merge results.")

    if args.all:
        input_file = "../../OMIn_dataset/data/FAA_data/Maintenance_Text_data_nona.csv"
    else:
        input_file = "../../OMIn_dataset/data/FAA_data/FAA_sample_100.csv"


    rows = []
    with open(input_file, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if 'c5' in row:
                c5_value = row['c5']
            else:
                c5_col = None
                for col in row.keys():
                    if 'c5' in col.lower():
                        c5_col = col
                        break
                if c5_col:
                    c5_value = row[c5_col]
                else:
                    c5_value = f"dummy_{len(rows)}"
            row['c5'] = c5_value
            if 'c119' in row:
                row['c119'] = row['c119'].upper()
            rows.append(row)

    # Determine number of rows for GPT-4o
    if args.gpt4o:
        if args.all and args.gpt4o_test_n is not None:
            print("Error: --gpt4o-test-n cannot be used with --all. If you want to process all rows, use only --gpt4o --all.")
            exit(1)
        if args.all:
            gpt4o_n = None  # None means all rows
        elif args.gpt4o_test_n is None:
            gpt4o_n = 100
        elif args.gpt4o_test_n is not None and not isinstance(args.gpt4o_test_n, int):
            gpt4o_n = 10
        else:
            gpt4o_n = args.gpt4o_test_n
        fieldnames = ["c5", "c119", "gpt4o_triplets", "gpt4o_triplets_clean"]
        total_prompt_tokens = 0
        total_completion_tokens = 0
        with open(output_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            if gpt4o_n is not None:
                rows = rows[:gpt4o_n]
            for row in tqdm(rows, desc=f"Processing rows (gpt4o-test-n={gpt4o_n})"):
                text = row['c119'] if 'c119' in row else next(iter(row.values()))
                # Call the API and get usage
                try:
                    response = client.chat.completions.create(
                        model="gpt-4o",
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant that extracts knowledge graph triplets from text."},
                            {"role": "user", "content": PROMPT_TEMPLATE.format(text=text)}
                        ],
                        max_tokens=256,
                        temperature=0.1
                    )
                    response_text = response.choices[0].message.content.strip()
                    triplets = extract_triplets_only(response_text)
                    usage = getattr(response, 'usage', None)
                    if usage:
                        total_prompt_tokens += usage.prompt_tokens
                        total_completion_tokens += usage.completion_tokens
                except Exception as e:
                    print(f"Error generating triplets: {str(e)}")
                    triplets = ""
                writer.writerow({
                    "c5": row['c5'],
                    "c119": text,
                    "gpt4o_triplets": triplets,
                    "gpt4o_triplets_clean": triplets
                })
                f.flush()
        # Pricing as of July 2024: $5/million input, $15/million output tokens
        input_cost = total_prompt_tokens * 0.000005
        output_cost = total_completion_tokens * 0.000015
        total_cost = input_cost + output_cost
        n_rows = gpt4o_n if gpt4o_n is not None else len(rows)
        print(f"\n--- GPT-4o Cost Estimation ({n_rows} rows) ---")
        print(f"Total prompt tokens: {total_prompt_tokens}")
        print(f"Total completion tokens: {total_completion_tokens}")
        print(f"Estimated input cost: ${input_cost:.4f}")
        print(f"Estimated output cost: ${output_cost:.4f}")
        print(f"Estimated total cost: ${total_cost:.4f}")
    else:
        # Load models
        model_objs = {}
        for shortname, modelname in models:
            print(f"Loading {modelname}...")
            model_objs[shortname] = load_model_and_tokenizer(modelname, shortname)
        fieldnames = ["c5", "c119"] + [f"{shortname}_triplets" for shortname, _ in models] + [f"{shortname}_triplets_clean" for shortname, _ in models]
        with open(output_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in tqdm(rows, desc="Processing rows"):
                text = row['c119'] if 'c119' in row else next(iter(row.values()))
                prompt = PROMPT_TEMPLATE.format(text=text)
                triplet_outputs = {}
                for shortname, _ in models:
                    tokenizer_or_processor, model = model_objs[shortname]
                    raw_output = generate_triplets(prompt, tokenizer_or_processor, model, shortname)
                    triplet_outputs[f"{shortname}_triplets"] = raw_output
                    triplet_outputs[f"{shortname}_triplets_clean"] = extract_triplets_only(raw_output)
                writer.writerow({
                    "c5": row['c5'],
                    "c119": text,
                    **triplet_outputs
                })
                f.flush()

if __name__ == "__main__":
    """
    Usage:
      python generate_kg_triplets.py [--size small|large|--gpt4o] [--all] [--gpt4o-test-n [N]]

    Options:
      --size small   Use small models (default):
                       - Gemma 3 4B It (google/gemma-3-4b-it)
                       - Phi 4 mini Instruct (microsoft/Phi-4-mini-instruct)
      --size large   Use large models:
                       - Gemma 12B (google/gemma-3-12b-it)
                       - Phi 12B (microsoft/phi-4)
                       - Mistral-Small-Instruct-2409 (mistralai/Mistral-Small-Instruct-2409)
      --gpt4o        Use OpenAI GPT-4o API for triplet extraction
      --gpt4o-test-n [N]  For --gpt4o: number of rows to process. If omitted, defaults to 100. If flag is present with no value, defaults to 10. If a value is given, uses that value. Cannot be used with --all.
      --all          Process all rows from the full CSV file (../../OMIn_dataset/data/FAA_data/Maintenance_Text_data_nona.csv) instead of the default 100-row CSV (../../OMIn_dataset/data/FAA_data/FAA_sample_100.csv). If used with --gpt4o, processes all rows. Cannot be combined with --gpt4o-test-n.

    Input:
      - Only CSV files are supported as input.
      - The input CSV must have a 'c5' column (document identifier) and a 'c119' column (input text).

    Output Format:
      The output CSV will include the following columns:
      - c5: Document identifier (extracted from input data)
      - c119: Input text
      - {model}_triplets: Raw triplet output from each model
      - {model}_triplets_clean: Cleaned triplet output from each model

    Notes:
      - Output CSV columns will match the models used.
      - Large models require significant GPU memory. If you encounter OOM errors, run each model separately and merge results.
      - c5 column is automatically extracted from CSV columns.
      - Only CSV input is supported (JSON input is no longer supported).
      - For --gpt4o: If --all is provided, all rows are processed. If --gpt4o-test-n is provided, that number of rows is processed. If neither is provided, 100 rows are processed by default. --all and --gpt4o-test-n cannot be used together.
    """
    main() 