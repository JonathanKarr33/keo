import csv
import os
from tqdm import tqdm
import argparse
import re
import importlib
import warnings
import torch
import concurrent.futures
import logging

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor
    from transformers import Gemma3ForConditionalGeneration
except ImportError:
    AutoTokenizer = None
    AutoModelForCausalLM = None
    AutoProcessor = None
    Gemma3ForConditionalGeneration = None

#Allow to recompile for models
import torch
    torch._dynamo.config.recompile_limit = float('inf')

# --- CONFIG ---
GS_CSV = "../../OMIn_dataset/data/FAA_data/FAA_sample_100.csv"
NON_GS_CSV = "../../OMIn_dataset/data/FAA_data/Maintenance_Text_data_nona.csv"

SMALL_MODELS = [
    ("gemma3_4b_it", "google/gemma-3-4b-it"),
    ("phi4mini_instruct", "microsoft/Phi-4-mini-instruct"),
]
LARGE_MODELS = [
    ("gemma3_12b_it", "google/gemma-3-12b-it"),
    ("phi4_12b", "microsoft/phi-4"),
    ("mistral_small_instruct_2409", "mistralai/Mistral-Small-Instruct-2409"),
]

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

Existing nodes in the knowledge graph: {node_list}
When extracting triplets, prefer to use these nodes as entities if possible, rather than inventing new entity mentions.

Example:
TEXT: THE WRIGHT BROTHERS DESIGNED THE FIRST SUCCESSFUL AIRPLANE IN 1903 IN KITTY HAWK.
Triplets:
<FIRST SUCCESSFUL AIRPLANE, DESIGNED BY, WRIGHT BROTHERS>
<FIRST SUCCESSFUL AIRPLANE, TIME PERIOD, 1903>
<FIRST SUCCESSFUL AIRPLANE, LOCATION, KITTY HAWK>

Target Text: {text}
Triplets:
'''

def extract_triplets_only(text):
    triplet_lines = re.findall(r'<[^>]+>', text)
    return '\n'.join(triplet_lines)

def parse_triplets(triplet_str):
    triplets = []
    for match in re.findall(r'<([^,]+),\s*([^,]+),\s*([^>]+)>', triplet_str):
        triplets.append(tuple(m.strip() for m in match))
    return triplets

def read_rows(csv_path, skip_c5=None, n=None):
    rows = []
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            c5 = row.get('c5')
            if skip_c5 and c5 in skip_c5:
                continue
            if 'c119' in row:
                row['c119'] = row['c119'].upper()
            rows.append(row)
            if n and len(rows) >= n:
                break
    return rows

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

def generate_triplets(prompt, tokenizer_or_processor, model, shortname, max_new_tokens=256, temperature=0.1):
    if shortname.startswith("gemma3"):
        messages = [
            {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
            {"role": "user", "content": [{"type": "text", "text": prompt}]}
        ]
        inputs = tokenizer_or_processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt"
        ).to(model.device)
        for k in ['top_p', 'top_k', 'temperature']:
            if k in inputs:
                del inputs[k]
    elif shortname.startswith("phi4"):
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
            try:
                generation = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
            except Exception as e:
                print(f"[ERROR][{shortname}] {str(e)}")
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
            try:
                generation = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
            except Exception as e:
                print(f"[ERROR][{shortname}] {str(e)}")
                raise
            generation = generation[0][input_len:]
        decoded = tokenizer_or_processor.decode(generation, skip_special_tokens=True)
        if 'Triplets:' in decoded:
            triplet_text = decoded.split('Triplets:')[-1].strip()
        else:
            triplet_text = decoded.strip()
        triplet_text = extract_triplets_only(decoded)
        return triplet_text
    else:
        with torch.no_grad():
            try:
                output = model.generate(
                    **inputs, 
                    max_new_tokens=max_new_tokens, 
                    do_sample=False,
                    temperature=temperature,
                    pad_token_id=tokenizer_or_processor.pad_token_id
                )
            except Exception as e:
                print(f"[ERROR][{shortname}] {str(e)}")
                raise
        decoded = tokenizer_or_processor.decode(output[0], skip_special_tokens=True)
        if 'Triplets:' in decoded:
            triplet_text = decoded.split('Triplets:')[-1].strip()
        else:
            triplet_text = decoded.strip()
        triplet_text = extract_triplets_only(decoded)
        return triplet_text

def main():
    parser = argparse.ArgumentParser(description="Extract triplets using selected LLMs, dynamically building the set of known nodes from previous batches. Generates 9 batch files: 100 GS, 100 next, ... up to 500, and cumulative 200, 300, 400, 500.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--size', choices=['small', 'large'], default='small', help='Model size set to use: small (default) or large')
    group.add_argument('--gpt4o', action='store_true', help='Use OpenAI GPT-4o API for triplet extraction')
    parser.add_argument('--gpt4o-test-n', nargs='?', type=int, default=None, help='For --gpt4o: number of rows to process. If omitted, defaults to 100. If flag is present with no value, defaults to 10. If a value is given, uses that value. Cannot be used with --all.')
    parser.add_argument('--all', action='store_true', help='If set, process all rows from the CSV file; otherwise, use the default 100-row CSV')
    parser.add_argument('--output-dir', type=str, default="output/kg_llm", help='Directory to write batch outputs')
    parser.add_argument('--batches', type=str, default=None, help='Comma-separated list of batches to process (e.g., 100,200,cum_200). If omitted, processes all batches.')
    args = parser.parse_args()

    if args.gpt4o and args.all and args.gpt4o_test_n is not None:
        print("Error: --gpt4o-test-n cannot be used with --all. If you want to process all rows, use only --gpt4o --all.")
        exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    # Read GS rows and get c5s
    gs_rows = read_rows(GS_CSV, n=100)
    gs_c5s = set(row['c5'] for row in gs_rows)
    # Read next 400 non-GS rows
    non_gs_rows = read_rows(NON_GS_CSV, skip_c5=gs_c5s, n=400)
    all_rows = gs_rows + non_gs_rows  # 500 rows

    # Define incremental and cumulative batches
    batch_ranges = [(0, 100), (100, 200), (200, 300), (300, 400), (400, 500)]
    batch_names = ["100", "200", "300", "400", "500"]
    cum_ranges = [(0, 200), (0, 300), (0, 400), (0, 500)]
    cum_names = ["cum_200", "cum_300", "cum_400", "cum_500"]
    all_batches = batch_ranges + cum_ranges
    all_names = batch_names + cum_names

    # Batch selection logic
    batch_name_to_range = dict(zip(all_names, all_batches))
    # Dependency map for cumulative batches
    cum_deps = {
        "cum_200": ["100", "200"],
        "cum_300": ["cum_200", "300"],
        "cum_400": ["cum_300", "400"],
        "cum_500": ["cum_400", "500"],
    }
    if args.batches:
        requested = [b.strip() for b in args.batches.split(",") if b.strip() in all_names]
        # Expand dependencies for cumulative batches
        def expand_batches(batches):
            expanded = set()
            def add_with_deps(b):
                if b in expanded:
                    return
                if b in cum_deps:
                    for dep in cum_deps[b]:
                        add_with_deps(dep)
                expanded.add(b)
            for b in batches:
                add_with_deps(b)
            return [b for b in all_names if b in expanded]
        selected_batches = expand_batches(requested)
    else:
        selected_batches = all_names

    # Efficient model loading: load once, reuse for all batches
    model_objs = None
    if not args.gpt4o:
        if args.size == 'small':
            models = SMALL_MODELS
        else:
            models = LARGE_MODELS
        model_objs = {}
        for shortname, modelname in models:
            print(f"Loading {modelname}...")
            model_objs[shortname] = load_model_and_tokenizer(modelname, shortname)

    # For cumulative node and triplet tracking
    all_nodes_so_far = set()
    all_triplets_so_far = set()
    # Store batch output paths for later cumulative concatenation
    batch_csv_paths = {}
    for (start, end), batch_name in zip(all_batches, all_names):
        if batch_name not in selected_batches:
            continue
        if batch_name.startswith("cum_"):
            continue  # We'll handle cumulative after all batches
        if args.gpt4o:
            batch_dir = os.path.join(args.output_dir, batch_name)
            os.makedirs(batch_dir, exist_ok=True)
            output_csv = os.path.join(batch_dir, f"gpt4o_withprevnodes_{batch_name}.csv")
            batch_csv_paths[batch_name] = output_csv
            print(f"Processing batch {batch_name} ({len(all_rows[start:end])} rows)...")
            batch_rows = all_rows[start:end]
            import importlib
            openai = importlib.import_module('openai')
            dotenv = importlib.import_module('dotenv')
            os_path = os.path
            dotenv.load_dotenv(os_path.join(os_path.dirname(os.path.dirname(__file__)), '.env'))
            openai_api_key = os.getenv("OPENAI_API_KEY")
            if not openai_api_key:
                raise ValueError("OPENAI_API_KEY environment variable not set. Please create a .env file in the kg directory with your OpenAI API key.")
            client = openai.OpenAI(api_key=openai_api_key)
            fieldnames = ["c5", "c119", "gpt4o_triplets", "gpt4o_triplets_clean"]
            total_prompt_tokens = 0
            total_completion_tokens = 0
            results = []
            node_str_cache = None
            last_nodes = None
            # Apply row limit for gpt4o if needed
            if (args.gpt4o and 'gpt4o_n' in locals() and gpt4o_n is not None) and (len(batch_rows) > gpt4o_n):
                batch_rows = batch_rows[:gpt4o_n]
            for row in tqdm(batch_rows, desc=f"Batch {batch_name}"):
                text = row['c119'] if 'c119' in row else next(iter(row.values()))
                current_nodes = tuple(sorted(all_nodes_so_far))
                if current_nodes != last_nodes:
                    node_str_cache = ', '.join(current_nodes) if current_nodes else '(none)'
                    last_nodes = current_nodes
                prompt = PROMPT_TEMPLATE.format(text=text, node_list=node_str_cache)
                try:
                    response = client.chat.completions.create(
                        model="gpt-4o",
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant that extracts knowledge graph triplets from text."},
                            {"role": "user", "content": prompt}
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
                    for e1, rel, e2 in parse_triplets(triplets):
                        all_nodes_so_far.add(e1)
                        all_nodes_so_far.add(e2)
                        all_triplets_so_far.add(f"<{e1}, {rel}, {e2}>")
                except Exception as e:
                    print(f"Error generating triplets: {str(e)}")
                    triplets = ""
                row_out = {
                    "c5": row['c5'],
                    "c119": text,
                    "gpt4o_triplets": triplets,
                    "gpt4o_triplets_clean": triplets
                }
                results.append(row_out)
            with open(output_csv, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for row_out in results:
                    writer.writerow(row_out)
            input_cost = total_prompt_tokens * 0.000005
            output_cost = total_completion_tokens * 0.000015
            total_cost = input_cost + output_cost
            n_rows = len(batch_rows)
            print(f"\n--- GPT-4o Cost Estimation for batch {batch_name} ({n_rows} rows) ---")
            print(f"Total prompt tokens: {total_prompt_tokens}")
            print(f"Total completion tokens: {total_completion_tokens}")
            print(f"Estimated input cost: ${input_cost:.4f}")
            print(f"Estimated output cost: ${output_cost:.4f}")
            print(f"Estimated total cost: ${total_cost:.4f}")
            print(f"Batch {batch_name} saved to {output_csv}")
        else:
            # For each model, create separate output directories
            for shortname, _ in models:
                model_output_dir = os.path.join(args.output_dir, f"{shortname}_with_nodes_batches")
                batch_dir = os.path.join(model_output_dir, batch_name)
                os.makedirs(batch_dir, exist_ok=True)
                output_csv = os.path.join(batch_dir, f"llm_with_existing_nodes_{shortname}_{batch_name}.csv")
                batch_csv_paths[(shortname, batch_name)] = output_csv
                print(f"Processing batch {batch_name} for model {shortname} ({len(all_rows[start:end])} rows)...")
                fieldnames = ["c5", "c119", f"{shortname}_triplets", f"{shortname}_triplets_clean"]
                batch_rows = all_rows[start:end] if batch_name == "100" else non_gs_rows[start-100:end-100]
                results = []
                last_nodes = None
                node_str_cache = None
                for row in tqdm(batch_rows, desc=f"Batch {batch_name} [{shortname}]"):
                    text = row['c119'] if 'c119' in row else next(iter(row.values()))
                    current_nodes = tuple(sorted(all_nodes_so_far))
                    if current_nodes != last_nodes:
                        node_str_cache = ', '.join(current_nodes) if current_nodes else '(none)'
                        last_nodes = current_nodes
                    prompt = PROMPT_TEMPLATE.format(text=text, node_list=node_str_cache)
                    tokenizer_or_processor, model = model_objs[shortname]
                    try:
                        raw_output = generate_triplets(prompt, tokenizer_or_processor, model, shortname)
                    except Exception as e:
                        raw_output = ""
                    triplets_clean = extract_triplets_only(raw_output)
                    for e1, rel, e2 in parse_triplets(raw_output):
                        all_nodes_so_far.add(e1)
                        all_nodes_so_far.add(e2)
                        all_triplets_so_far.add(f"<{e1}, {rel}, {e2}>")
                    row_out = {
                        "c5": row['c5'],
                        "c119": text,
                        f"{shortname}_triplets": raw_output,
                        f"{shortname}_triplets_clean": triplets_clean
                    }
                    results.append(row_out)
                with open(output_csv, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    for row_out in results:
                        writer.writerow(row_out)
                print(f"Batch {batch_name} for model {shortname} saved to {output_csv}")

    # Now create cumulative CSVs by concatenating previous batch CSVs
    def concat_csvs(csv_paths, out_path):
        import shutil
        with open(out_path, 'w', encoding='utf-8', newline='') as fout:
            writer = None
            for i, path in enumerate(csv_paths):
                with open(path, 'r', encoding='utf-8') as fin:
                    reader = csv.reader(fin)
                    header = next(reader)
                    if writer is None:
                        writer = csv.writer(fout)
                        writer.writerow(header)
                    for row in reader:
                        writer.writerow(row)
    # Cumulative logic
    cum_map = [
        ("cum_200", ["100", "200"]),
        ("cum_300", ["cum_200", "300"]),
        ("cum_400", ["cum_300", "400"]),
        ("cum_500", ["cum_400", "500"]),
    ]
    for cum_name, parts in cum_map:
        if cum_name not in selected_batches:
            continue
        if args.gpt4o:
            batch_dir = os.path.join(args.output_dir, cum_name)
            os.makedirs(batch_dir, exist_ok=True)
            out_csv = os.path.join(batch_dir, f"gpt4o_withprevnodes_{cum_name}.csv")
            part_paths = [os.path.join(args.output_dir, p, f"gpt4o_withprevnodes_{p}.csv") for p in parts]
            concat_csvs(part_paths, out_csv)
            print(f"Cumulative batch {cum_name} saved to {out_csv}")
        else:
            for shortname, _ in models:
                model_output_dir = os.path.join(args.output_dir, f"{shortname}_with_nodes_batches")
                batch_dir = os.path.join(model_output_dir, cum_name)
                os.makedirs(batch_dir, exist_ok=True)
                out_csv = os.path.join(batch_dir, f"llm_with_existing_nodes_{shortname}_{cum_name}.csv")
                part_paths = [os.path.join(model_output_dir, p, f"llm_with_existing_nodes_{shortname}_{p}.csv") for p in parts]
                concat_csvs(part_paths, out_csv)
                print(f"Cumulative batch {cum_name} for model {shortname} saved to {out_csv}")

if __name__ == "__main__":
    """
    Usage:
      python generate_kg_triplets_with_nodes.py [--size small|large|--gpt4o] [--all] [--gpt4o-test-n [N]]

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

    Batch logic:
      - 9 batches: 100 GS, 100 next, ... up to 500, and cumulative 200, 300, 400, 500.
      - For each batch, the prompt includes all entities (nodes) found in previous batches' extracted triplets.
      - This encourages the model to normalize entity mentions to existing nodes, improving consistency and reducing redundancy in the knowledge graph.

    Cost estimation:
      - For --gpt4o, the script prints token usage and estimated cost for each batch.

    Output:
      - 9 CSV files, one for each batch, in the specified output directory.
      - Each CSV contains columns: c5, c119, {model}_triplets, {model}_triplets_clean

    Example usage:
        Run all batches with GPT-4o:
        python generate_kg_triplets_with_nodes.py --gpt4o
    
        Run only the 100 and 200 batches (small models):
        python generate_kg_triplets_with_nodes.py --size small --batches 100,200
    
        Run only the cumulative 300 batch (and its dependencies):
        python generate_kg_triplets_with_nodes.py --size small --batches cum_300
    """
    main() 