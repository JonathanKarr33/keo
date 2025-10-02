# Knowledge Graph (kg) Directory

This directory contains scripts and resources for generating, normalizing, and analyzing knowledge graphs (KGs) from text using large language models (LLMs) and entity normalization tools.

## Main Scripts

### 1. `generate_kg_triplets_with_nodes.py`
- **Purpose:** Extracts knowledge graph triplets from text using LLMs (e.g., Gemma, Phi, GPT-4o), incrementally building a set of known nodes to encourage entity normalization.
- **Input:** CSV files with text (see `GS_CSV` and `NON_GS_CSV` in the script).
- **Output:** Batch and cumulative CSVs with extracted triplets, organized by model and batch in `output/kg_llm/`.
- **Usage:**
  ```bash
  # Run on a specific GPU (e.g., GPU 1) for Gemma 4B
  python generate_kg_triplets_with_nodes.py --size small --model-shortname gemma3_4b_it --gpu 1
  # Run in background and log output
  nohup python generate_kg_triplets_with_nodes.py --size small --model-shortname gemma3_4b_it --gpu 1 > gemma4b_gpu1.out 2>&1 &
  ```
- **Parallelism:** Launch multiple processes with different `--gpu` values for parallel batch processing on multiple GPUs.

### 2. `fix_entity_mentions.py`
- **Purpose:** Normalizes entity mentions in triplet CSVs using SBERT-based semantic similarity, reducing redundancy in the KG.
- **Input:** Batch CSVs in `output/kg_llm/*_with_nodes_batches/`.
- **Output:** Fixed CSVs with `_fixed.csv` suffix and a replacements mapping CSV.
- **Usage:**
  ```bash
  python fix_entity_mentions.py --input output/kg_llm/
  # To force reprocessing even if fixed files exist:
  python fix_entity_mentions.py --input output/kg_llm/ --force
  ```
- **Note:** The script supports both `_withprevnodes_` and `llm_with_existing_nodes_` filename patterns.

### 3. `generate_fixed_kg.py`
- **Purpose:** Converts fixed triplet CSVs into graph files (GML), PNG visualizations, and node summary CSVs.
- **Input:** Fixed CSVs from `fix_entity_mentions.py`.
- **Output:** `.gml`, `.png`, and `_nodes.csv` files in the same directory as the input CSV.
- **Usage:**
  ```bash
  python generate_fixed_kg.py --input path/to/your_fixed.csv
  ```

## Directory Structure
- `output/kg_llm/` — Output directory for all batch and cumulative CSVs, organized by model and batch.
- `output/kg_llm/<model>_with_nodes_batches/<batch>/llm_with_existing_nodes_<model>_<batch>.csv` — Main output CSVs for each batch/model.
- `output/kg_llm/<model>_with_nodes_batches/<batch>/llm_with_existing_nodes_<model>_<batch>_fixed.csv` — Entity-normalized CSVs.

## Notes
- Make sure to install all required dependencies (see script headers for requirements).
- For Gemma models, you may need to log in to HuggingFace and accept the model license.
- For multi-GPU parallelism, launch one process per GPU with the appropriate `--gpu` argument.

---

For questions or issues, see the script docstrings or contact the repository maintainer. 