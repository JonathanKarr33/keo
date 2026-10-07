# Second-Domain Transfer Pilot

Addresses AC request to widen beyond aviation-OMIn specificity.

## Design
1. Use a **public** narrative maintenance/safety corpus (default target: NASA ASRS sample or any CSV with a free-text column).
2. Build a small KG (≤200 records) with the existing LLM triplet scripts or a lightweight heuristic extractor.
3. Reuse KEO ablation flags (`--method`, `--use-mst`, `--edge-weight`) on 10–20 hand-written sensemaking prompts.
4. Report as a **pilot** in Appendix `\ref{sec:second_domain}` — not a full second benchmark.

## Layout
```
keo/second_domain/
  README.md            # this file
  data/                # place public CSV here (gitignored if sensitive)
  prompts.json         # starter sensemaking prompts
  build_kg_stub.py     # CSV → toy NetworkX GML
  run_pilot.sh         # wraps generate_ablation_answers.py
```

## Quick start
```bash
# 1) Drop a CSV with column `narrative` into data/sample.csv
python build_kg_stub.py --csv data/sample.csv --out data/pilot_kg.gml --text-col narrative

# 2) Run KEO on pilot prompts against the local Phi-4 server (hash embeddings by default —
#    use EMB=local when MiniLM is free). Override MODEL/PROVIDER for cloud.
bash run_pilot.sh --kg data/pilot_kg.gml --sample-size 5
# ANSWER_MODEL=mlx-community/phi-4-4bit PROVIDER=local bash run_pilot.sh ...
```

## Status
Scaffolding + executed pilot (2026-09-26): 102 ASRS-*style* synthetic narratives
(not the NASA dump; HF `NASA-AIML/ASRS` was gated), co-occurrence KG (40 nodes /
413 edges), local Phi-4 answers on 5 sensemaking prompts. Paper appendix
`\ref{sec:second_domain}` reports plumbing success, not a judged transfer score.

To swap in real ASRS CSV later: drop a file with a free-text column into `data/`,
rebuild with `build_kg_stub.py`, re-run `run_pilot.sh`.