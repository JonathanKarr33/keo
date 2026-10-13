# KEO TODO

Do not pool Gemma and Phi, or Haiku and Sonnet. Do not invent judged numbers. Launch scripts plan until `--execute`.

**Updated:** 2026-10-12 22:05. Preference adjudication is done. Final labels: 23 KEO / 4 chunk / 13 neither. Primaries still 28–6–6 and 22–7–11; agreement 22/40 (κ≈0.18). The adjudicator resolved all 18 disagreements: neither on 11, KEO on 5, chunk on 2. Ann2 severity is still 0. Jev stays out of the paper.

## Before October 12

- [ ] **Edit Figure 1** (`fig:KEO_overview`, `papers/latex/figures/workflow.pdf`). Overview of KG creation, graph RAG, chunk baseline, benchmark, and judges.
- [x] **Preference adjudication.** Final 23 KEO / 4 chunk / 13 neither. Ann2 severity is still 0/100, so the safety claim stays locked.
  - `cd keo/sensemaking_QA/human_validation/pref_site && python3 serve.py` → [http://127.0.0.1:8765/](http://127.0.0.1:8765/)
  - Annotator **1** / `keo`; Annotator **2** / `eok`; Adjudicator **3** / `adj`
  - Prefs → `human_pref_votes_annotator{1,2}.json`; severity → `severity_votes_annotator{1,2}.json`
  - Status: [http://127.0.0.1:8765/status.html](http://127.0.0.1:8765/status.html)

## Open

- [x] **Sonnet × Gemma-4 31B** — complete (`openrouter_gemma4_31b_sonnet/`, 7/7). KEO 2.692; `chunk_tok` $+0.055$ n.s.\ (win 36%); `chunk_sum` $+0.169$. Paired stats + `tab:controls-31b-sonnet`; forest extended. Do not pool with Haiku.
- [x] Offline **specificity audit** (`specificity_audit.py`) + interim Ann1 lean (8 KEO / 1 chunk / 1 tie on 10/40); wired into Sec.\ specificity + appendix table.
- [x] **Jev on OpenRouter, both styles** (`typesafe/jev-1.13`, `jev_judge.py`). 34 `jev_holistic.json` + 18 `jev_sourcecard.json`, 83/83 each, $0.16. Claude files untouched. Not comparable to Haiku, Sonnet, or GPT, and not a paper result: holistic sits near 4 and does not separate KEO from the shuffle; grounded coverage vs shuffle is the only consistent lean (E4B +0.275, Phi +0.280, 31B +0.685, labeled +0.370) and it disappears against the no-relation arm. Attribution stays near the floor (E4B 1.34). Chunk arms have holistic scores only. GPT-4.1 for this same 4,316 calls would be about $40 ($20 on the Batch API); do not run it for the draft.
- [ ] **Claude Grounded / Source-card Haiku — do not resume for this draft.** Limit resets 1:40am America/Indianapolis. Already-scored items are skipped if it is restarted later. Chunk arms have no stored text. Finished E4B graph arms do not change the claim (attribution vs shuffle is a tie; coverage vs shuffle +0.185 on n=81). GPT runs only Holistic / Sensemaking.
  ```bash
  cd keo/sensemaking_QA
  KEO_JUDGE_WORKERS=16 ../../.venv/bin/python -u grounded_judge.py --current --workers 16
  ```
  Done 83/83: `_gemma4e4b` norel, shuffle s13, `mst_on_freq`; `phi4/ctrl_keo_norel`. Partial: `phi4/ctrl_keo_shuf_s13` 49/83. Other 13 graph arms not started.
- [ ] Expert severity consensus (`severity_expert_n100.json`) after Ann2, then replace the Haiku proxy and rerun `score_adjudicated_critical_recall.py`. Not required for the October 12 upload; the paper already treats lexicon recall as circular until then.
- [ ] Post this draft as the next version on the **same arXiv id**, with the **original title**, after meta-reviews if the anonymity incentive is kept. Do not cite a prior version in the paper. Do not open a new arXiv id.
- [ ] Page budget: body already ends on page 8; appendix after the references does not count. Leave Jev and Claude source-card scores out of the tables.
- [ ] Optional: full (non-4-bit) Phi-4; interventional entity yield; question-conditioned MST-vs-risk; expand n>83; fresh `HF_TOKEN` only if regenerating archived rows.
- [ ] **Later (compute):** regenerate the labeled Phi-4 seven-arm matrix with MiniLM, new run id `labeled_graph_minilm` (do not overwrite hash `labeled_graph`). Stopped 2026-09-30 after a partial `mst_on_freq` (answers_meta already shows `all-MiniLM-L6-v2`). Resume when the machine is free; judge only after generation, own table, do not pool with the hash run. Paper limitations still say this rerun has not been judged.
  ```bash
  cd keo/sensemaking_QA
  # keep the mlx server alive in the same session (do not use --detach)
  ../../.venv-mlx/bin/python -m mlx_lm server --model mlx-community/phi-4-4bit --port 8000
  SKIP_OFFLINE_TESTS=1 RUN_ID=labeled_graph_minilm bash run_labeled_controls.sh --execute
  ```

## ACL Findings readiness (offline done this pass)

- [x] Cross-stack **forest plot** of `chunk_tok`−KEO (Fig `fig:chunk_forest`) + win rates in text.
- [x] Per-rubric **Haiku–Sonnet dim flips** (Fig `fig:judge_dims`, `tab:dim-flips`).
- [x] **Answer-length residual** after context matching (Fig `fig:length_resid`, `sec:length_resid`).
- [x] Findings framing: abstract lead + Conclusion “What a Graph-RAG paper should report.”
- [x] Regenerable assets: `keo/sensemaking_QA/make_acl_assets.py` → `papers/latex/figures/*`, `papers/latex/generated/*`, `evaluation_results/stats/acl_cross_stack_summary.json`.
- [x] PDF: `papers/KEO_paper.pdf`.

## Done (prior)

- [x] Sonnet seven-arm on Phi-4 + Gemma E4B (sign flips). Haiku×31B matrix. Labeled-graph Phi-4. K2A NLP. Human pack protocol. TikZ overview. Judge agreement appendix. `run_evaluation.py` exit 4 on incomplete global.

## Headline numbers (do not invent beyond these)


| Stack                             | KEO       | `chunk_tok` Δ | chunk win rate | Note                 |
| --------------------------------- | --------- | ------------- | -------------- | -------------------- |
| Gemma E4B / Haiku / unlabeled     | ~2.27     | **+0.278**    | 53%            | detectable chunk win |
| Gemma E4B / Sonnet / same answers | 2.248     | **−0.108**    | 21%            | sign flip            |
| Phi-4 / Haiku / unlabeled         | 2.230     | +0.060        | 45%            | underpowered         |
| Phi-4 / Sonnet / same answers     | 2.292     | **−0.099**    | 14%            | sign flip            |
| Phi-4 / Haiku / labeled 1468e     | 2.245     | −0.042        | 43%            | structure n.s.       |
| Gemma 31B / Haiku / unlabeled     | **3.219** | **−0.072**    | 40%            | capacity; gap n.s.   |
| Gemma 31B / Sonnet / same answers | **2.692** | **+0.055**    | 36%            | capacity; gap n.s.   |


Δ = other arm − KEO. Never pool generators, judges, or graphs.

### How to run

```bash
# Regenerable ACL figures/tables (no LLM)
cd keo/sensemaking_QA && ../../.venv/bin/python make_acl_assets.py

# Sonnet resume on 31B (after quota)
export KEO_JUDGE_WORKERS=8 KEO_ACTION_NLP_ONLY=1
# run_evaluation.py --final per incomplete arm under openrouter_gemma4_31b_sonnet/

# Paper
bash papers/compile_pdf.sh
```

