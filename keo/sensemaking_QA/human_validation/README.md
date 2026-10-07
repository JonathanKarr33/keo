# Human Validation Protocol (KEO)

## Answer calibration (primary)

Blind pairwise ratings of Phi-4 **KEO vs length-matched `chunk_tok`** on $n{=}40$ sensemaking questions. Sides randomized; system key is separate.

**Give each primary annotator**

1. `ratings/answer_calibration_RUBRIC.txt`
2. Website: http://127.0.0.1:8765/ (Annotator 1 or 2)

**Adjudication (after both finish)**

- Login as Adjudicator (password offline) → http://127.0.0.1:8765/adjudicate.html
- Only contested items (system preference differs)
- Resolve as solo adjudicator **or** team conference
- Consensus → `ratings/human_pref_consensus.json`

**Do not give annotators** `answer_calibration_n40_key.json`.

Humans choose Prefer A / Prefer B / Neither — not the five Claude rubric dimensions. After adjudication, unblind for judge–human agreement. Do not pool human scores into the Haiku/Sonnet mechanism tables.

Typical time: ~3–5 min/item (~2–3.5 h for 40).

```bash
# regenerate pack (no model calls)
python export_annotator_sheets.py answers
```

## Severity expert sheet (on the website)

Open **Severity** at http://127.0.0.1:8765/severity.html after signing in.
Pack: `ratings/severity_expert_n100.json` (class labels only; lexicon prior stripped).
Key (offline only): `ratings/severity_expert_n100_key.json`.
Votes: `severity_votes_annotator{1,2,3}.json` → `severity_consensus.json`.
Do **not** show annotators the prior class.

## Legacy question/triple batch

`sample_items.py` + `schema.json` cover clarity/difficulty/plausibility sheets. An interim Haiku fill exists; it is **not** an independent check of the matrix judge. Prefer the answer-calibration pack above for the paper’s human gate.

## Reporting

Report preference agreement, quadratic-weighted κ on overall, consensus rule for disagreements, then human–Haiku and human–Sonnet agreement after unblinding. Paste into Appendix `\ref{sec:human_val}`. Do not claim gold-standard status for auto-generated items.
