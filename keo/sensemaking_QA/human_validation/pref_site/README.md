# KEO human-validation site

```bash
cd keo/sensemaking_QA/human_validation/pref_site
python3 serve.py
```

| Page | URL | Who |
|------|-----|-----|
| Sign in | http://127.0.0.1:8765/ | all |
| Prefs | http://127.0.0.1:8765/rate.html | Annotators 1–2 |
| Severity | http://127.0.0.1:8765/severity.html | Annotators 1–2 (+ 3 on conflicts) |
| Adjudicate | http://127.0.0.1:8765/adjudicate.html | Adjudicator (answer prefs) |
| Status | http://127.0.0.1:8765/status.html | live progress |

**Passwords (offline):** Annotator 1 `keo` · Annotator 2 `eok` · Adjudicator `adj`

## Answer preferences
1. Primaries rate all 40 items (Prefer A / B / Neither). Sides randomized; systems hidden.
2. Contested *system* prefs → **Adjudicate**.
3. Resolve as solo adjudicator or team conference.
4. Consensus → `human_pref_consensus.json`

## Severity / risk triples
1. Primaries open **Severity** and label each triple: catastrophic / hazardous / major / minor / none.
2. Lexicon prior is **never** shown (`severity_expert_n100_key.json` stays offline).
3. Class disagreements → Adjudicator on the Severity page (contested queue).
4. Consensus → `severity_consensus.json`; primaries also mirrored into `severity_expert_n100.json`.

**Outputs**
- Prefs: `../ratings/human_pref_votes_annotator{1,2,3}.json`, `human_pref_labels.csv`, `human_pref_consensus.json`
- Severity: `../ratings/severity_votes_annotator{1,2,3}.json`, `severity_labels.csv`, `severity_consensus.json`
