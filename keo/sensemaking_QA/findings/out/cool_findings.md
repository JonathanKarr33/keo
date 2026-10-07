# KEO Cool Findings (offline mine)

Generated: 2026-09-17T16:39:28.981903

## DoD gap (why this exists)
### Data gravity / accreditation
- **Problem:** Sensitive maintenance & military-adjacent text cannot freely hit commercial LLM APIs (FedRAMP/ITAR/FISMA).
- **KEO:** Operational answers on local Gemma/Phi/Mistral; GPT-4o only offline for KG/Q/judge; optional local embeddings.

### Gov API capacity vs mission tempo
- **Problem:** Army Enterprise LLM Workspace / Azure OpenAI Gov help but are limited relative to volume and classification needs.
- **KEO:** On-prem / air-gapped path with structured retrieval so smaller local models punch above weight.

### Wrong retrieval for the wrong job
- **Problem:** One RAG style fails: graphs help corpus sensemaking; chunks help procedural action.
- **KEO:** Dual benchmark + task router (sensemaking→KG, action→chunks, mixed→hybrid).

### Auditability for safety decisions
- **Problem:** Black-box answers are unacceptable for airworthiness / maintenance advice.
- **KEO:** Evidence cards with triple/chunk IDs and retrieval confidence for human verification.

### Rare but critical faults
- **Problem:** Frequency-weighted graphs bury low-count, high-severity relations.
- **KEO:** Rarity/severity edge weights + MST ablations.

## Why others should use this

- Any org with regulated data (defense, aviation, energy, healthcare) that cannot ship prompts to public APIs.
- Teams that need both dashboard-style sensemaking and step-level procedures from the same corpus.
- Operators who need citation-like evidence, not just fluent text.
- Researchers needing an OMIn-grounded aviation QA split that exposes the graph-vs-chunk tradeoff.

## Headline takeaways

- gemma-3-27b-it: KEO beats text-chunk on sensemaking by 0.61 overall (mean over mined runs).
- mistral-nemo-12b-it: KEO beats text-chunk on sensemaking by 0.48 overall (mean over mined runs).
- phi-4-14b: KEO beats text-chunk on sensemaking by 0.41 overall (mean over mined runs).
- gemma-3-it: oracle task routing scores 4.133 vs always-KEO 4.141 (-0.008) and always-chunk 4.015 (+0.119).
- phi-4: oracle task routing scores 4.12 vs always-KEO 4.041 (+0.079) and always-chunk 4.002 (+0.119).
- mistral-nemo-it: oracle task routing scores 3.814 vs always-KEO 3.799 (+0.015) and always-chunk 3.795 (+0.019).
- 4/5 high-severity edges and 51/60 safety-relevant edges fall outside the top 20% by frequency — frequency-only MST can under-prioritize rare but safety-critical relations.

## Oracle router (from paper tables)

| Model | Always KEO | Always chunk | Oracle routed | Δ vs KEO |
|-------|------------|--------------|---------------|----------|
| gemma-3-it | 4.141 | 4.015 | 4.133 | -0.008 |
| phi-4 | 4.041 | 4.002 | 4.12 | +0.079 |
| mistral-nemo-it | 3.799 | 3.795 | 3.814 | +0.015 |

## Rarity demo
4/5 high-severity edges and 51/60 safety-relevant edges fall outside the top 20% by frequency — frequency-only MST can under-prioritize rare but safety-critical relations.
