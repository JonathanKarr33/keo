#!/usr/bin/env python3
"""
Verify the answer model actually serves a completion before a sweep spends hours.

Motivation: a smoke run of the full ablation matrix generated 11 arms, judged 3 of them,
and produced `overall_avg=1.000` tables — all because the requested model ID was not
served by the provider. Every arm stored
"Error generating answer: Model <id> is not supported by provider <p>" as its answer.
One 8-token request up front turns that into a five-second, unambiguous failure.

The check deliberately bypasses the completion cache so a cached success from an earlier
model cannot mask a broken configuration.

The judge is checked the same way, and for a second reason. Unreadable rubric output is now
refused rather than imputed as a midpoint, so a judge whose response format has drifted
aborts the run — better to learn that from one probe than after generation is paid for. The
probe therefore goes through the real rubric prompt and the real parser, not a bare "reply
with ready", because only that exercises the format.

Usage:
  python preflight_model.py --answer-model MODEL --provider huggingface --API-provider together
  python preflight_model.py --judge-model haiku --judge-provider claude_cli
  python preflight_model.py --answer-model ... --judge-model ...   # both
Exit codes: 0 usable, 1 unusable (message on stderr), 2 bad invocation.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from answer_generator import SensemakingAnswerGenerator

PROBE = [
    {"role": "user", "content": "Reply with the single word: ready"},
]


def resolve_api_key(provider: str) -> str:
    """Mirror the key lookup the generators use, with a clear message when absent."""
    if provider == "local":
        return os.getenv("LOCAL_LLM_API_KEY", "EMPTY")
    if provider == "openrouter":
        from openrouter_client import openrouter_api_key

        return openrouter_api_key()
    env_var = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "huggingface": "HF_TOKEN",
    }.get(provider)
    if env_var is None:
        raise RuntimeError(f"unsupported provider for preflight: {provider}")
    key = os.getenv(env_var)
    if not key:
        raise RuntimeError(f"{env_var} is not set (required for provider={provider})")
    return key


def preflight(model: str, provider: str, api_provider: str, base_url: str | None) -> str:
    """Return the probe completion, or raise with the provider's own error text."""
    generator = SensemakingAnswerGenerator(
        api_key=resolve_api_key(provider),
        model=model,
        provider=provider,
        API_provider=api_provider,
        base_url=base_url,
        embedding_backend="hash",   # no embedder needed to probe the LLM
        use_completion_cache=False,  # a stale hit would defeat the whole check
        emit_evidence=False,
    )
    return generator._create_chat_completion(PROBE, max_tokens=8, temperature=0.0)


# A question/answer pair short enough to be cheap but real enough that the judge fills in
# every criterion of the rubric.
JUDGE_PROBE_QUESTION = {
    "id": "preflight",
    "question": "What recurring themes appear across these maintenance records?",
}
JUDGE_PROBE_ANSWER = {
    "question_id": "preflight",
    "answer": (
        "Two themes recur. First, fuel contamination from water ingress appears across "
        "several airframes, usually reported after cold-weather operations. Second, "
        "hydraulic leaks cluster on landing gear assemblies, often following hard landings."
    ),
}


def preflight_judge(model: str, provider: str, api_provider: str) -> dict:
    """Exercise the real rubric prompt and parser; raise if either is unusable."""
    from evaluator import SensemakingEvaluator

    evaluator = SensemakingEvaluator(
        api_key=resolve_api_key(provider) if provider != "claude_cli" else "dummy",
        model=model,
        provider=provider,
        API_provider=api_provider,
    )
    result = evaluator._evaluate_global_sensemaking_answer(
        JUDGE_PROBE_ANSWER, JUDGE_PROBE_QUESTION
    )
    scores = result.get("global_metrics") or {}
    expected = ["global_perspective", "theme_identification", "synthesis_quality",
                "strategic_value", "pattern_recognition"]
    missing = [c for c in expected if c not in scores]
    if missing:
        raise RuntimeError(
            f"judge replied but the rubric did not parse; missing {missing}. "
            f"Raw reply: {str(result.get('evaluation_text'))[:400]!r}"
        )
    return {"scores": scores, "exact": result.get("evaluator_model_exact")}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--answer-model", default=None)
    parser.add_argument("--provider", default="huggingface")
    parser.add_argument("--API-provider", dest="api_provider", default="together")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--judge-model", default=None,
                        help="Also preflight the judge (e.g. haiku)")
    parser.add_argument("--judge-provider", default="claude_cli")
    args = parser.parse_args()

    if not args.answer_model and not args.judge_model:
        parser.error("give --answer-model and/or --judge-model")

    failures = 0

    if args.answer_model:
        target = f"{args.answer_model} via {args.provider}"
        if args.provider == "huggingface":
            target += f"/{args.api_provider}"
        try:
            reply = preflight(args.answer_model, args.provider, args.api_provider,
                              args.base_url)
            if not reply.strip():
                raise RuntimeError("returned an empty completion")
            print(f"✓ Preflight OK: {target} → {reply.strip()[:40]!r}")
        except Exception as e:
            print(f"PREFLIGHT FAILED (answers): {target}", file=sys.stderr)
            print(f"  {type(e).__name__}: {e}", file=sys.stderr)
            print(
                "  Nothing was generated. Fix the model ID or provider before running the "
                "sweep, otherwise every arm records the error text as its answer.",
                file=sys.stderr,
            )
            failures += 1

    if args.judge_model:
        target = f"{args.judge_model} via {args.judge_provider}"
        try:
            info = preflight_judge(args.judge_model, args.judge_provider, args.api_provider)
            got = ", ".join(f"{k}={v}" for k, v in sorted(info["scores"].items())
                            if k != "total_evaluated")
            print(f"✓ Preflight OK: judge {target} (exact={info['exact']}) parsed {got}")
        except Exception as e:
            print(f"PREFLIGHT FAILED (judge): {target}", file=sys.stderr)
            print(f"  {type(e).__name__}: {e}", file=sys.stderr)
            print(
                "  The judge is unusable or its rubric output no longer parses. Imputing "
                "midpoints is refused, so the sweep would abort after paying for "
                "generation. Fix the judge first.",
                file=sys.stderr,
            )
            failures += 1

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
