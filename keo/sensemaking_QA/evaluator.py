"""
Evaluation Framework for Sensemaking Questions and Answers
Based on GraphRAG evaluation methodology with qualitative metrics
Enhanced with ground truth evaluation for action-specific questions
"""

import os
import json
import pandas as pd
from typing import List, Dict, Optional, Tuple, Any, Callable
from openai import OpenAI
from tqdm import tqdm
import numpy as np
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import re
from collections import Counter

from model_provenance import build_run_provenance, observe_completion_model
from anthropic_client import (
    DEFAULT_JUDGE_MODEL,
    DEFAULT_JUDGE_TEMPERATURE,
    anthropic_chat,
    make_anthropic_client,
)
from claude_cli import (
    ClaudeAuthError,
    ClaudeNotFoundError,
    ClaudeQuotaError,
    claude_cli_chat,
    claude_code_version_text,
    ensure_logged_in,
)
from judge_policy import (
    GRADE_PAPER,
    PAPER_PROVIDER,
    infer_grade,
    judge_meta,
    warn_if_non_paper,
)
from judge_styles import HOLISTIC_ID, HOLISTIC_NAMES

try:
    from huggingface_hub import InferenceClient
    HUGGINGFACE_AVAILABLE = True
except ImportError:
    HUGGINGFACE_AVAILABLE = False

# NLP evaluation metrics imports
try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from nltk.translate.meteor_score import meteor_score
    from rouge_score import rouge_scorer
    import nltk
    # Download required NLTK data
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet')
    NLP_METRICS_AVAILABLE = True
    NLP_METRICS_MISSING = ""
except ImportError as e:
    NLP_METRICS_AVAILABLE = False
    NLP_METRICS_MISSING = str(e)

# Without nltk/rouge-score every BLEU/METEOR/ROUGE value is silently 0.0, which is
# indistinguishable from a genuinely terrible score once it reaches a table. 13 earlier
# runs were reported that way. Refuse by default; set KEO_ALLOW_MISSING_NLP=1 to opt out.
ALLOW_MISSING_NLP = os.getenv("KEO_ALLOW_MISSING_NLP", "").strip().lower() in ("1", "true", "yes")
# Skip the LLM leg of action scoring (BLEU/METEOR/ROUGE + exact match only). Use when
# Claude quota is tight or for a fast matrix over already-generated answers.
ACTION_NLP_ONLY = os.getenv("KEO_ACTION_NLP_ONLY", "").strip().lower() in ("1", "true", "yes")


def require_nlp_metrics() -> None:
    """Abort before action-specific metrics are computed as structural zeros."""
    if NLP_METRICS_AVAILABLE:
        return
    message = (
        f"NLP metric libraries unavailable ({NLP_METRICS_MISSING}). "
        "BLEU/METEOR/ROUGE would be recorded as 0.0 and are indistinguishable from real "
        "scores in the output. Install them:\n"
        "    pip install -r requirements.txt   # or: pip install nltk rouge-score\n"
        "To proceed anyway (LLM scores only, NOT reportable), set KEO_ALLOW_MISSING_NLP=1."
    )
    if ALLOW_MISSING_NLP:
        print(f"WARNING: {message}")
        return
    raise RuntimeError(message)


# Judge scores are defined on a 1-5 scale. A number outside it did not come from the
# rubric — it is a parse artefact (a year, a count, "4/5" read as 45) and must not be
# averaged as if it were a rating.
JUDGE_SCALE_MIN = 1.0
JUDGE_SCALE_MAX = 5.0

# When a criterion cannot be read, the parsers substitute the midpoint of the scale. That
# is the most dangerous default available: it pulls every arm toward 3.0, which *shrinks
# differences between arms*. The mechanism controls are interpreted through null results
# ("shuffling edges changes nothing"), so imputed midpoints manufacture exactly the
# finding the paper wants to read.
#
# Refused by default, following require_nlp_metrics(): a run that quietly substitutes
# midpoints is worse than one that stops. Failing fast is also cheap — a prompt/format
# mismatch shows up on the first judged response, so this costs one API call rather than a
# whole sweep. Set KEO_ALLOW_JUDGE_IMPUTATION=1 to fall back to midpoints (NOT reportable).
ALLOW_JUDGE_IMPUTATION = os.getenv("KEO_ALLOW_JUDGE_IMPUTATION", "").strip().lower() in (
    "1", "true", "yes",
)

# Only rubrics that feed *between-arm* comparisons refuse. Question-quality scoring rates
# the benchmark's questions rather than comparing methods, so an unreadable criterion there
# cannot compress an arm difference — it is counted but tolerated.
COMPARISON_RUBRICS = frozenset({"global", "answer", "action"})


def judge_workers() -> int:
    """How many judge calls to have in flight at once.

    Each call is a `claude` subprocess that spends almost all its time waiting on the network, so
    threads give near-linear speedup and the GIL is not the constraint. Measured on a real arm:
    18.4 s per judged response sequentially versus 2.9 s at 6 workers, a 6.3x speedup with
    imputation still at 0.0%.

    Per-call output does not change — the model snapshot is pinned and each call is independent with
    no shared conversation — so only the wall clock moves. The count is recorded in provenance as
    `judge_parse.judge_workers`. Set KEO_JUDGE_WORKERS=1 to reproduce a sequential run, or lower it
    if a subscription rate window is being hit: concurrency does not change how many calls a sweep
    makes, but it does spend them sooner.
    """
    raw = os.getenv("KEO_JUDGE_WORKERS", "6").strip()
    try:
        workers = int(raw)
    except ValueError:
        return 1
    return max(1, min(workers, 16))


class JudgeParseError(RuntimeError):
    """Judge output could not be read and imputing a midpoint was refused.

    A distinct type because the per-answer handlers catch broad Exceptions to keep one bad
    item from killing a sweep. Without this, refusing to impute would only drop the item
    and shrink n silently — the same class of bug it exists to prevent.
    """

# Above this share of imputed criteria the judge's output format has probably changed and
# the run should not be trusted. Only reachable when imputation is explicitly allowed.
JUDGE_IMPUTATION_ALARM = 0.10

_MARKDOWN_NOISE = str.maketrans("", "", "*`_#")


def _normalize_criterion_line(line: str) -> str:
    """
    Strip list/emphasis markup so '- **Faithfulness**: 4' matches the bare criterion.

    Claude routinely formats rubric output as markdown. The original parsers used a
    case-sensitive `startswith` on the raw line, so a single bold marker made every
    criterion unmatchable — and the failure mode was silent midpoint imputation rather
    than an error.
    """
    return line.translate(_MARKDOWN_NOISE).strip().lstrip("-•> ").strip()


def parse_criterion_scores(evaluation_text: str, criteria: List[str]) -> Tuple[Dict, Dict]:
    """
    Read one score per criterion out of free-form judge text.

    Returns (scores, report). `scores` holds only criteria that yielded a number inside
    the rubric's scale, keyed by the criterion exactly as supplied. `report` explains
    every criterion that did not, so callers can surface a broken judge format instead of
    quietly filling in midpoints.

    Later mentions win, matching the original parsers: judges often restate a score in a
    summary line, and that trailing value was the one previously kept.
    """
    scores: Dict[str, float] = {}
    out_of_range: List[Tuple[str, float]] = []
    no_number: List[str] = []

    for raw_line in evaluation_text.split("\n"):
        line = _normalize_criterion_line(raw_line)
        lowered = line.lower()
        for criterion in criteria:
            if not lowered.startswith(criterion.lower()):
                continue
            remainder = line[len(criterion):]
            # Require a real delimiter. A bare space would let prose score itself —
            # "Strategic Value in 2 areas was weak" would read as a 2 — which the original
            # colon-based split could never do.
            if remainder[:1] not in (":", "-", "=", ""):
                continue
            numbers = re.findall(r"\d+(?:\.\d+)?", remainder)
            if not numbers:
                no_number.append(criterion)
                continue
            value = float(numbers[0])
            if not JUDGE_SCALE_MIN <= value <= JUDGE_SCALE_MAX:
                out_of_range.append((criterion, value))
                continue
            scores[criterion] = value

    report = {
        "requested": list(criteria),
        "parsed": sorted(scores.keys()),
        "missing": [c for c in criteria if c not in scores],
        "no_number": sorted(set(no_number)),
        "out_of_range": [{"criterion": c, "value": v} for c, v in out_of_range],
    }
    return scores, report


@dataclass
class EvaluationMetrics:
    """Structure for evaluation metrics"""
    comprehensiveness: float
    human_enfranchisement: float
    diversity: float
    faithfulness: float
    overall_score: float
    explanation: str


@dataclass
class ActionSpecificMetrics:
    """Structure for action-specific evaluation metrics with ground truth"""
    bleu_score: float
    meteor_score: float
    rouge_l_f1: float
    rouge_1_f1: float
    rouge_2_f1: float
    semantic_similarity: float
    llm_evaluation_score: float
    exact_match: bool
    explanation: str


@dataclass
class GroundTruthEvaluation:
    """Structure for ground truth evaluation results"""
    question_id: str
    ground_truth: str
    predicted_answer: str
    nlp_metrics: ActionSpecificMetrics
    llm_evaluation: Dict
    overall_score: float


class SensemakingEvaluator:
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_JUDGE_MODEL,
        provider: str = PAPER_PROVIDER,
        API_provider: str = "nebius",
        judge_grade: Optional[str] = None,
    ):
        """
        Initialize the evaluator with GraphRAG-style metrics
        
        Args:
            api_key: API key when using openai/anthropic/huggingface (unused for claude_cli)
            model: Pinned model ID (aliases like haiku expand automatically)
            provider: claude_cli (default paper-parity) | anthropic | openai | huggingface
            API_provider: HF InferenceClient provider (only when provider=huggingface)
            judge_grade: paper | smoke | legacy (auto-inferred from provider if omitted)
        """
        self.provider = provider
        self.model = model
        self.API_provider = API_provider
        self.judge_grade = infer_grade(provider, judge_grade)
        self._observed_model_ids: set = set()
        self._last_call_meta: Dict[str, Any] = {}
        # How much of each rubric had to be imputed; see parse_health_summary()
        self._parse_health: Dict[str, Dict[str, Any]] = {}
        # Guards the counters above, which several judge threads update concurrently when
        # KEO_JUDGE_WORKERS > 1. Without it the imputation totals silently undercount, which is
        # exactly the number used to decide whether a null result is trustworthy.
        self._state_lock = threading.Lock()
        self.judge_workers = judge_workers()
        self.client = None
        
        # Initialize the appropriate client based on provider
        if provider == "claude_cli":
            try:
                ensure_logged_in()
            except (ClaudeAuthError, ClaudeNotFoundError) as e:
                raise ValueError(str(e)) from e
            ver = claude_code_version_text() or "unknown"
            print(
                f"Claude Code judge (pinned snapshot / Pro-Max) version={ver} "
                f"grade={self.judge_grade}"
            )
        elif provider == "openai":
            if not api_key:
                raise ValueError("api_key required for provider=openai")
            self.client = OpenAI(api_key=api_key)
        elif provider == "anthropic":
            if not api_key:
                raise ValueError("api_key required for provider=anthropic")
            self.client = make_anthropic_client(api_key)
        elif provider == "huggingface":
            if not HUGGINGFACE_AVAILABLE:
                raise ImportError("huggingface_hub is not installed. Please install it with: pip install huggingface_hub")
            if not api_key:
                raise ValueError("api_key required for provider=huggingface")
            self.client = InferenceClient(
                provider=API_provider,
                api_key=api_key,
            )
        else:
            raise ValueError(
                f"Unsupported provider: {provider}. "
                "Supported: 'claude_cli' (paper-parity / Pro-Max), 'anthropic' (API), "
                "'openai' (legacy), 'huggingface'"
            )

        self._run_provenance = build_run_provenance(
            role="judge",
            provider=provider,
            model_id=model,
            client=self.client,
            api_provider=API_provider if provider == "huggingface" else None,
            observed_model_ids=self._observed_model_ids,
        )
        self._run_provenance["judge"] = judge_meta(
            provider=provider, model=model, grade=self.judge_grade
        )
        warn = warn_if_non_paper(self.judge_grade)
        if warn:
            print(warn)
        print(
            f"Judge model: requested={model} "
            f"exact={self._run_provenance['llm'].get('exact_id', model)} "
            f"provider={provider} grade={self.judge_grade} "
            f"reportable={self._run_provenance['judge']['reportable_for_paper']}"
        )
        
        # GraphRAG evaluation criteria
        self.evaluation_criteria = {
            'comprehensiveness': {
                'description': 'Completeness within the framing of the implied context of the question',
                'scale': '1-5 where 5 = thoroughly comprehensive, 1 = incomplete or superficial'
            },
            'human_enfranchisement': {
                'description': 'Provision of supporting source material or contextual information',
                'scale': '1-5 where 5 = excellent provenance and citations, 1 = no supporting evidence'
            },
            'diversity': {
                'description': 'Provision of differing viewpoints or angles on the question posed',
                'scale': '1-5 where 5 = multiple perspectives considered, 1 = single narrow view'
            },
            'faithfulness': {
                'description': 'Factual accuracy and grounding in source material',
                'scale': '1-5 where 5 = fully accurate and grounded, 1 = inaccurate or hallucinated'
            }
        }

        if ALLOW_JUDGE_IMPUTATION:
            print(
                "WARNING: KEO_ALLOW_JUDGE_IMPUTATION=1 — unreadable judge criteria will be "
                f"set to the scale midpoint, which compresses differences between arms. "
                "Scores from this run are NOT reportable; see judge_parse in the provenance."
            )
        else:
            print(
                "✓ Judge parse: refusing midpoint imputation on "
                f"{', '.join(sorted(COMPARISON_RUBRICS))} rubrics"
            )

    def get_model_provenance(self) -> Dict:
        prov = build_run_provenance(
            role="judge",
            provider=self.provider,
            model_id=self.model,
            client=self.client,
            api_provider=self.API_provider if self.provider == "huggingface" else None,
            observed_model_ids=self._observed_model_ids,
        )
        prov["judge"] = judge_meta(
            provider=self.provider, model=self.model, grade=self.judge_grade
        )
        # Records whether BLEU/METEOR/ROUGE in this run are real or structural zeros
        prov["nlp_metrics"] = {
            "available": NLP_METRICS_AVAILABLE,
            "missing": NLP_METRICS_MISSING or None,
            "allow_missing_override": ALLOW_MISSING_NLP,
        }
        # Records how many judge criteria were read vs. set to the scale midpoint
        prov["judge_parse"] = self.parse_health_summary()
        if self._last_call_meta:
            prov["last_call"] = self._last_call_meta
        return prov

    def exact_model_id(self) -> str:
        if self._observed_model_ids:
            return sorted(self._observed_model_ids)[-1]
        llm = (self._run_provenance or {}).get("llm") or {}
        return llm.get("exact_id") or llm.get("resolved_id") or self.model

    def _openai_chat_kwargs(self, messages, max_tokens: int, temperature: float) -> Dict[str, Any]:
        """Build chat.completions kwargs; GPT-5.x prefers max_completion_tokens + reasoning_effort."""
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
        }
        mid = (self.model or "").lower()
        if mid.startswith("gpt-5") or mid.startswith("gpt-6") or mid.startswith("o1") or mid.startswith("o3"):
            kwargs["max_completion_tokens"] = max_tokens
            kwargs["reasoning_effort"] = "none"
        else:
            kwargs["max_tokens"] = max_tokens
            kwargs["temperature"] = temperature
        return kwargs

    def _pace_after_judge_call(self) -> None:
        """Optional inter-call delay (Claude Code Pro/Max pacing)."""
        if self.provider != "claude_cli":
            return
        gap = os.getenv("CLAUDE_CLI_CALL_GAP_S", "0.75").strip()
        try:
            sec = float(gap)
        except ValueError:
            sec = 0.75
        if sec > 0:
            time.sleep(sec)

    def _map_judge_calls(
        self,
        items: List[Any],
        fn,
        desc: str,
        on_item_done: Optional[Callable[[int, Any], None]] = None,
    ) -> List[Any]:
        """Apply fn to every item, concurrently when configured, always in input order.

        Order is preserved because scores are later matched to questions positionally in places;
        returning them in completion order would silently attach each score to the wrong question.
        Exceptions are re-raised at the call site so the existing per-item handling and the
        abort-on-JudgeParseError path behave identically either way.

        `on_item_done(idx, value)` runs after each successful item (under the state lock when
        parallel) so callers can checkpoint before a later quota abort drops in-memory scores.
        """
        workers = min(self.judge_workers, len(items)) if items else 1
        if workers <= 1:
            results: List[Any] = []
            for idx, item in enumerate(tqdm(items, desc=desc)):
                value = fn(item)
                results.append(value)
                if on_item_done is not None:
                    on_item_done(idx, value)
            return results

        results = [None] * len(items)

        def run(indexed):
            idx, item = indexed
            return idx, fn(item)

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(run, (i, item)): i for i, item in enumerate(items)}
            with tqdm(total=len(items), desc=f"{desc} ({workers}x)") as bar:
                try:
                    for fut in as_completed(futures):
                        idx, value = fut.result()
                        results[idx] = value
                        bar.update(1)
                        if on_item_done is not None:
                            on_item_done(idx, value)
                except BaseException:
                    for fut in futures:
                        fut.cancel()
                    raise
        return results

    def _reraise_fatal_judge_errors(self, exc: BaseException) -> None:
        """Abort the whole run on auth / missing CLI — do not swallow as per-item errors."""
        if isinstance(exc, (ClaudeAuthError, ClaudeNotFoundError, ClaudeQuotaError, JudgeParseError)):
            raise
        # Also catch ValueError wrapping those from __init__ path if re-raised mid-run
        msg = str(exc).lower()
        if "claude auth login" in msg or "oauth" in msg or "could not find `claude`" in msg:
            raise
    
    def _judge_temperature(self, temperature: float) -> float:
        """Paper Anthropic judges force temp=0; others keep caller value."""
        if self.provider == "anthropic" and self.judge_grade == GRADE_PAPER:
            return DEFAULT_JUDGE_TEMPERATURE
        return temperature

    def _create_chat_completion(self, messages: List[Dict], max_tokens: int = 1000, temperature: float = 0.3) -> str:
        """
        Create a chat completion using the configured provider
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            max_tokens: Maximum number of tokens to generate
            temperature: Temperature for generation
            
        Returns:
            Generated text response
        """
        temperature = self._judge_temperature(temperature)

        if self.provider == "openai":
            response = self.client.chat.completions.create(
                **self._openai_chat_kwargs(messages, max_tokens, temperature)
            )
            observed = observe_completion_model(response)
            if observed:
                self._observed_model_ids.add(observed)
            content = response.choices[0].message.content or ""
            return content.strip()

        elif self.provider == "anthropic":
            text, observed, call_meta = anthropic_chat(
                self.client,
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                disable_thinking=True,
            )
            self._last_call_meta = call_meta
            if observed:
                self._observed_model_ids.add(observed)
            self._pace_after_judge_call()
            return text

        elif self.provider == "claude_cli":
            # Paper-parity via Claude Code: pinned ID, tools off, JSON envelope
            text, observed, call_meta = claude_cli_chat(
                messages=messages,
                model=self.model,
                effort=os.getenv("CLAUDE_CLI_EFFORT", "low"),
                tools="",
            )
            self._last_call_meta = call_meta
            if observed:
                self._observed_model_ids.add(observed)
            self._pace_after_judge_call()
            return text
        
        elif self.provider == "huggingface":
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature
            )
            observed = observe_completion_model(response)
            if observed:
                self._observed_model_ids.add(observed)
            return response.choices[0].message.content.strip()
        
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def evaluate_questions(self, questions: List[Dict]) -> List[Dict]:
        """
        Evaluate the quality of generated sensemaking questions
        
        Args:
            questions: List of question dictionaries
        
        Returns:
            List of question evaluations
        """
        print("Evaluating sensemaking questions...")
        evaluations = []
        
        for question_data in tqdm(questions, desc="Evaluating questions"):
            try:
                evaluation = self._evaluate_single_question(question_data)
                evaluations.append(evaluation)
                
                # Rate limiting
                #time.sleep(0.3)
                
            except Exception as e:
                self._reraise_fatal_judge_errors(e)
                print(f"Error evaluating question {question_data.get('id', '')}: {e}")
                evaluations.append({
                    'question_id': question_data.get('id', ''),
                    'error': True,
                    'error_message': str(e)
                })
        
        return evaluations
    
    def evaluate_answers(self, 
                        answers: List[Dict], 
                        questions: List[Dict],
                        reference_data: Optional[Dict] = None) -> List[Dict]:
        """
        Evaluate answers using GraphRAG-style qualitative metrics
        
        Args:
            answers: List of answer dictionaries
            questions: Original questions for context
            reference_data: Optional reference data for faithfulness checking
        
        Returns:
            List of answer evaluations
        """
        evaluations = []
        
        # Create question lookup
        question_lookup = {q.get('id', ''): q for q in questions}
        
        for answer_data in tqdm(answers, desc="Evaluating answers"):
            try:
                if answer_data.get('error'):
                    continue
                
                question_id = answer_data.get('question_id', '')
                question_data = question_lookup.get(question_id, {})
                
                evaluation = self._evaluate_single_answer(
                    answer_data, question_data, reference_data
                )
                evaluations.append(evaluation)
                
                # Rate limiting
                #time.sleep(0.5)
                
            except Exception as e:
                self._reraise_fatal_judge_errors(e)
                print(f"Error evaluating answer {answer_data.get('question_id', '')}: {e}")
                evaluations.append({
                    'question_id': answer_data.get('question_id', ''),
                    'answer_id': answer_data.get('id', ''),
                    'error': True,
                    'error_message': str(e)
                })
        
        return evaluations
    
    def compare_answer_methods(self, 
                             method1_answers: List[Dict],
                             method2_answers: List[Dict],
                             questions: List[Dict],
                             method1_name: str = "method1",
                             method2_name: str = "method2") -> Dict:
        """
        Compare two answer generation methods
        
        Args:
            method1_answers: Answers from first method
            method2_answers: Answers from second method
            questions: Original questions
            method1_name: Name of first method
            method2_name: Name of second method
        
        Returns:
            Comparative evaluation results
        """
        print(f"Comparing {method1_name} vs {method2_name} answers...")
        
        # Evaluate both sets of answers
        method1_evaluations = self.evaluate_answers(method1_answers, questions)
        method2_evaluations = self.evaluate_answers(method2_answers, questions)
        
        # Perform pairwise comparisons
        pairwise_comparisons = self._perform_pairwise_comparisons(
            method1_answers, method2_answers, questions, method1_name, method2_name
        )
        
        # Calculate aggregate metrics
        comparison_results = self._calculate_comparison_metrics(
            method1_evaluations, method2_evaluations, pairwise_comparisons, method1_name, method2_name
        )
        
        return comparison_results
    
    def evaluate_global_sensemaking_capability(self,
                                             questions: List[Dict],
                                             answers: List[Dict],
                                             prior_evaluations: Optional[List[Dict]] = None,
                                             on_progress: Optional[Callable[[List[Dict]], None]] = None) -> Dict:
        """
        Evaluate capability on global sensemaking questions specifically

        prior_evaluations: already-scored rows (with question_id) to skip and keep.
        on_progress: called with the merged successful evaluations so far (for quota checkpoints).
        """
        print("Evaluating global sensemaking capability...")
        
        # Filter for global questions
        global_questions = [q for q in questions if q.get('type') != 'action_specific']
        global_answers = [a for a in answers if any(
            a.get('question_id') == q.get('id') for q in global_questions
        )]
        
        if not global_questions or not global_answers:
            return {"error": "No global sensemaking questions/answers found"}

        prior = [
            e for e in (prior_evaluations or [])
            if isinstance(e, dict) and e.get("question_id") and not e.get("error")
        ]
        done_ids = {e["question_id"] for e in prior}
        todo_answers = [a for a in global_answers if a.get("question_id") not in done_ids]
        if prior:
            print(f"  resume: keeping {len(prior)} scored, {len(todo_answers)} remaining")

        collected: List[Dict] = list(prior)
        collected_lock = threading.Lock()
        failed: List[str] = []
        
        def judge_one(answer):
            """Score one answer. Returns (evaluation or None, failed question id or None)."""
            question_id = answer.get('question_id', '')
            try:
                question = next((q for q in global_questions if q.get('id') == question_id), {})
                if not question:
                    return None, None
                return self._evaluate_global_sensemaking_answer(answer, question), None
            except Exception as e:
                # Fatal cases (auth, missing CLI, refused imputation) re-raise and abort the run,
                # including out of a worker thread.
                self._reraise_fatal_judge_errors(e)
                print(f"Error evaluating global answer {question_id}: {e}")
                return None, question_id

        def on_item_done(_idx, outcome):
            ev, fail_qid = outcome if isinstance(outcome, tuple) else (outcome, None)
            notify = False
            with collected_lock:
                if ev is not None:
                    collected.append(ev)
                    notify = on_progress is not None
                snapshot = list(collected) if notify else None
            if notify and snapshot is not None:
                on_progress(snapshot)

        try:
            outcomes = self._map_judge_calls(
                todo_answers, judge_one, "Evaluating global answers", on_item_done=on_item_done
            )
        except ClaudeQuotaError:
            if on_progress is not None:
                with collected_lock:
                    snapshot = list(collected)
                on_progress(snapshot)
                print(f"  quota stop — checkpointed {len(snapshot)} successful judgments")
            raise

        # Prefer collected (includes prior + progress) over re-deriving from outcomes alone
        new_evals = [ev for ev, _ in outcomes if ev is not None]
        new_failed = [qid for _, qid in outcomes if qid]
        # collected may already contain new_evals via on_item_done; rebuild uniquely by qid
        by_qid = {e["question_id"]: e for e in prior}
        orphans = []
        for e in new_evals:
            qid = e.get("question_id")
            if qid:
                by_qid[qid] = e
            else:
                orphans.append(e)
        global_evaluations = list(by_qid.values()) + orphans
        failed = new_failed
        
        # Calculate global sensemaking metrics
        global_metrics = self._calculate_global_metrics(global_evaluations)
        usable = [e for e in global_evaluations if not e.get('error')]
        
        if not usable:
            # An arm where every judgement failed must not come back looking scored; the
            # caller treats a result without 'error' as valid and would report it as 0.0.
            return {
                'error': (
                    f"no answers could be scored ({len(failed)} raised, "
                    f"{len(global_evaluations) - len(usable)} returned errors of "
                    f"{len(global_answers)} answers)"
                ),
                'global_questions_evaluated': 0,
                'global_metrics': global_metrics,
                'individual_evaluations': global_evaluations,
                'failed_question_ids': failed,
                'incomplete': True,
                'expected_n': len(global_answers),
            }
        
        if failed:
            print(f"  ⚠️  {len(failed)} of {len(global_answers)} answers could not be scored; "
                  f"metrics are over the remaining {len(usable)}")

        incomplete = len(usable) < len(global_answers)
        return {
            'judge_style': HOLISTIC_ID,
            'judge_style_names': list(HOLISTIC_NAMES),
            'global_questions_evaluated': len(usable),
            'global_metrics': global_metrics,
            'individual_evaluations': global_evaluations,
            'failed_question_ids': failed,
            'incomplete': incomplete,
            'expected_n': len(global_answers),
        }
    
    def compare_action_specific_methods(self,
                                      method1_answers: List[Dict],
                                      method2_answers: List[Dict],
                                      questions: List[Dict],
                                      method1_name: str = "method1",
                                      method2_name: str = "method2") -> Dict:
        """
        Compare two methods for action-specific questions with ground truth
        
        Args:
            method1_answers: First method answers
            method2_answers: Second method answers
            questions: Questions with ground truth
            method1_name: Name of first method
            method2_name: Name of second method
        
        Returns:
            Comparison results for action-specific questions
        """
        print(f"Comparing {method1_name} vs {method2_name} on action-specific questions with ground truth...")
        
        # Evaluate both methods
        method1_results = self.evaluate_action_specific_answers(method1_answers, questions)
        method2_results = self.evaluate_action_specific_answers(method2_answers, questions)
        
        if method1_results.get('error') or method2_results.get('error'):
            return {"error": "Failed to evaluate one or both methods"}
        
        # Compare aggregate metrics
        comparison = self._compare_action_metrics(
            method1_results['aggregate_metrics'],
            method2_results['aggregate_metrics'],
            method1_name,
            method2_name
        )
        
        return {
            f'{method1_name}_results': method1_results,
            f'{method2_name}_results': method2_results,
            'comparison': comparison,
            'winner': self._determine_action_winner(comparison, method1_name, method2_name)
        }
    
    def _evaluate_single_question(self, question_data: Dict) -> Dict:
        """Evaluate a single question for quality"""
        
        prompt = f"""
Evaluate this aviation maintenance sensemaking question for quality:

Question: {question_data['question']}
Category: {question_data.get('category', 'unknown')}
Type: {question_data.get('type', 'unknown')}

Rate the question on these criteria (1-5 scale):

1. Clarity: Is the question clear and well-formulated?
2. Relevance: How relevant is it to aviation maintenance safety?
3. Complexity: Does it require synthesis across multiple data points?
4. Actionability: Could answers lead to actionable insights?
5. Sensemaking Value: Does it support understanding of broader patterns?

For each criterion, provide:
- Score (1-5)
- Brief explanation

Then provide an overall assessment and suggestions for improvement if any.

Format your response as:
Clarity: [score] - [explanation]
Relevance: [score] - [explanation]
Complexity: [score] - [explanation]
Actionability: [score] - [explanation]
Sensemaking Value: [score] - [explanation]

Overall Assessment: [overall score 1-5] - [explanation]
Suggestions: [any improvements needed]
"""
        
        evaluation_text = self._create_chat_completion(
            messages=[
                {"role": "system", "content": "You are an expert evaluator of research questions in aviation safety and maintenance, with deep knowledge of what makes effective sensemaking questions."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=600,
            temperature=0.2
        )
        
        # Parse scores (simplified parsing)
        scores = self._parse_question_scores(evaluation_text)
        
        return {
            'question_id': question_data.get('id', ''),
            'question': question_data['question'],
            'category': question_data.get('category', ''),
            'evaluation_scores': scores,
            'evaluation_text': evaluation_text,
            'evaluator_model': self.model,
            'evaluator_model_exact': self.exact_model_id(),
        }
    
    def _evaluate_single_answer(self, 
                              answer_data: Dict, 
                              question_data: Dict,
                              reference_data: Optional[Dict] = None) -> Dict:
        """Evaluate a single answer using GraphRAG criteria"""
        
        prompt = f"""
Evaluate this answer to an aviation maintenance sensemaking question using the following criteria:

Question: {question_data.get('question', '')}
Answer: {answer_data.get('answer', '')}
Method: {answer_data.get('method', 'unknown')}

Rate the answer on these GraphRAG-style criteria (1-5 scale):

1. COMPREHENSIVENESS: Completeness within the framing of the implied context of the question
   - Does the answer thoroughly address all aspects of the question?
   - Is it complete within the expected scope?

2. HUMAN ENFRANCHISEMENT: Provision of supporting source material or contextual information
   - Does it provide evidence, examples, or source references?
   - Can a human verify and build upon the information?

3. DIVERSITY: Provision of differing viewpoints or angles on the question posed
   - Does it consider multiple perspectives or approaches?
   - Are different factors or scenarios addressed?

4. FAITHFULNESS: Factual accuracy and grounding in source material
   - Is the information accurate and well-grounded?
   - Are there any apparent hallucinations or inaccuracies?

For each criterion, provide:
- Score (1-5)
- Detailed explanation

Then provide an overall assessment.

Format your response as:
Comprehensiveness: [score] - [explanation]
Human Enfranchisement: [score] - [explanation]
Diversity: [score] - [explanation]
Faithfulness: [score] - [explanation]

Overall Score: [average score] - [overall assessment]
Strengths: [key strengths]
Weaknesses: [areas for improvement]
"""
        
        evaluation_text = self._create_chat_completion(
            messages=[
                {"role": "system", "content": "You are an expert evaluator of analytical responses in aviation safety, using GraphRAG evaluation methodology to assess answer quality."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=800,
            temperature=0.2
        )
        
        # Parse scores
        metrics = self._parse_answer_scores(evaluation_text)
        
        return {
            'question_id': answer_data.get('question_id', ''),
            'answer_method': answer_data.get('method', ''),
            'evaluation_metrics': metrics,
            'evaluation_text': evaluation_text,
            'evaluator_model': self.model,
            'evaluator_model_exact': self.exact_model_id()
        }
    
    def _evaluate_global_sensemaking_answer(self, answer_data: Dict, question_data: Dict) -> Dict:
        """Holistic / Sensemaking rubric. GPT runs only this style.

        The Grounded / Source-card rubric lives in grounded_judge.py and is not
        called from here. Scores from the two styles are not pooled.
        """
        
        prompt = f"""
Evaluate this answer to a GLOBAL SENSEMAKING question about aviation maintenance data:

Question: {question_data.get('question', '')}
Answer: {answer_data.get('answer', '')}

Global sensemaking questions require synthesis across entire datasets to identify overarching themes and patterns.

Rate this answer on these specialized criteria (1-5 scale):

1. GLOBAL PERSPECTIVE: Does the answer demonstrate understanding of dataset-wide patterns?
2. THEME IDENTIFICATION: Are major themes and patterns clearly identified?
3. SYNTHESIS QUALITY: How well does it synthesize information across multiple sources?
4. STRATEGIC VALUE: Does it provide insights useful for high-level decision making?
5. PATTERN RECOGNITION: Are complex relationships and dependencies identified?

For each criterion, provide score and explanation.

Then assess whether this answer successfully demonstrates global sensemaking capability.

Format as:
Global Perspective: [score] - [explanation]
Theme Identification: [score] - [explanation]
Synthesis Quality: [score] - [explanation]
Strategic Value: [score] - [explanation]
Pattern Recognition: [score] - [explanation]

Global Sensemaking Assessment: [overall evaluation of global capability]
"""
        
        evaluation_text = self._create_chat_completion(
            messages=[
                {"role": "system", "content": "You are an expert in evaluating global sensemaking capabilities in data analysis systems, particularly for aviation safety applications."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=700,
            temperature=0.2
        )
        
        # Parse global sensemaking scores
        global_scores = self._parse_global_scores(evaluation_text)
        
        return {
            'question_id': question_data.get('id', ''),
            'global_metrics': global_scores,
            'evaluation_text': evaluation_text,
            'evaluator_model': self.model,
            'evaluator_model_exact': self.exact_model_id()
        }
    
    def evaluate_action_specific_answers(self,
                                       answers: List[Dict],
                                       questions: List[Dict]) -> Dict:
        """
        Evaluate action-specific answers using ground truth and NLP metrics
        
        Args:
            answers: List of answer dictionaries
            questions: List of questions with ground truth for action-specific questions
        
        Returns:
            Comprehensive evaluation results for action-specific questions
        """
        print("Evaluating action-specific answers with ground truth metrics...")
        require_nlp_metrics()
        
        # Filter for action-specific questions with ground truth
        action_questions = {q.get('id', ''): q for q in questions 
                          if q.get('category') == 'action_specific' and q.get('ground_truth_answer')}
        
        action_answers = [a for a in answers 
                         if a.get('question_id') in action_questions and not a.get('error')]
        
        if not action_questions or not action_answers:
            return {"error": "No action-specific answers to be evaluated"}
        
        print(f"Found {len(action_questions)} action-specific questions with ground truth")
        print(f"Found {len(action_answers)} corresponding answers to evaluate")
        
        # Evaluate each answer
        evaluations = []
        def judge_one_action(answer):
            question_id = answer.get('question_id', '')
            try:
                question = action_questions.get(question_id)
                if not question:
                    return None
                return self._evaluate_single_action_answer(answer, question)
            except Exception as e:
                self._reraise_fatal_judge_errors(e)
                print(f"Error evaluating action answer {question_id}: {e}")
                return None

        evaluations = [
            ev for ev in self._map_judge_calls(
                action_answers, judge_one_action, "Evaluating action-specific answers"
            )
            if ev is not None
        ]
        
        # Calculate aggregate metrics
        aggregate_metrics = self._calculate_action_aggregate_metrics(evaluations)
        
        return {
            'total_action_questions': len(action_questions),
            'evaluated_answers': len(evaluations),
            'individual_evaluations': evaluations,
            'aggregate_metrics': aggregate_metrics,
            'evaluation_summary': self._generate_action_evaluation_summary(aggregate_metrics)
        }
    
    def _evaluate_single_action_answer(self, answer: Dict, question: Dict) -> GroundTruthEvaluation:
        """Evaluate a single action-specific answer against ground truth"""
        
        ground_truth = question.get('ground_truth_answer', '').strip()
        predicted = answer.get('answer', '').strip()
        question_id = question.get('id', '')
        
        # Calculate NLP metrics
        nlp_metrics = self._calculate_nlp_metrics(predicted, ground_truth)

        if ACTION_NLP_ONLY:
            llm_evaluation = {
                "score": None,
                "skipped": True,
                "reason": "KEO_ACTION_NLP_ONLY=1",
                "explanation": "LLM action judge skipped; NLP metrics only",
            }
            overall_score = float(nlp_metrics.rouge_l_f1)
        else:
            llm_evaluation = self._llm_evaluate_action_answer(
                question.get('question', ''), predicted, ground_truth
            )
            overall_score = self._calculate_overall_action_score(nlp_metrics, llm_evaluation)

        return GroundTruthEvaluation(
            question_id=question_id,
            ground_truth=ground_truth,
            predicted_answer=predicted,
            nlp_metrics=nlp_metrics,
            llm_evaluation=llm_evaluation,
            overall_score=overall_score
        )
    
    def _calculate_nlp_metrics(self, predicted: str, ground_truth: str) -> ActionSpecificMetrics:
        """Calculate NLP metrics for action-specific answers"""
        
        if not NLP_METRICS_AVAILABLE:
            return ActionSpecificMetrics(
                bleu_score=0.0,
                meteor_score=0.0,
                rouge_l_f1=0.0,
                rouge_1_f1=0.0,
                rouge_2_f1=0.0,
                semantic_similarity=0.0,
                llm_evaluation_score=0.0,
                exact_match=False,
                explanation="NLP metrics libraries not available"
            )
        
        # Preprocess text
        predicted_clean = self._preprocess_text(predicted)
        ground_truth_clean = self._preprocess_text(ground_truth)
        
        # Exact match
        exact_match = predicted_clean.lower() == ground_truth_clean.lower()
        
        # Tokenize
        predicted_tokens = predicted_clean.split()
        ground_truth_tokens = ground_truth_clean.split()
        
        # BLEU score
        smoothing = SmoothingFunction().method1
        bleu = sentence_bleu([ground_truth_tokens], predicted_tokens, smoothing_function=smoothing)
        
        # METEOR score
        try:
            meteor = meteor_score([ground_truth_clean], predicted_clean)
        except:
            meteor = 0.0
        
        # ROUGE scores
        try:
            scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
            rouge_scores = scorer.score(ground_truth_clean, predicted_clean)
            rouge_1_f1 = rouge_scores['rouge1'].fmeasure
            rouge_2_f1 = rouge_scores['rouge2'].fmeasure
            rouge_l_f1 = rouge_scores['rougeL'].fmeasure
        except:
            rouge_1_f1 = rouge_2_f1 = rouge_l_f1 = 0.0
        
        # Semantic similarity (simple word overlap F1)
        semantic_sim = self._calculate_word_overlap_f1(predicted_tokens, ground_truth_tokens)
        
        return ActionSpecificMetrics(
            bleu_score=bleu,
            meteor_score=meteor,
            rouge_l_f1=rouge_l_f1,
            rouge_1_f1=rouge_1_f1,
            rouge_2_f1=rouge_2_f1,
            semantic_similarity=semantic_sim,
            llm_evaluation_score=0.0,  # Will be filled by LLM evaluation
            exact_match=exact_match,
            explanation=f"BLEU: {bleu:.3f}, METEOR: {meteor:.3f}, ROUGE-L: {rouge_l_f1:.3f}"
        )
    
    def _llm_evaluate_action_answer(self, question: str, predicted: str, ground_truth: str) -> Dict:
        """Use LLM to evaluate action-specific answer quality"""
        
        prompt = f"""
Evaluate this predicted answer for an action-specific aviation maintenance question against the ground truth.

Question: {question}
Ground Truth Answer: {ground_truth}
Predicted Answer: {predicted}

Rate the predicted answer on these criteria (1-5 scale):

1. CORRECTNESS: How factually correct is the predicted answer?
2. COMPLETENESS: Does it include all necessary action steps?
3. PRACTICALITY: How practical and actionable is the suggested action?
4. SAFETY: Does it maintain or improve safety standards?
5. CLARITY: How clear and understandable is the instruction?

For each criterion, provide:
- Score (1-5)
- Brief explanation

Then provide:
- Overall assessment of answer quality
- Whether the predicted answer would be safe to follow
- Key differences from ground truth
- Suggestions for improvement

Format as:
Correctness: [score] - [explanation]
Completeness: [score] - [explanation]
Practicality: [score] - [explanation]
Safety: [score] - [explanation]
Clarity: [score] - [explanation]

Overall Score: [average score] - [overall assessment]
Safety Assessment: [Safe/Unsafe/Partially Safe] - [explanation]
Key Differences: [main differences from ground truth]
Improvement Suggestions: [suggestions if any]
"""
        
        try:
            evaluation_text = self._create_chat_completion(
                messages=[
                    {"role": "system", "content": "You are an expert aviation maintenance evaluator with deep knowledge of proper maintenance procedures and safety standards."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=700,
                temperature=0.2
            )
            
            # Parse LLM evaluation scores
            llm_scores = self._parse_action_llm_scores(evaluation_text)
            
            return {
                'llm_scores': llm_scores,
                'evaluation_text': evaluation_text,
                'overall_llm_score': np.mean(list(llm_scores.values())) if llm_scores else 3.0
            }
            
        except Exception as e:
            self._reraise_fatal_judge_errors(e)
            print(f"Error in LLM evaluation: {e}")
            return {
                'llm_scores': {},
                'evaluation_text': f"Error: {str(e)}",
                'overall_llm_score': 3.0
            }
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for NLP metrics"""
        # Remove extra whitespace and normalize
        text = ' '.join(text.split())
        # Remove common maintenance text artifacts
        text = re.sub(r'[^\w\s&]', ' ', text)  # Keep & for maintenance actions
        text = ' '.join(text.split())
        return text
    
    def _calculate_word_overlap_f1(self, predicted_tokens: List[str], ground_truth_tokens: List[str]) -> float:
        """Calculate F1 score based on word overlap"""
        if not predicted_tokens or not ground_truth_tokens:
            return 0.0
        
        predicted_set = set(token.lower() for token in predicted_tokens)
        ground_truth_set = set(token.lower() for token in ground_truth_tokens)
        
        intersection = predicted_set & ground_truth_set
        
        if not intersection:
            return 0.0
        
        precision = len(intersection) / len(predicted_set)
        recall = len(intersection) / len(ground_truth_set)
        
        f1 = 2 * precision * recall / (precision + recall)
        return f1
    
    def _calculate_action_aggregate_metrics(self, evaluations: List[GroundTruthEvaluation]) -> Dict:
        """Calculate aggregate metrics from action-specific evaluations"""
        
        if not evaluations:
            return {
                'total_evaluated': 0,
                'bleu_scores_mean': 0.0,
                'meteor_scores_mean': 0.0,
                'rouge_l_f1_scores_mean': 0.0,
                'rouge_1_f1_scores_mean': 0.0,
                'rouge_2_f1_scores_mean': 0.0,
                'semantic_similarity_scores_mean': 0.0,
                'exact_matches_rate': 0.0,
                'llm_evaluation_scores_mean': 0.0
            }
        
        # Collect all scores
        bleu_scores = []
        meteor_scores = []
        rouge_l_f1_scores = []
        rouge_1_f1_scores = []
        rouge_2_f1_scores = []
        semantic_similarity_scores = []
        exact_matches = []
        llm_evaluation_scores = []
        
        for evaluation in evaluations:
            try:
                # Extract NLP metrics
                nlp_metrics = evaluation.nlp_metrics
                bleu_scores.append(nlp_metrics.bleu_score)
                meteor_scores.append(nlp_metrics.meteor_score)
                rouge_l_f1_scores.append(nlp_metrics.rouge_l_f1)
                rouge_1_f1_scores.append(nlp_metrics.rouge_1_f1)
                rouge_2_f1_scores.append(nlp_metrics.rouge_2_f1)
                semantic_similarity_scores.append(nlp_metrics.semantic_similarity)
                exact_matches.append(nlp_metrics.exact_match)
                
                # Extract LLM evaluation score (omit when NLP-only skip)
                llm = evaluation.llm_evaluation or {}
                if not llm.get("skipped"):
                    score = llm.get("overall_llm_score", llm.get("score"))
                    if score is not None:
                        llm_evaluation_scores.append(float(score))
                
            except Exception as e:
                continue
        
        # Calculate means
        aggregate_metrics = {
            'total_evaluated': len(evaluations),
            'bleu_scores_mean': np.mean(bleu_scores) if bleu_scores else 0.0,
            'meteor_scores_mean': np.mean(meteor_scores) if meteor_scores else 0.0,
            'rouge_l_f1_scores_mean': np.mean(rouge_l_f1_scores) if rouge_l_f1_scores else 0.0,
            'rouge_1_f1_scores_mean': np.mean(rouge_1_f1_scores) if rouge_1_f1_scores else 0.0,
            'rouge_2_f1_scores_mean': np.mean(rouge_2_f1_scores) if rouge_2_f1_scores else 0.0,
            'semantic_similarity_scores_mean': np.mean(semantic_similarity_scores) if semantic_similarity_scores else 0.0,
            'exact_matches_rate': np.mean(exact_matches) if exact_matches else 0.0,
            'llm_evaluation_scores_mean': (
                float(np.mean(llm_evaluation_scores)) if llm_evaluation_scores else None
            ),
            'llm_scored_n': len(llm_evaluation_scores),
            'nlp_only': ACTION_NLP_ONLY or len(llm_evaluation_scores) == 0,
        }
        
        return aggregate_metrics
    
    def _generate_action_evaluation_summary(self, aggregate_metrics: Dict) -> str:
        """Generate a summary of action-specific evaluation results"""
        if not aggregate_metrics:
            return "No evaluation metrics available"
        
        summary_parts = []
        summary_parts.append("ACTION-SPECIFIC EVALUATION SUMMARY")
        summary_parts.append("=" * 40)
        summary_parts.append(f"Total Questions Evaluated: {aggregate_metrics.get('total_evaluated', 0)}")
        summary_parts.append(f"Average BLEU Score: {aggregate_metrics.get('bleu_scores_mean', 0):.3f}")
        summary_parts.append(f"Average METEOR Score: {aggregate_metrics.get('meteor_scores_mean', 0):.3f}")
        summary_parts.append(f"Average ROUGE-L F1: {aggregate_metrics.get('rouge_l_f1_scores_mean', 0):.3f}")
        summary_parts.append(f"Exact Match Rate: {aggregate_metrics.get('exact_matches_rate', 0):.3f}")
        llm_mean = aggregate_metrics.get('llm_evaluation_scores_mean')
        if llm_mean is None or aggregate_metrics.get('nlp_only'):
            summary_parts.append("LLM Evaluation Score: skipped (NLP-only)")
        else:
            summary_parts.append(f"LLM Evaluation Score: {llm_mean:.3f}")
        
        return "\n".join(summary_parts)
    
    def _calculate_overall_action_score(self, nlp_metrics: ActionSpecificMetrics, llm_evaluation: Dict) -> float:
        """Calculate overall score for action-specific answer"""
        # Weighted combination of NLP and LLM scores
        nlp_score = (nlp_metrics.bleu_score + nlp_metrics.meteor_score + nlp_metrics.rouge_l_f1) / 3.0
        llm_score = llm_evaluation.get('overall_llm_score', 0.0) / 5.0  # Normalize from 1-5 to 0-1
        
        # 60% NLP metrics, 40% LLM evaluation
        overall_score = (0.6 * nlp_score) + (0.4 * llm_score)
        return overall_score
    
    def _record_parse_health(self, kind: str, report: Dict, imputed: List[str]) -> None:
        """
        Accumulate how much of the judge's output had to be guessed at.

        Surfaced through get_model_provenance() as `judge_parse` so a reader can tell a
        run where every score was read from one where the format broke and midpoints were
        substituted. Those two look identical in the score tables.
        """
        with self._state_lock:
            health = self._parse_health.setdefault(
                kind,
                {"responses": 0, "criteria_expected": 0, "criteria_parsed": 0,
                 "imputed": Counter(), "no_number": Counter(), "out_of_range": []},
            )
            health["responses"] += 1
            health["criteria_expected"] += len(report["requested"])
            health["criteria_parsed"] += len(report["parsed"])
            for criterion in imputed:
                health["imputed"][criterion] += 1
            for criterion in report["no_number"]:
                health["no_number"][criterion] += 1
            health["out_of_range"].extend(report["out_of_range"])

        if imputed and kind in COMPARISON_RUBRICS and not ALLOW_JUDGE_IMPUTATION:
            midpoint = JUDGE_SCALE_MIN + (JUDGE_SCALE_MAX - JUDGE_SCALE_MIN) / 2
            raise JudgeParseError(
                f"Judge output for the {kind} rubric was unreadable for {imputed}. "
                f"Substituting the scale midpoint ({midpoint:.1f}) would compress the "
                "difference between arms, which is the comparison this run exists to "
                "measure. Check the judge's output format against the rubric; the first "
                f"unreadable response is reproduced below.\n"
                f"Report: {report}\n"
                "To accept midpoints anyway (NOT reportable), set "
                "KEO_ALLOW_JUDGE_IMPUTATION=1."
            )

    def parse_health_summary(self) -> Dict:
        """Per-rubric imputation rates, plus an overall flag for the run."""
        summary: Dict[str, Any] = {
            "refuses_imputation": not ALLOW_JUDGE_IMPUTATION,
            "allow_imputation_override": ALLOW_JUDGE_IMPUTATION,
            "comparison_rubrics": sorted(COMPARISON_RUBRICS),
            "alarm_threshold": JUDGE_IMPUTATION_ALARM,
            "scale": [JUDGE_SCALE_MIN, JUDGE_SCALE_MAX],
            "judge_workers": self.judge_workers,
            "by_rubric": {},
        }
        total_expected = total_parsed = 0
        for kind, health in self._parse_health.items():
            expected = health["criteria_expected"]
            parsed = health["criteria_parsed"]
            total_expected += expected
            total_parsed += parsed
            summary["by_rubric"][kind] = {
                "responses": health["responses"],
                "criteria_expected": expected,
                "criteria_parsed": parsed,
                "imputation_rate": (expected - parsed) / expected if expected else 0.0,
                "imputed_by_criterion": dict(health["imputed"]),
                "no_number_by_criterion": dict(health["no_number"]),
                "out_of_range": health["out_of_range"][:20],
            }
        rate = (total_expected - total_parsed) / total_expected if total_expected else 0.0
        summary["imputation_rate"] = rate
        summary["exceeds_alarm"] = rate > JUDGE_IMPUTATION_ALARM
        return summary

    def warn_if_scores_were_imputed(self) -> Optional[str]:
        """Print-ready warning when enough of the rubric was guessed to distort arms."""
        summary = self.parse_health_summary()
        if not summary["exceeds_alarm"]:
            return None
        return (
            f"WARNING: {summary['imputation_rate']:.1%} of judge criteria could not be "
            f"parsed and were set to the scale midpoint ({JUDGE_SCALE_MIN + (JUDGE_SCALE_MAX - JUDGE_SCALE_MIN) / 2:.1f}). "
            "Midpoint imputation compresses differences between arms, so null results in "
            "this run may be artefacts of the judge's output format rather than findings. "
            "Inspect judge_parse in the run provenance; unset KEO_ALLOW_JUDGE_IMPUTATION "
            "to fail on the first unreadable response instead of imputing."
        )

    def _parse_action_llm_scores(self, evaluation_text: str) -> Dict:
        """Parse LLM evaluation scores for action-specific answers"""
        criteria = ['Correctness', 'Completeness', 'Practicality', 'Safety', 'Clarity']
        parsed, report = parse_criterion_scores(evaluation_text, criteria)

        scores = {c.lower(): v for c, v in parsed.items()}
        # Only criteria the judge actually mentioned were imputed before; a criterion it
        # never mentioned stayed absent. Preserved so archived numbers stay comparable.
        mentioned_but_unreadable = [
            c for c in report["no_number"] if c not in parsed
        ] + [item["criterion"] for item in report["out_of_range"] if item["criterion"] not in parsed]
        for criterion in mentioned_but_unreadable:
            scores[criterion.lower()] = 3.0
        self._record_parse_health("action", report, mentioned_but_unreadable)
        return scores
    
    def _parse_question_scores(self, evaluation_text: str) -> Dict:
        """Parse LLM evaluation scores for question quality"""
        criteria = ['Clarity', 'Relevance', 'Complexity', 'Actionability',
                    'Sensemaking Value', 'Overall Assessment']
        parsed, report = parse_criterion_scores(evaluation_text, criteria)

        def key(criterion: str) -> str:
            return criterion.lower().replace(' ', '_')

        scores = {key(c): v for c, v in parsed.items()}
        mentioned_but_unreadable = [
            c for c in report["no_number"] if c not in parsed
        ] + [item["criterion"] for item in report["out_of_range"] if item["criterion"] not in parsed]
        for criterion in mentioned_but_unreadable:
            scores[key(criterion)] = 3.0
        self._record_parse_health("question", report, mentioned_but_unreadable)
        return scores

    def _parse_answer_scores(self, evaluation_text: str) -> EvaluationMetrics:
        """Parse GraphRAG-style evaluation scores from LLM response"""
        default_scores = {
            'comprehensiveness': 3.0,
            'human_enfranchisement': 3.0,
            'diversity': 3.0,
            'faithfulness': 3.0
        }
        criteria = list(default_scores.keys())
        scores, report = parse_criterion_scores(evaluation_text, criteria)

        # This rubric fills every unread criterion, not just the mentioned ones, so its
        # imputation set is the full `missing` list.
        for criterion in report["missing"]:
            scores[criterion] = default_scores[criterion]
        self._record_parse_health("answer", report, report["missing"])

        overall_score = sum(scores.values()) / len(scores)
        
        return EvaluationMetrics(
            comprehensiveness=scores['comprehensiveness'],
            human_enfranchisement=scores['human_enfranchisement'],
            diversity=scores['diversity'],
            faithfulness=scores['faithfulness'],
            overall_score=overall_score,
            explanation=evaluation_text
        )

    def _parse_global_scores(self, evaluation_text: str) -> Dict:
        """Parse global sensemaking evaluation scores"""
        criteria = ['Global Perspective', 'Theme Identification', 'Synthesis Quality',
                    'Strategic Value', 'Pattern Recognition']
        parsed, report = parse_criterion_scores(evaluation_text, criteria)

        def key(criterion: str) -> str:
            return criterion.lower().replace(' ', '_')

        scores = {key(c): v for c, v in parsed.items()}
        mentioned_but_unreadable = [
            c for c in report["no_number"] if c not in parsed
        ] + [item["criterion"] for item in report["out_of_range"] if item["criterion"] not in parsed]
        for criterion in mentioned_but_unreadable:
            scores[key(criterion)] = 3.0
        self._record_parse_health("global", report, mentioned_but_unreadable)
        return scores

    def _perform_pairwise_comparisons(self, method1_answers: List[Dict], method2_answers: List[Dict], 
                                     questions: List[Dict], method1_name: str, method2_name: str) -> List[Dict]:
        """Perform pairwise comparisons between two answer generation methods"""
        comparisons = []
        
        # Create lookups
        method1_lookup = {a.get('question_id', ''): a for a in method1_answers}
        method2_lookup = {a.get('question_id', ''): a for a in method2_answers}
        question_lookup = {q.get('id', ''): q for q in questions}
        
        # Find matching questions
        common_question_ids = set(method1_lookup.keys()) & set(method2_lookup.keys())
        
        for question_id in list(common_question_ids)[:5]:  # Limit for demo
            try:
                method1_answer = method1_lookup[question_id]
                method2_answer = method2_lookup[question_id]
                question = question_lookup.get(question_id, {})
                
                comparison = self._compare_answer_pair(method1_answer, method2_answer, question, method1_name, method2_name)
                comparisons.append(comparison)
                
                #time.sleep(0.5)
                
            except Exception as e:
                self._reraise_fatal_judge_errors(e)
                print(f"Error in pairwise comparison for {question_id}: {e}")
        
        return comparisons

    def _compare_answer_pair(self, method1_answer: Dict, method2_answer: Dict, question: Dict, 
                           method1_name: str, method2_name: str) -> Dict:
        """Compare a pair of answers from two different methods"""
        
        prompt = f"""
Compare these two answers to the same aviation maintenance question:

Question: {question.get('question', '')}

Answer A ({method1_name}): {method1_answer.get('answer', '')}
Answer B ({method2_name}): {method2_answer.get('answer', '')}

Compare the answers on these criteria:
1. Which is more comprehensive?
2. Which provides better supporting evidence?
3. Which offers more diverse perspectives?
4. Which is more factually accurate?

For each criterion, indicate: A, B, or Tie

Then provide an overall preference: A, B, or Tie

Format as:
Comprehensiveness: [A/B/Tie] - [explanation]
Supporting Evidence: [A/B/Tie] - [explanation]
Diverse Perspectives: [A/B/Tie] - [explanation]
Factual Accuracy: [A/B/Tie] - [explanation]

Overall Preference: [A/B/Tie] - [explanation]
"""
        
        try:
            comparison_text = self._create_chat_completion(
                messages=[
                    {"role": "system", "content": "You are an expert evaluator comparing analytical responses for aviation safety questions."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.2
            )
            
            # Parse comparison results
            comparison_scores = self._parse_comparison_scores(comparison_text, method1_name, method2_name)
            
            return {
                'question_id': question.get('id', ''),
                f'{method1_name}_answer_id': method1_answer.get('id', ''),
                f'{method2_name}_answer_id': method2_answer.get('id', ''),
                'comparison_scores': comparison_scores,
                'comparison_text': comparison_text,
                'method1_name': method1_name,
                'method2_name': method2_name
            }
            
        except Exception as e:
            self._reraise_fatal_judge_errors(e)
            return {
                'question_id': question.get('id', ''),
                'error': str(e)
            }

    def _parse_comparison_scores(self, comparison_text: str, method1_name: str, method2_name: str) -> Dict:
        """Parse pairwise comparison scores"""
        scores = {}
        lines = comparison_text.split('\n')
        
        criteria = ['Comprehensiveness', 'Supporting Evidence', 'Diverse Perspectives', 'Factual Accuracy', 'Overall Preference']
        
        for line in lines:
            for criterion in criteria:
                if line.startswith(criterion):
                    try:
                        result_part = line.split(':')[1].strip()
                        if result_part.startswith('A'):
                            scores[criterion.lower().replace(' ', '_')] = method1_name
                        elif result_part.startswith('B'):
                            scores[criterion.lower().replace(' ', '_')] = method2_name
                        else:
                            scores[criterion.lower().replace(' ', '_')] = 'tie'
                    except:
                        scores[criterion.lower().replace(' ', '_')] = 'tie'
        
        return scores

    def _calculate_comparison_metrics(self, method1_evaluations: List[Dict], method2_evaluations: List[Dict], 
                                    pairwise_comparisons: List[Dict], method1_name: str, method2_name: str) -> Dict:
        """Calculate comparison metrics between methods"""
        
        # Calculate average scores for each method
        method1_avg = self._calculate_average_scores(method1_evaluations)
        method2_avg = self._calculate_average_scores(method2_evaluations)
        
        # Analyze pairwise comparisons
        pairwise_results = self._analyze_pairwise_results(pairwise_comparisons, method1_name, method2_name)
        
        return {
            f'{method1_name}_average_scores': method1_avg,
            f'{method2_name}_average_scores': method2_avg,
            'pairwise_comparison_results': pairwise_results,
            'evaluation_summary': self._generate_comparison_summary(method1_avg, method2_avg, pairwise_results, method1_name, method2_name)
        }

    def _calculate_average_scores(self, evaluations: List[Dict]) -> Dict:
        """Calculate average scores from evaluations"""
        if not evaluations:
            return {'total_evaluated': 0, 'error': 'no answers could be scored'}
        
        total_scores = {
            'comprehensiveness': 0,
            'human_enfranchisement': 0,
            'diversity': 0,
            'faithfulness': 0,
            'overall_score': 0
        }
        count = 0
        
        for evaluation in evaluations:
            if evaluation.get('error'):
                continue
            
            metrics = evaluation.get('evaluation_metrics')
            if metrics:
                for key in total_scores.keys():
                    if hasattr(metrics, key):
                        total_scores[key] += getattr(metrics, key)
                count += 1
        
        if count == 0:
            # Returning total_scores here would report a clean set of 0.0 ratings for an arm
            # where nothing was actually scored, which reads as a real (terrible) result.
            return {'total_evaluated': 0, 'error': 'no answers could be scored'}
        
        # Calculate averages
        average_scores = {key: score / count for key, score in total_scores.items()}
        average_scores['total_evaluated'] = count
        return average_scores

    def _analyze_pairwise_results(self, pairwise_comparisons: List[Dict], method1_name: str, method2_name: str) -> Dict:
        """Analyze pairwise comparison results"""
        if not pairwise_comparisons:
            return {}
        
        criteria = ['comprehensiveness', 'supporting_evidence', 'diverse_perspectives', 'factual_accuracy', 'overall_preference']
        
        results = {}
        for criterion in criteria:
            method1_wins = 0
            method2_wins = 0
            ties = 0
            
            for comparison in pairwise_comparisons:
                if comparison.get('error'):
                    continue
                
                scores = comparison.get('comparison_scores', {})
                result = scores.get(criterion, 'tie')
                
                if result == method1_name:
                    method1_wins += 1
                elif result == method2_name:
                    method2_wins += 1
                else:
                    ties += 1
            
            total = method1_wins + method2_wins + ties
            if total > 0:
                results[criterion] = {
                    f'{method1_name}_wins': method1_wins,
                    f'{method2_name}_wins': method2_wins,
                    'ties': ties,
                    f'{method1_name}_win_rate': method1_wins / total,
                    f'{method2_name}_win_rate': method2_wins / total,
                    'tie_rate': ties / total
                }
        
        return results

    def _generate_comparison_summary(self, method1_avg: Dict, method2_avg: Dict, pairwise_results: Dict, 
                                   method1_name: str, method2_name: str) -> str:
        """Generate a summary of comparison results"""
        
        summary_parts = []
        summary_parts.append("EVALUATION COMPARISON SUMMARY")
        summary_parts.append("=" * 40)
        
        # Average scores comparison. An unscorable side carries an 'error' key instead of
        # criterion keys; printing it would render as 0.00 and read as a rating.
        unscorable = [
            name for name, avg in ((method1_name, method1_avg), (method2_name, method2_avg))
            if not avg or avg.get('error')
        ]
        if unscorable:
            summary_parts.append(f"Average Scores: unavailable ({', '.join(unscorable)} "
                                 "had no scorable answers)")
        else:
            summary_parts.append("Average Scores:")
            for criterion in ['comprehensiveness', 'human_enfranchisement', 'diversity', 'faithfulness', 'overall_score']:
                method1_score = method1_avg.get(criterion, 0)
                method2_score = method2_avg.get(criterion, 0)
                summary_parts.append(f"  {criterion.title()}: {method1_name} {method1_score:.2f} vs {method2_name} {method2_score:.2f}")
        
        # Pairwise results
        if pairwise_results:
            summary_parts.append("\nPairwise Comparison Results:")
            for criterion, results in pairwise_results.items():
                method2_rate = results.get(f'{method2_name}_win_rate', 0)
                summary_parts.append(f"  {criterion.replace('_', ' ').title()}: {method2_name} wins {method2_rate:.1%} of comparisons")
        
        return "\n".join(summary_parts)

    def _calculate_global_metrics(self, global_evaluations: List[Dict]) -> Dict:
        """Calculate metrics for global sensemaking evaluation"""
        if not global_evaluations:
            return {'total_evaluated': 0, 'error': 'no answers could be scored'}
        
        criteria = ['global_perspective', 'theme_identification', 'synthesis_quality', 'strategic_value', 'pattern_recognition']
        
        total_scores = {criterion: 0 for criterion in criteria}
        count = 0
        
        for evaluation in global_evaluations:
            if evaluation.get('error'):
                continue
            
            global_metrics = evaluation.get('global_metrics', {})
            for criterion in criteria:
                if criterion in global_metrics:
                    total_scores[criterion] += global_metrics[criterion]
            count += 1
        
        if count == 0:
            # Same trap as above: all-zero criterion scores are indistinguishable from a
            # genuinely awful arm, and 0 is outside the 1-5 rubric scale entirely.
            return {'total_evaluated': 0, 'error': 'no answers could be scored'}
        
        # Calculate averages
        average_scores = {key: score / count for key, score in total_scores.items()}
        average_scores['total_evaluated'] = count
        
        return average_scores
    
    def save_evaluation_results(self, results: Dict, output_path: str) -> None:
        """Save evaluation results to file (includes exact judge model provenance)."""
        try:
            if isinstance(results, dict):
                results = dict(results)
                results["model_provenance"] = self.get_model_provenance()
                results["evaluator_model"] = self.model
                results["evaluator_model_exact"] = self.exact_model_id()
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            meta_path = os.path.splitext(output_path)[0] + "_judge_meta.json"
            with open(meta_path, "w") as f:
                json.dump(self.get_model_provenance(), f, indent=2, default=str)
            print(f"Evaluation results saved to {output_path}")
            print(
                f"Judge model: requested={self.model} exact={self.exact_model_id()} → {meta_path}"
            )
        except Exception as e:
            print(f"Error saving evaluation results: {e}")
    
    def _determine_action_winner(self, comparison: Dict, method1_name: str = "vanilla", method2_name: str = "graphrag") -> Dict:
        """
        Determine the winner based on action-specific comparison metrics
        
        Args:
            comparison: Comparison metrics between two methods
            method1_name: Name of first method
            method2_name: Name of second method
        
        Returns:
            Winner information
        """
        # Count wins for each method across different metrics
        method1_wins = 0
        method2_wins = 0
        
        metrics_to_check = ['bleu_scores_mean', 'meteor_scores_mean', 'rouge_l_f1_scores_mean', 
                           'exact_matches_rate', 'llm_evaluation_scores_mean']
        
        for metric in metrics_to_check:
            metric_comparison = comparison.get(metric, {})
            winner = metric_comparison.get('winner', 'tie')
            if winner == method1_name:
                method1_wins += 1
            elif winner == method2_name:
                method2_wins += 1
        
        # Determine overall winner
        if method1_wins > method2_wins:
            overall_winner = method1_name
        elif method2_wins > method1_wins:
            overall_winner = method2_name
        else:
            overall_winner = "tie"
        
        total_comparisons = method1_wins + method2_wins
        method1_win_rate = method1_wins / total_comparisons if total_comparisons > 0 else 0
        method2_win_rate = method2_wins / total_comparisons if total_comparisons > 0 else 0
        
        return {
            'overall_winner': overall_winner,
            f'{method1_name}_wins': method1_wins,
            f'{method2_name}_wins': method2_wins,
            f'{method1_name}_win_rate': method1_win_rate,
            f'{method2_name}_win_rate': method2_win_rate,
            'total_metrics_compared': len(metrics_to_check)
        }
    
    def compare_action_methods_flexible(self,
                                      method1_answers: List[Dict],
                                      method2_answers: List[Dict],
                                      questions: List[Dict],
                                      method1_name: str = "method1",
                                      method2_name: str = "method2") -> Dict:
        """
        Compare two methods for action-specific questions with ground truth - flexible version
        
        Args:
            method1_answers: First method answers
            method2_answers: Second method answers
            questions: Questions with ground truth
            method1_name: Name of first method
            method2_name: Name of second method
        
        Returns:
            Comparison results for action-specific questions
        """
        print(f"Comparing {method1_name} vs {method2_name} on action-specific questions with ground truth...")
        
        # Evaluate both methods with detailed error tracking
        method1_results = self.evaluate_action_specific_answers(method1_answers, questions)
        if method1_results.get('error'):
            return {"error": f"Failed to evaluate {method1_name}: {method1_results.get('error')}"}
        
        method2_results = self.evaluate_action_specific_answers(method2_answers, questions)
        if method2_results.get('error'):
            return {"error": f"Failed to evaluate {method2_name}: {method2_results.get('error')}"}
        
        # Check if aggregate metrics exist
        if 'aggregate_metrics' not in method1_results:
            return {"error": f"{method1_name} results missing aggregate_metrics"}
        
        if 'aggregate_metrics' not in method2_results:
            return {"error": f"{method2_name} results missing aggregate_metrics"}
        
        # Compare aggregate metrics
        try:
            comparison = self._compare_action_metrics(
                method1_results['aggregate_metrics'],
                method2_results['aggregate_metrics']
            )
        except Exception as e:
            return {"error": f"Failed to compare metrics: {e}"}
        
        # Determine winner
        try:
            winner = self._determine_action_winner(comparison, method1_name, method2_name)
        except Exception as e:
            return {"error": f"Failed to determine winner: {e}"}
        
        return {
            f'{method1_name}_results': method1_results,
            f'{method2_name}_results': method2_results,
            'comparison': comparison,
            'winner': winner
        }
    
    def _compare_action_metrics(self, method1_metrics: Dict, method2_metrics: Dict) -> Dict:
        """
        Compare action-specific metrics between two methods
        
        Args:
            method1_metrics: Aggregate metrics from first method
            method2_metrics: Aggregate metrics from second method
            
        Returns:
            Detailed comparison results
        """
        
        comparison = {}
        
        # List of metrics to compare
        metrics_to_compare = [
            'bleu_scores_mean', 'meteor_scores_mean', 'rouge_l_f1_scores_mean', 
            'rouge_1_f1_scores_mean', 'rouge_2_f1_scores_mean', 'semantic_similarity_scores_mean',
            'exact_matches_rate', 'llm_evaluation_scores_mean'
        ]
        
        for metric in metrics_to_compare:
            method1_value = method1_metrics.get(metric, 0)
            method2_value = method2_metrics.get(metric, 0)
            
            # Determine winner for this metric
            if method1_value > method2_value:
                winner = "method1"  # Use generic names here
            elif method2_value > method1_value:
                winner = "method2"
            else:
                winner = "tie"
            
            comparison[metric] = {
                'method1_score': method1_value,
                'method2_score': method2_value,
                'winner': winner,
                'difference': abs(method1_value - method2_value)
            }
        
        return comparison
    


if __name__ == "__main__":
    # Example: Claude Code paper-parity judge (or set JUDGE_PROVIDER=anthropic)
    from judge_policy import resolve_judge_defaults

    provider, model, grade = resolve_judge_defaults()
    api_key = None
    if provider == "anthropic":
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not set")
    elif provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set")

    evaluator = SensemakingEvaluator(
        api_key, model=model, provider=provider, judge_grade=grade
    )
    
    # Load questions and answers if available
    try:
        with open("aviation_sensemaking_questions.json", 'r') as f:
            questions = json.load(f)
        print(f"Loaded {len(questions)} questions for evaluation")
        
        # Evaluate questions
        question_evaluations = evaluator.evaluate_questions(questions[:5])  # Sample
        evaluator.save_evaluation_results(
            {'question_evaluations': question_evaluations}, 
            "question_evaluations.json"
        )
        
        # Separate action-specific questions for ground truth evaluation
        action_questions = [q for q in questions if q.get('category') == 'action_specific']
        if action_questions:
            print(f"Found {len(action_questions)} action-specific questions with ground truth")
        
    except FileNotFoundError:
        print("Questions file not found")
    
    # Load and evaluate answers if available
    try:
        with open("vanilla_answers.json", 'r') as f:
            vanilla_answers = json.load(f)
        
        with open("graphrag_answers.json", 'r') as f:
            graphrag_answers = json.load(f)
        
        # Standard comparison for non-action questions
        non_action_questions = [q for q in questions if q.get('category') != 'action_specific']
        if non_action_questions:
            comparison_results = evaluator.compare_answer_methods(
                vanilla_answers, graphrag_answers, non_action_questions[:5]
            )
            evaluator.save_evaluation_results(comparison_results, "method_comparison_results.json")
            print("Standard Evaluation Results:")
            print(comparison_results.get('evaluation_summary', 'No summary available'))
        
        # Action-specific evaluation with ground truth
        if action_questions:
            print("\n" + "="*50)
            print("ACTION-SPECIFIC EVALUATION WITH GROUND TRUTH")
            print("="*50)
            
            # Evaluate action-specific answers with ground truth metrics
            action_comparison = evaluator.compare_action_specific_methods(
                vanilla_answers, graphrag_answers, action_questions
            )
            
            if not action_comparison.get('error'):
                evaluator.save_evaluation_results(action_comparison, "action_specific_evaluation.json")
                
                print("\nAction-Specific Results Summary:")
                print(action_comparison['vanilla_results'].get('evaluation_summary', ''))
                print("\nGraphRAG vs Vanilla Comparison:")
                winner_info = action_comparison.get('winner', {})
                print(f"Overall Winner: {winner_info.get('overall_winner', 'Unknown')}")
                print(f"GraphRAG Win Rate: {winner_info.get('graphrag_win_rate', 0):.1%}")
                print(f"Vanilla Win Rate: {winner_info.get('vanilla_win_rate', 0):.1%}")
            else:
                print("Error in action-specific evaluation:", action_comparison.get('error'))
        
        # Global sensemaking evaluation
        global_questions = [q for q in questions if q.get('type') == 'global']
        if global_questions:
            print("\n" + "="*50)
            print("GLOBAL SENSEMAKING EVALUATION")
            print("="*50)
            
            global_results = evaluator.evaluate_global_sensemaking_capability(
                global_questions, graphrag_answers  # GraphRAG should be better for global questions
            )
            
            if not global_results.get('error'):
                evaluator.save_evaluation_results(global_results, "global_sensemaking_evaluation.json")
                print(f"Evaluated {global_results.get('global_questions_evaluated', 0)} global questions")
        
    except FileNotFoundError:
        print("Answer files not found. Generate answers first using answer_generator.py")
    
    print("\nEvaluation complete!")
    print("\nFiles generated:")
    print("- question_evaluations.json")
    print("- method_comparison_results.json")
    print("- action_specific_evaluation.json (if action questions found)")
    print("- global_sensemaking_evaluation.json (if global questions found)")
