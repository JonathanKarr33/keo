"""
GraphRAG-style Answer Generator for Aviation Maintenance Sensemaking Questions
Generates answers using both vanilla LLM and GraphRAG approaches
"""

import os
import json
import math
import random
import pandas as pd
import networkx as nx
from typing import List, Dict, Optional, Tuple, Any, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from openai import OpenAI
import numpy as np
from tqdm import tqdm
import time
import sys
import hashlib
import pickle
sys.path.append('..')
from graph_rag.KEO_GraphRAG import GraphRetriever
from local_embeddings import EmbeddingBackend
from task_router import route_question, explain_route
from model_provenance import (
    build_run_provenance,
    embedding_provenance,
    observe_completion_model,
)
from anthropic_client import (
    ANTHROPIC_AVAILABLE,
    DEFAULT_ANSWER_MODEL as ANTHROPIC_DEFAULT_ANSWER,
    anthropic_chat,
    make_anthropic_client,
)
from openrouter_client import (
    PaidAPIInterrupt,
    looks_like_paid_interrupt,
    make_openrouter_client,
    openrouter_model,
)

try:
    from huggingface_hub import InferenceClient
    HUGGINGFACE_AVAILABLE = True
except ImportError:
    HUGGINGFACE_AVAILABLE = False


# Severity lexicon for rarity-aware weighting (aviation safety priors).
# Lives in severity_lexicon so the rarity ablation and the risk-weighted retrieval
# objective score facts identically. Multipliers for the original terms are pinned
# there, so --edge-weight rarity is unchanged.
from severity_lexicon import SEVERITY_TERMS, edge_severity  # noqa: E402
from risk_retrieval import (  # noqa: E402
    edge_tokens,
    greedy_budgeted_coverage,
    safety_metrics,
)

# Context-selection policies for the graph arm
SELECTORS = ("mst", "subgraph", "risk")

# Chat-template turn terminators. Some local servers (observed with mlx_lm serving
# mlx-community/Phi-4-mini-instruct-4bit) decode these into the response text, so every
# answer would end with a literal "<|end|>" and the judge would score it as content. Sent
# as stop sequences, and stripped afterwards for servers that ignore stop.
CHAT_TERMINATORS = (
    "<|end|>",
    "<|im_end|>",
    "<|eot_id|>",
    "<|end_of_turn|>",
    "<end_of_turn>",
    "<|endoftext|>",
)



def gen_workers() -> int:
    """Parallel answer generation workers (API-bound OpenRouter etc.).

    Default 1 keeps local answerers sequential. Set KEO_GEN_WORKERS=10 for OpenRouter.
    Capped at 16. Same role as KEO_JUDGE_WORKERS for judging.
    """
    raw = os.getenv("KEO_GEN_WORKERS", "1").strip()
    try:
        workers = int(raw)
    except ValueError:
        workers = 1
    return max(1, min(workers, 16))


class SensemakingAnswerGenerator:
    def __init__(
        self,
        api_key: str,
        model: str = "gpt-5.6-luna",
        cache_dir: str = "./text_chunk_cache",
        provider: str = "openai",
        API_provider: str = "auto",
        use_mst: bool = True,
        edge_weight: str = "freq",
        query_weight_alpha: float = 1.0,
        embedding_backend: str = "local",
        embedding_model: Optional[str] = None,
        emit_evidence: bool = True,
        base_url: Optional[str] = None,
        completion_cache_dir: str = "./completion_cache",
        use_completion_cache: bool = True,
        max_answer_tokens: Optional[int] = None,
        text_column: Optional[str] = None,
        shuffle_edges: bool = False,
        strip_relations: bool = False,
        dedup_chunks: bool = False,
        match_context_budget: bool = False,
        control_seed: int = 13,
        selector: str = "mst",
        risk_budget_tokens: int = 1200,
        risk_floor: float = 0.25,
        risk_fill_budget: bool = True,
    ):
        """
        Initialize the answer generator
        
        Args:
            api_key: API key (OpenAI, HuggingFace, or dummy for local)
            model: Model to use for generation
            cache_dir: Directory to store embedding cache files
            provider: openai | anthropic | huggingface | local | openrouter
            API_provider: HF InferenceClient provider (nebius, etc.)
            use_mst / edge_weight / query_weight_alpha: retrieval ablations
            embedding_backend: openai | local | hash
            embedding_model: optional embedder model name
            emit_evidence: attach evidence cards
            base_url: OpenAI-compatible base URL for provider=local (vLLM/Ollama)
                or provider=openrouter (defaults to https://openrouter.ai/api/v1)
            completion_cache_dir: disk cache for chat completions (cuts re-run cost)
            use_completion_cache: enable disk completion cache
            max_answer_tokens: optional hard cap (defaults: 300 action / 600 sense)
            text_column: optional corpus text column (else auto: c119/narrative/text)

        Mechanism controls (isolate *why* graph context helps; see papers/ACL_STRATEGY.md):
            shuffle_edges: permute edge endpoints at KG load (keeps entities/degrees,
                destroys true relational pairings) — tests structure vs entity coverage
            strip_relations: serialize connectivity without relation labels
            dedup_chunks: drop near-duplicate chunks before filling the chunk budget
            match_context_budget: give chunk retrieval the same per-question character
                budget the graph path would have consumed
            control_seed: RNG seed for shuffle_edges (reproducible permutation)

        Context selection policy (papers/ACL_STRATEGY.md §3 Pivot C):
            selector: "mst" (default, current KEO) | "subgraph" (no MST) | "risk"
                (risk-weighted budgeted coverage — maximise rel(e,q)·sev(e) over covered
                entities subject to a token budget, with a reserved share for
                high-severity edges). "risk" ignores use_mst.
            risk_budget_tokens: token budget B for selector="risk"
            risk_floor: fraction of B reserved for high-severity edges before general
                selection, which converts the severity preference into a floor
            risk_fill_budget: after the objective saturates, spend the leftover budget on
                the next-best edges. On by default because judged comparisons are
                length-sensitive and this holds context length roughly constant; turn it
                off to see how little context the objective actually needs.
        """
        if selector not in SELECTORS:
            raise ValueError(f"selector must be one of {SELECTORS}, got {selector!r}")
        self.provider = provider
        self.model = model
        self.cache_dir = cache_dir
        self.embedding_cache = {}
        self.chunk_cache = {}
        self.use_mst = use_mst
        self.edge_weight = edge_weight
        self.query_weight_alpha = query_weight_alpha
        self.embedding_backend_name = embedding_backend
        self.emit_evidence = emit_evidence
        self.text_column = text_column
        self.base_url = base_url or os.getenv("LOCAL_LLM_BASE_URL", "http://127.0.0.1:8000/v1")
        self.use_completion_cache = use_completion_cache
        self.completion_cache_dir = completion_cache_dir
        self.max_answer_tokens = max_answer_tokens
        self.shuffle_edges = shuffle_edges
        self.strip_relations = strip_relations
        self.dedup_chunks = dedup_chunks
        self.match_context_budget = match_context_budget
        self.control_seed = control_seed
        self.selector = selector
        self.risk_budget_tokens = risk_budget_tokens
        self.risk_floor = risk_floor
        self.risk_fill_budget = risk_fill_budget
        self._risk_selection_stats: List[Dict[str, Any]] = []
        self._base_edge_weights = {}
        self._tls = threading.local()
        self._gen_lock = threading.RLock()  # graph reweight + completion-cache writes
        self._progress_lock = threading.Lock()
        self.gen_workers = gen_workers()
        self._last_evidence = {}
        self._completion_mem: Dict[str, str] = {}
        self._observed_model_ids: set = set()
        self.API_provider = API_provider
        
        if provider == "openai":
            self.client = OpenAI(api_key=api_key)
        elif provider == "anthropic":
            self.client = make_anthropic_client(api_key)
            print(f"Anthropic client model={model}")
        elif provider == "huggingface":
            if not HUGGINGFACE_AVAILABLE:
                raise ImportError("huggingface_hub is not installed. Please install it with: pip install huggingface_hub")
            self.client = InferenceClient(
                provider=API_provider,
                api_key=api_key,
            )
        elif provider == "local":
            # vLLM / Ollama / LM Studio OpenAI-compatible server
            self.client = OpenAI(api_key=api_key or "EMPTY", base_url=self.base_url)
            print(f"Local LLM client → {self.base_url} model={model}")
        elif provider == "openrouter":
            # Larger hosted answerer. Key comes from OPENROUTER_API_KEY; no call is made here.
            # The class default model is an OpenAI id; swap it for the OpenRouter Gemma default
            # unless the caller passed a model or set OPENROUTER_MODEL.
            self.base_url = base_url or os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
            if not (model or "").strip() or model == "gpt-5.6-luna":
                model = openrouter_model()
                self.model = model
            self.client = make_openrouter_client(api_key, base_url=self.base_url)
            print(f"OpenRouter client → {self.base_url} model={self.model}")
        else:
            raise ValueError(
                f"Unsupported provider: {provider}. "
                "Supported: 'openai', 'anthropic', 'huggingface', 'local', 'openrouter'"
            )
        
        openai_key = os.getenv("OPENAI_API_KEY")
        self.embedder = EmbeddingBackend(
            backend=embedding_backend,
            model=embedding_model,
            openai_api_key=openai_key if embedding_backend == "openai" else None,
        )
        # Sync if ST load fell back to hash
        self.embedding_backend_name = self.embedder.backend
        self.embedding_client = self.embedder
        
        self.knowledge_graph = None
        self.graph_retriever = None
        
        os.makedirs(cache_dir, exist_ok=True)
        if use_completion_cache:
            os.makedirs(completion_cache_dir, exist_ok=True)
        self._load_cache()

        # Resolve exact model IDs once at init (HF SHA / OpenAI card / local /v1/models)
        self._run_provenance = build_run_provenance(
            role="answerer",
            provider=provider,
            model_id=model,
            client=self.client,
            api_provider=API_provider if provider == "huggingface" else None,
            base_url=self.base_url if provider == "local" else None,
            embedding=embedding_provenance(self.embedder),
            observed_model_ids=self._observed_model_ids,
            extra={
                "use_mst": use_mst,
                "edge_weight": edge_weight,
                "embedding_backend": embedding_backend,
                "controls": self.control_config(),
                "selection": self.selection_config(),
            },
        )
        exact = self._run_provenance["llm"].get("exact_id") or model
        print(f"Answer model: requested={model} exact={exact}")

    def selection_config(self) -> Dict[str, Any]:
        """Active context-selection policy, for provenance."""
        config: Dict[str, Any] = {"selector": self.selector, "use_mst": self.use_mst}
        if self.selector == "risk":
            config["risk_budget_tokens"] = self.risk_budget_tokens
            config["risk_floor"] = self.risk_floor
            config["risk_fill_budget"] = self.risk_fill_budget
        return config

    def control_config(self) -> Dict[str, Any]:
        """Active mechanism controls (all False = the standard KEO / chunk arms)."""
        return {
            "shuffle_edges": self.shuffle_edges,
            "strip_relations": self.strip_relations,
            "dedup_chunks": self.dedup_chunks,
            "match_context_budget": self.match_context_budget,
            "control_seed": self.control_seed if self.shuffle_edges else None,
        }

    def get_model_provenance(self) -> Dict[str, Any]:
        """Full run provenance (requested + resolved + observed snapshot IDs)."""
        # Refresh observed IDs into a fresh snapshot
        return build_run_provenance(
            role="answerer",
            provider=self.provider,
            model_id=self.model,
            client=self.client,
            api_provider=self.API_provider if self.provider == "huggingface" else None,
            base_url=self.base_url if self.provider == "local" else None,
            embedding=embedding_provenance(self.embedder),
            observed_model_ids=self._observed_model_ids,
            extra=self._run_provenance.get("extra"),
        )


    @property
    def _last_evidence(self) -> Dict[str, Any]:
        ev = getattr(self._tls, "last_evidence", None)
        if ev is None:
            ev = {}
            self._tls.last_evidence = ev
        return ev

    @_last_evidence.setter
    def _last_evidence(self, value: Dict[str, Any]) -> None:
        self._tls.last_evidence = value if value is not None else {}

    def exact_model_id(self) -> str:
        if self._observed_model_ids:
            return sorted(self._observed_model_ids)[-1]
        llm = (self._run_provenance or {}).get("llm") or {}
        return llm.get("exact_id") or llm.get("resolved_id") or self.model

    def _model_fields(self) -> Dict[str, Any]:
        """Compact per-answer model identity for reproducibility."""
        return {
            "model": self.model,
            "model_exact": self.exact_model_id(),
            "model_provider": self.provider,
        }

    def _local_extra_body(self) -> Dict[str, Any]:
        """Host kwargs to disable thinking on local reasoning models when supported.

        Left enabled, a reasoning model spends the completion budget in its separate
        `reasoning` field and returns a truncated or absent `content` — see the empty
        completion guard in _create_chat_completion.
        """
        mid = (self.model or "").lower()
        if "qwen" in mid or "gemma-4" in mid or "gemma4" in mid:
            return {"chat_template_kwargs": {"enable_thinking": False}}
        return {}

    @staticmethod
    def _strip_thinking(text: str) -> str:
        """Drop <think>...</think> blocks if a model ignored disable-thinking."""
        if not text:
            return text
        import re
        cleaned = re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.IGNORECASE).strip()
        return cleaned or text

    @staticmethod
    def _strip_chat_terminators(text: str) -> str:
        """Remove leaked chat-template turn markers (see CHAT_TERMINATORS)."""
        if not text:
            return text
        cleaned = text
        for marker in CHAT_TERMINATORS:
            cleaned = cleaned.replace(marker, "")
        return cleaned.strip()
        
    def load_knowledge_graph(self, graph_path: str) -> bool:
        """Load aviation maintenance knowledge graph"""
        try:
            self.knowledge_graph = nx.read_gml(graph_path)
            if self.shuffle_edges:
                self.knowledge_graph = self._permute_graph_edges(self.knowledge_graph)
            emb_model = getattr(self.embedder, "model", "") or "default"
            emb_tag = hashlib.md5(emb_model.encode()).hexdigest()[:8]
            kg_cache_dir = './kg_cache/temp_cache.json'
            if "kg/output/knowledge_graph.gml" in graph_path:
                kg_cache_dir = f'./kg_cache/kg_gs100_cache_{self.embedding_backend_name}_{emb_tag}.json'
            else:
                graph_dir = os.path.dirname(graph_path)
                parts = graph_dir.split('/')
                if len(parts) >= 2:
                    kg_cache_dir = (
                        f'./kg_cache/{parts[-2]}/{parts[-1]}/'
                        f'kg_cache_{self.embedding_backend_name}_{emb_tag}.json'
                    )
            self.graph_retriever = GraphRetriever(
                self.knowledge_graph,
                os.getenv("OPENAI_API_KEY") or "",
                kg_cache_dir,
                embedding_backend=self.embedding_backend_name,
                embedding_model=getattr(self.embedder, "model", None),
                embedder=self.embedder,
            )
            self.graph_retriever.generate_embeddings()
            self._store_base_edge_weights()
            self._apply_edge_weight_scheme(self.edge_weight)
            print(
                f"Knowledge graph loaded: {self.knowledge_graph.number_of_nodes()} nodes, "
                f"{self.knowledge_graph.number_of_edges()} edges "
                f"(use_mst={self.use_mst}, edge_weight={self.edge_weight}, "
                f"emb={self.embedding_backend_name})"
            )
            return True
        except Exception as e:
            print(f"Error loading knowledge graph: {e}")
            return False

    def _permute_graph_edges(self, graph: nx.Graph) -> nx.Graph:
        """
        Edge-permutation control (G-shuf).

        Reassigns each edge's tail to a node sampled proportional to its original
        degree. Preserves the node set, node attributes, edge count, per-head
        out-degree, and the multiset of relation labels and weights; destroys the
        true head-tail pairings. If KEO's advantage survives this, the advantage
        comes from entity coverage rather than from real relational structure.
        """
        rng = random.Random(self.control_seed)
        nodes = list(graph.nodes())
        if len(nodes) < 3:
            return graph

        degrees = [max(1, graph.degree(n)) for n in nodes]
        permuted = graph.__class__()
        permuted.add_nodes_from(graph.nodes(data=True))

        edges = (
            graph.edges(keys=True, data=True)
            if graph.is_multigraph()
            else graph.edges(data=True)
        )
        for edge in edges:
            head = edge[0]
            data = dict(edge[-1])
            tail = head
            allow_parallel = graph.is_multigraph()
            for _ in range(10):
                candidate = rng.choices(nodes, weights=degrees, k=1)[0]
                if candidate == head:
                    continue
                if allow_parallel or not permuted.has_edge(head, candidate):
                    tail = candidate
                    break
            if tail == head:
                continue
            permuted.add_edge(head, tail, **data)

        print(
            f"[control] edge permutation (seed={self.control_seed}): "
            f"{graph.number_of_edges()} → {permuted.number_of_edges()} edges, "
            f"{permuted.number_of_nodes()} nodes preserved"
        )
        return permuted

    def _store_base_edge_weights(self) -> None:
        """Snapshot original edge weights for reweighting ablations."""
        self._base_edge_weights = {}
        if self.knowledge_graph is None:
            return
        # Support both simple and multi graphs
        if self.knowledge_graph.is_multigraph():
            edge_iter = self.knowledge_graph.edges(keys=True, data=True)
            for u, v, key, data in edge_iter:
                w = data.get("weight", 1.0)
                try:
                    w = float(w)
                except (TypeError, ValueError):
                    w = 1.0
                self._base_edge_weights[(u, v, key)] = w
                data["weight"] = w
                data["base_weight"] = w
        else:
            for u, v, data in self.knowledge_graph.edges(data=True):
                w = data.get("weight", 1.0)
                try:
                    w = float(w)
                except (TypeError, ValueError):
                    w = 1.0
                self._base_edge_weights[(u, v, 0)] = w
                data["weight"] = w
                data["base_weight"] = w

    def _apply_edge_weight_scheme(self, scheme: str, question: Optional[str] = None) -> None:
        """
        Reweight KG edges in-place.
        freq | tfidf | pmi | query | rarity
        rarity = PMI × severity prior (boost rare but safety-critical links)
        """
        if self.knowledge_graph is None or not self._base_edge_weights:
            return
        scheme = (scheme or "freq").lower()
        n_nodes = max(self.knowledge_graph.number_of_nodes(), 1)
        degrees = dict(self.knowledge_graph.degree())
        total_weight = sum(self._base_edge_weights.values()) or 1.0

        query_sims = {}
        if scheme == "query" and question and self.graph_retriever:
            try:
                q_emb = self.embedder.embed(question)
                for node in self.knowledge_graph.nodes():
                    if node in self.graph_retriever.embeddings:
                        query_sims[node] = self.graph_retriever.calculate_semantic_similarity(
                            q_emb, self.graph_retriever.embeddings[node]
                        )
            except Exception as e:
                print(f"Warning: query reweighting fell back to freq ({e})")
                scheme = "freq"

        def _severity(u: str, v: str) -> float:
            text = f"{u} {v}".lower()
            score = 1.0
            for term, boost in SEVERITY_TERMS.items():
                if term in text:
                    score = max(score, boost)
            return score

        def _new_weight(u, v, base):
            if scheme == "freq":
                return base
            if scheme == "tfidf":
                idf_u = np.log((n_nodes + 1) / (degrees.get(u, 1) + 1)) + 1.0
                idf_v = np.log((n_nodes + 1) / (degrees.get(v, 1) + 1)) + 1.0
                return base * 0.5 * (idf_u + idf_v)
            if scheme == "pmi":
                p_uv = (base + 1e-3) / (total_weight + 1e-3)
                deg_sum = sum(degrees.values()) + 1e-3
                p_u = (degrees.get(u, 1) + 1e-3) / deg_sum
                p_v = (degrees.get(v, 1) + 1e-3) / deg_sum
                return max(np.log(p_uv / (p_u * p_v + 1e-12)), 0.0) * base
            if scheme == "query":
                sim = max(query_sims.get(u, 0.0), query_sims.get(v, 0.0))
                return base * (1.0 + self.query_weight_alpha * sim)
            if scheme == "rarity":
                # Invert frequency popularity, then apply severity prior
                inv = 1.0 / (1.0 + base)
                p_uv = (base + 1e-3) / (total_weight + 1e-3)
                deg_sum = sum(degrees.values()) + 1e-3
                p_u = (degrees.get(u, 1) + 1e-3) / deg_sum
                p_v = (degrees.get(v, 1) + 1e-3) / deg_sum
                pmi = max(np.log(p_uv / (p_u * p_v + 1e-12)), 0.0)
                return (inv + pmi) * _severity(u, v)
            return base

        if self.knowledge_graph.is_multigraph():
            for u, v, key, data in self.knowledge_graph.edges(keys=True, data=True):
                base = self._base_edge_weights.get((u, v, key), float(data.get("weight", 1.0)))
                data["weight"] = float(_new_weight(u, v, base))
        else:
            for u, v, data in self.knowledge_graph.edges(data=True):
                base = self._base_edge_weights.get((u, v, 0), float(data.get("weight", 1.0)))
                data["weight"] = float(_new_weight(u, v, base))

    def _create_chat_completion(self, messages: List[Dict], max_tokens: int = 1000, temperature: float = 0.3) -> str:
        """
        Create a chat completion using the configured provider.
        Uses disk cache when enabled to avoid re-paying for identical prompts.
        """
        if self.max_answer_tokens is not None:
            max_tokens = min(max_tokens, self.max_answer_tokens)

        cache_key = None
        if self.use_completion_cache:
            blob = json.dumps(
                {
                    "provider": self.provider,
                    "api_provider": getattr(self, "API_provider", None),
                    "base_url": self.base_url if self.provider in ("local", "openrouter") else None,
                    "model": self.model,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    # Host kwargs change the reply for the same prompt (thinking on vs off),
                    # so they belong in the key or the two answers collide.
                    "extra_body": self._local_extra_body() if self.provider == "local" else None,
                    "cache_version": 3,
                },
                sort_keys=True,
            )
            cache_key = hashlib.md5(blob.encode()).hexdigest()
            if cache_key in self._completion_mem:
                return self._completion_mem[cache_key]
            path = os.path.join(self.completion_cache_dir, f"{cache_key}.txt")
            if os.path.exists(path):
                with open(path, "r") as f:
                    text = f.read()
                self._completion_mem[cache_key] = text
                return text

        if self.provider == "openai":
            mid = (self.model or "").lower()
            kwargs = {"model": self.model, "messages": messages}
            if mid.startswith("gpt-5") or mid.startswith("gpt-6") or mid.startswith("o1") or mid.startswith("o3"):
                kwargs["max_completion_tokens"] = max_tokens
                kwargs["reasoning_effort"] = "none"
            else:
                kwargs["max_tokens"] = max_tokens
                kwargs["temperature"] = temperature
            response = self.client.chat.completions.create(**kwargs)
            observed = observe_completion_model(response)
            if observed:
                self._observed_model_ids.add(observed)
            text = (response.choices[0].message.content or "").strip()
            text = self._strip_thinking(text)
        elif self.provider == "openrouter":
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
            except Exception as exc:
                if looks_like_paid_interrupt(str(exc), type(exc).__name__):
                    raise PaidAPIInterrupt(
                        f"OpenRouter hard limit while calling {self.model}: {exc}. "
                        "Checkpointed answers are kept; re-run the same arm to resume."
                    ) from exc
                raise
            observed = observe_completion_model(response)
            if observed:
                self._observed_model_ids.add(observed)
            text = (response.choices[0].message.content or "").strip()
            if not text:
                raise RuntimeError(f"keo: empty completion from OpenRouter model {self.model}")
            text = self._strip_thinking(text)
        elif self.provider == "anthropic":
            text, observed, _meta = anthropic_chat(
                self.client,
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                disable_thinking=False,  # answers may benefit from thinking models
            )
            if observed:
                self._observed_model_ids.add(observed)
            text = self._strip_thinking(text)
        elif self.provider == "local":
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=list(CHAT_TERMINATORS),
                extra_body=self._local_extra_body(),
            )
            observed = observe_completion_model(response)
            if observed:
                self._observed_model_ids.add(observed)
            message = response.choices[0].message
            text = (message.content or "").strip()
            if not text:
                # A reasoning model that used the whole budget thinking returns an empty or
                # absent `content`. Coercing that to "" writes a blank answer with no error,
                # so the arm looks complete and the judge scores silence.
                reasoning = (getattr(message, "reasoning", None) or "").strip()
                detail = (
                    f"the model returned {len(reasoning)} characters of `reasoning` and no answer; "
                    "the completion budget was spent thinking. Disable thinking for this model in "
                    "_local_extra_body, or raise --max-answer-tokens above the reasoning length."
                    if reasoning
                    else "the model returned no content and no reasoning."
                )
                raise RuntimeError(f"keo: empty completion from local model {self.model}: {detail}")
            text = self._strip_thinking(text)
            text = self._strip_chat_terminators(text)
        elif self.provider == "huggingface":
            create_kwargs = dict(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            extra = self._local_extra_body()
            if extra:
                create_kwargs["extra_body"] = extra
            response = self.client.chat.completions.create(**create_kwargs)
            observed = observe_completion_model(response)
            if observed:
                self._observed_model_ids.add(observed)
            text = (response.choices[0].message.content or "").strip()
            text = self._strip_thinking(text)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

        if cache_key and self.use_completion_cache:
            with self._gen_lock:
                self._completion_mem[cache_key] = text
                path = os.path.join(self.completion_cache_dir, f"{cache_key}.txt")
                with open(path, "w") as f:
                    f.write(text)
        return text

    @staticmethod
    def _usable_answer_record(ans: Optional[Dict]) -> bool:
        if not isinstance(ans, dict) or ans.get("error"):
            return False
        return bool(str(ans.get("answer") or "").strip())

    def _iter_generate(
        self,
        questions: List[Dict],
        *,
        desc: str,
        produce: Callable[[Dict], Dict],
        method_label: str,
        prior_answers: Optional[List[Dict]] = None,
        on_progress: Optional[Callable[[List[Dict]], None]] = None,
    ) -> List[Dict]:
        """
        Generate answers with resume + progress checkpoints.

        prior_answers: already-paid successes to keep (skipped on re-run).
        on_progress: flushes answers.json so a paid-API abort does not drop work.
        Parallelism via KEO_GEN_WORKERS (default 1; use 10 for OpenRouter).
        """
        prior_ok: Dict[str, Dict] = {}
        for ans in prior_answers or []:
            if not self._usable_answer_record(ans):
                continue
            qid = ans.get("question_id")
            if qid:
                prior_ok[qid] = ans

        ordered_ids = [q.get("id", "") for q in questions]
        todo = [q for q in questions if q.get("id", "") not in prior_ok]
        by_id: Dict[str, Dict] = dict(prior_ok)

        def snapshot() -> List[Dict]:
            return [by_id[i] for i in ordered_ids if i in by_id]

        def record(qid: str, ans: Dict) -> None:
            with self._progress_lock:
                if qid:
                    by_id[qid] = ans
                snap = snapshot() if on_progress is not None else None
            if on_progress is not None and snap is not None:
                on_progress(snap)

        def run_one(question_data: Dict) -> Dict:
            qid = question_data.get("id", "")
            try:
                if self.edge_weight in ("query", "rarity"):
                    with self._gen_lock:
                        return produce(question_data)
                return produce(question_data)
            except PaidAPIInterrupt:
                raise
            except Exception as e:
                print(f"Error generating {method_label} answer for question {qid}: {e}")
                if looks_like_paid_interrupt(str(e), type(e).__name__):
                    raise PaidAPIInterrupt(
                        f"OpenRouter hard limit on {qid}: {e}. Re-run to resume."
                    ) from e
                return {
                    "question_id": qid,
                    "question": question_data.get("question", ""),
                    "answer": f"Error generating answer: {e}",
                    "method": method_label,
                    "error": True,
                }

        if prior_ok:
            print(f"  resume: keeping {len(prior_ok)} answers, {len(todo)} remaining")
            if on_progress is not None:
                on_progress(snapshot())

        workers = min(self.gen_workers, len(todo)) if todo else 1
        try:
            if workers <= 1:
                for question_data in tqdm(todo, desc=desc):
                    qid = question_data.get("id", "")
                    record(qid, run_one(question_data))
            else:
                print(f"  generating with KEO_GEN_WORKERS={workers}")
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    futures = {pool.submit(run_one, q): q.get("id", "") for q in todo}
                    with tqdm(total=len(todo), desc=f"{desc} ({workers}x)") as bar:
                        for fut in as_completed(futures):
                            qid = futures[fut]
                            try:
                                ans = fut.result()
                            except PaidAPIInterrupt:
                                for other in futures:
                                    other.cancel()
                                if on_progress is not None:
                                    with self._progress_lock:
                                        snap = snapshot()
                                    on_progress(snap)
                                    print(f"  paid-API stop — checkpointed {len(snap)} answers")
                                raise
                            record(qid, ans)
                            bar.update(1)
        except PaidAPIInterrupt:
            raise

        return snapshot()

    def generate_vanilla_answers(self, 
                               questions: List[Dict],
                               prior_answers: Optional[List[Dict]] = None,
                               on_progress: Optional[Callable[[List[Dict]], None]] = None) -> List[Dict]:
        """
        Generate answers using vanilla LLM with simple prompts
        Args:
            questions: List of question dictionaries
            prior_answers: usable answers to keep on resume
            on_progress: optional checkpoint callback
        Returns:
            List of answers with metadata
        """
        print("Generating vanilla LLM answers...")

        def produce(question_data: Dict) -> Dict:
            question = question_data['question']
            question_type = question_data.get('type', '')

            if question_type == 'actionable':
                prompt = f"""
You are an expert aviation maintenance technician. Answer the following question with a direct, concise action.

Question: {question}

Provide a brief, specific answer that:
1. States the exact maintenance action to take
2. Is clear and actionable
3. Follows standard aviation maintenance procedures, and only gives the action to take

Answer:
"""
            else:
                prompt = f"""
You are an expert aviation safety analyst. Answer the following question about aviation maintenance.

Question: {question}

Provide a simple and concise answer that:
1. Directly addresses the question
2. Provides actionable insights
3. Maintains focus on aviation safety and maintenance

Answer:
"""

            answer = self._create_chat_completion(
                messages=[
                    {"role": "system", "content": "You are an expert aviation safety analyst with deep knowledge of maintenance practices, failure analysis, and safety recommendations."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=250 if question_type == 'actionable' else 600,
                temperature=0.3
            )

            return {
                'question_id': question_data.get('id', ''),
                'question': question,
                'category': question_data.get('category', ''),
                'answer': answer,
                'method': 'vanilla_llm',
                **self._model_fields(),
            }

        return self._iter_generate(
            questions,
            desc="Generating vanilla answers",
            produce=produce,
            method_label="vanilla_llm",
            prior_answers=prior_answers,
            on_progress=on_progress,
        )


    def generate_textchunkrag_answers(self, 
                               questions: List[Dict],
                               datasets: Dict[str, pd.DataFrame],
                               context_window_size: int = 4000,
                               method_label: str = "textchunkrag",
                               prior_answers: Optional[List[Dict]] = None,
                               on_progress: Optional[Callable[[List[Dict]], None]] = None) -> List[Dict]:
        """
        Generate answers using text chunking based RAG approach
        
        Args:
            questions: List of question dictionaries
            datasets: Aviation maintenance datasets
            context_window_size: Max characters for context
            method_label: Stored in answer records (textchunkrag / chunk_tok / chunk_dedup)
            prior_answers: usable answers to keep on resume
            on_progress: optional checkpoint callback
        
        Returns:
            List of answers with metadata
        """
        print(f"Generating {method_label} answers...")

        def produce(question_data: Dict) -> Dict:
            question = question_data['question']
            question_type = question_data.get('type', '')

            relevant_context, budget, top_k = self._chunk_context_for(
                question, datasets, context_window_size
            )

            if question_type == 'actionable':
                prompt = f"""
You are an expert aviation maintenance technician. Answer the following question with a direct, concise action based on your own knowledge and the provided aviation maintenance records.

Question: {question}

Possibly Related Aviation Maintenance Records:
{relevant_context}

Provide a brief answer that:
1. States the exact maintenance action to take
2. Is clear and actionable
3. Directly gives the action to take, not the reasoning or background information
4. Uses information from the Aviation Maintenance Records if they are relevant to the question, but not solely based on them, also use your own knowledge

One example:
Q: What action could be taken when: #2 & 4 rocker covers are leaking?
A: Remove & replace #4 gaskets.

Answer:
"""
            else:
                prompt = f"""
You are an expert aviation safety analyst. Answer the following question by using your own knowledge and the provided aviation maintenance records.

Question: {question}

Possibly Related Aviation Maintenance Records:
{relevant_context}

Provide a concise and simple answer that:
1. Directly addresses the question
2. Uses specific information from the Aviation Maintenance records if they are relevant, but do not rely solely on them, also use your own knowledge, and the answer should be high-level and analytical
3. Provides actionable insights, but not specific analysis or reasoning on the records
4. Maintains focus on aviation safety and maintenance

One example:
Q: To what extent does the inspection and maintenance of aircraft wiring systems influence avionics reliability and flight safety?
A: Regular inspections help identify potential issues such as wear, corrosion, or insulation damage that could lead to electrical failures. Proper maintenance practices, including timely repairs and replacements, significantly reduce the risk of in-flight malfunctions, thereby enhancing overall safety.

Answer:
"""

            answer = self._create_chat_completion(
                messages=[
                    {"role": "system", "content": "You are an expert aviation safety analyst with deep knowledge of maintenance practices, failure analysis, and safety recommendations."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=250 if question_type == 'actionable' else 600,
                temperature=0.3
            )

            return {
                'question_id': question_data.get('id', ''),
                'question': question,
                'category': question_data.get('category', ''),
                'answer': answer,
                'method': method_label,
                'context_used': len(relevant_context),
                'context_budget': budget,
                'chunks_retrieved': top_k,
                'controls': self.control_config(),
                **self._model_fields(),
            }

        return self._iter_generate(
            questions,
            desc=f"Generating {method_label} answers",
            produce=produce,
            method_label=method_label,
            prior_answers=prior_answers,
            on_progress=on_progress,
        )

    def _chunk_budget(
        self,
        question: str,
        datasets: Dict[str, pd.DataFrame],
        default_chars: int,
        preview_chars: int = 200,
    ) -> Tuple[int, int, bool]:
        """
        Character budget, starting chunk count, and whether the budget is matched.

        With match_context_budget (C-tok control) the budget is whatever the KEO
        arm would have spent on this question, so any remaining difference cannot
        be attributed to one arm simply receiving more context than the other.
        """
        if not self.match_context_budget or self.knowledge_graph is None:
            return default_chars, 5, False
        matched = self.keo_context_chars(question, datasets)
        if matched <= 0:
            return default_chars, 5, False
        top_k = max(5, min(60, math.ceil(matched / preview_chars)))
        return matched, top_k, True

    def _chunk_context_for(
        self,
        question: str,
        datasets: Dict[str, pd.DataFrame],
        default_chars: int,
        max_top_k: int = 80,
        fill_ratio: float = 0.9,
    ) -> Tuple[str, int, int]:
        """
        Chunk context plus the budget it was built against → (context, budget, top_k).

        Under the C-tok control the retrieved text is escalated until it actually
        fills the matched budget; retrieving `budget/preview` chunks undershoots
        badly because previews are truncated and some columns hold no narrative.
        A budget the arm never spends would not control for length at all.
        """
        budget, top_k, matched = self._chunk_budget(question, datasets, default_chars)
        context = self._retrieve_basic_context(question, datasets, budget, top_k=top_k)
        if not matched:
            return context, budget, top_k

        while len(context) < fill_ratio * budget and top_k < max_top_k:
            next_k = min(max_top_k, top_k * 2)
            candidate = self._retrieve_basic_context(question, datasets, budget, top_k=next_k)
            if len(candidate) <= len(context):
                break  # corpus exhausted; no point escalating further
            context, top_k = candidate, next_k
        return context, budget, top_k
    
    def generate_graphrag_answers(self, 
                                questions: List[Dict],
                                datasets: Dict[str, pd.DataFrame],
                                max_path_length: int = 3,
                                top_k_nodes: int = 10,
                                include_community: bool = True,
                                include_graph: bool = True,
                                include_chunks: bool = False,
                                method_label: str = "graphrag",
                                prior_answers: Optional[List[Dict]] = None,
                                on_progress: Optional[Callable[[List[Dict]], None]] = None) -> List[Dict]:
        """
        Generate answers using GraphRAG / KEO approach (and ablation variants).
        
        Args:
            questions: List of question dictionaries
            datasets: Aviation maintenance datasets
            max_path_length: Maximum path length for graph traversal
            top_k_nodes: Number of top relevant nodes to consider
            include_community: Include connected-component community summaries
            include_graph: Include MST/component graph narrative
            include_chunks: Also include text-chunk RAG context (hybrid)
            method_label: Stored in answer records (keo/community/hybrid/...)
            prior_answers: usable answers to keep on resume
            on_progress: optional checkpoint callback
        
        Returns:
            List of answers with graph-based context
        """
        # Fail loudly: returning [] here writes an arm whose every answer field is null,
        # with no error marker anywhere, which reads downstream as a completed run.
        # `is None` rather than truthiness — nx.Graph falls back to __len__, so a 0-node
        # graph is otherwise indistinguishable from an unloaded one.
        if self.knowledge_graph is None or self.graph_retriever is None:
            raise RuntimeError(
                f"{method_label}: knowledge graph not loaded "
                f"(graph={self.knowledge_graph is not None}, "
                f"retriever={self.graph_retriever is not None}). "
                "Load it before generating graph answers."
            )
        if self.knowledge_graph.number_of_nodes() == 0:
            raise RuntimeError(
                f"{method_label}: knowledge graph loaded but empty (0 nodes); "
                "check the --kg-path target."
            )
        
        print(
            f"Generating {method_label} answers "
            f"(mst={self.use_mst}, weight={self.edge_weight}, "
            f"graph={include_graph}, community={include_community}, chunks={include_chunks})..."
        )

        def produce(question_data: Dict) -> Dict:
            question = question_data['question']
            question_type = question_data.get('type', '')

            if self.edge_weight in ("query", "rarity"):
                self._apply_edge_weight_scheme(self.edge_weight, question=question)

            self._last_evidence = {}
            graph_context = (
                self._retrieve_graph_context(question, max_path_length, top_k_nodes)
                if include_graph else ""
            )
            community_context = (
                self._get_community_summaries(question) if include_community else ""
            )
            dataset_context = self._retrieve_basic_context(
                question, datasets, 2000, top_k=3
            )
            if include_chunks:
                chunk_context = self._retrieve_textchunk_context(
                    question, datasets, max_chars=2000, top_k=3
                )
                dataset_context = f"{dataset_context}\n\nAdditional retrieved chunks:\n{chunk_context}"

            if question_type == 'actionable':
                prompt = f"""
You are an expert aviation maintenance technician. Answer the following question with a direct, concise action based on your own knowledge and the provided aviation maintenance records.

Question: {question}

Knowledge Graph-Based Context (which shows related importance entities in Aviation Maintenance Records and their relationships):
{graph_context}

Knowledge Graph-Based Community Insights:
{community_context}

Relevant Aviation Maintenance Records:
{dataset_context}

Provide a brief answer that:
1. States the exact maintenance action to take
2. Is clear and actionable
3. Directly gives the action to take, not the reasoning or background information
4. Uses information from the Aviation Maintenance Records if they are relevant to the question, but not solely based on them, also use your own knowledge

One example:
Q: What action could be taken when: #2 & 4 rocker covers are leaking?
A: Remove & replace #4 gaskets.

Answer:
"""
            else:
                prompt = f"""
You are an expert aviation safety analyst. Answer the following question by using your own knowledge and the provided aviation maintenance knowledge graph and records.

Question: {question}

Knowledge Graph-Based Context (which shows related importance entities in Aviation Maintenance Records and their relationships):
{graph_context}

Knowledge Graph-Based Community Insights:
{community_context}

Relevant Aviation Maintenance Records:
{dataset_context}

Provide a concise and simple answer that:
1. Directly addresses the question
2. Uses specific information from the knowledge graph and records if they are relevant, but do not rely solely on them, also use your own knowledge, and the answer should be high-level and analytical
3. Provides actionable insights, but not specific analysis or reasoning on individual records
4. Maintains focus on aviation safety and maintenance

One example:
Q: To what extent does the inspection and maintenance of aircraft wiring systems influence avionics reliability and flight safety?
A: Regular inspections help identify potential issues such as wear, corrosion, or insulation damage that could lead to electrical failures. Proper maintenance practices, including timely repairs and replacements, significantly reduce the risk of in-flight malfunctions, thereby enhancing overall safety.

Answer:
"""

            answer = self._create_chat_completion(
                messages=[
                    {"role": "system", "content": "You are an expert aviation safety analyst with deep knowledge of maintenance practices, failure analysis, and safety recommendations."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=250 if question_type == 'actionable' else 600,
                temperature=0.3
            )

            return {
                'question_id': question_data.get('id', ''),
                'question': question,
                'category': question_data.get('category', ''),
                'answer': answer,
                'method': method_label,
                'graph_nodes_used': len(graph_context.split('\n')) if graph_context else 0,
                'graph_available': True,
                'evidence': self._last_evidence if self.emit_evidence else None,
                'use_mst': self.use_mst,
                'edge_weight': self.edge_weight,
                'embedding_backend': self.embedding_backend_name,
                'controls': self.control_config(),
                'selection': self.selection_config(),
                **self._model_fields(),
            }

        return self._iter_generate(
            questions,
            desc=f"Generating {method_label} answers",
            produce=produce,
            method_label=method_label,
            prior_answers=prior_answers,
            on_progress=on_progress,
        )

    def generate_chunksum_answers(
        self,
        questions: List[Dict],
        datasets: Dict[str, pd.DataFrame],
        context_window_size: int = 4000,
        method_label: str = "chunk_sum",
        prior_answers: Optional[List[Dict]] = None,
        on_progress: Optional[Callable[[List[Dict]], None]] = None,
    ) -> List[Dict]:
        """
        C-sum control: chunk retrieval plus an LLM cross-record summary, no graph.

        The KEO arm supplies two things the chunk arm lacks: aggregation across
        records, and relational structure. This arm supplies the aggregation alone.
        If it matches KEO, the honest finding is that cross-record aggregation —
        not graph structure — drives the sensemaking gain. Prompts mirror the KEO
        prompts so only the middle context block differs.
        """
        print(f"Generating {method_label} answers (chunks + cross-record summary, no graph)...")

        def produce(question_data: Dict) -> Dict:
            question = question_data['question']
            question_type = question_data.get('type', '')

            dataset_context, budget, top_k = self._chunk_context_for(
                question, datasets, context_window_size
            )

            summary_prompt = f"""
You are analyzing a set of aviation maintenance records retrieved for a question.

Question: {question}

Retrieved Aviation Maintenance Records:
{dataset_context}

Summarize the patterns that hold ACROSS these records, not within any single one:
1. Entities (components, systems, aircraft) that recur across multiple records
2. Failure modes or causes that repeat, and how often
3. Relationships between components, causes, and actions that the records imply
4. Anything rare but safety-critical

Do not answer the question. Report only the cross-record patterns.

Cross-record patterns:
"""
            cross_record_summary = self._create_chat_completion(
                messages=[
                    {"role": "system", "content": "You aggregate patterns across maintenance records without answering the user's question."},
                    {"role": "user", "content": summary_prompt}
                ],
                max_tokens=400,
                temperature=0.3
            )

            if question_type == 'actionable':
                prompt = f"""
You are an expert aviation maintenance technician. Answer the following question with a direct, concise action based on your own knowledge and the provided aviation maintenance records.

Question: {question}

Cross-Record Pattern Summary (aggregated from the retrieved records):
{cross_record_summary}

Relevant Aviation Maintenance Records:
{dataset_context}

Provide a brief answer that:
1. States the exact maintenance action to take
2. Is clear and actionable
3. Directly gives the action to take, not the reasoning or background information
4. Uses information from the Aviation Maintenance Records if they are relevant to the question, but not solely based on them, also use your own knowledge

One example:
Q: What action could be taken when: #2 & 4 rocker covers are leaking?
A: Remove & replace #4 gaskets.

Answer:
"""
            else:
                prompt = f"""
You are an expert aviation safety analyst. Answer the following question by using your own knowledge and the provided aviation maintenance records.

Question: {question}

Cross-Record Pattern Summary (aggregated from the retrieved records):
{cross_record_summary}

Relevant Aviation Maintenance Records:
{dataset_context}

Provide a comprehensive and analytical answer that:
1. Directly addresses the question
2. Uses specific information from the Aviation Maintenance records if they are relevant, but do not rely solely on them, the answer should be high-level and analytical, and you should also use your own knowledge
3. Provides actionable insights, but not specific analysis or reasoning on the records
4. Maintains focus on aviation safety and maintenance

One example:
Q: To what extent does the inspection and maintenance of aircraft wiring systems influence avionics reliability and flight safety?
A: Regular inspections help identify potential issues such as wear, corrosion, or insulation damage that could lead to electrical failures. Proper maintenance practices, including timely repairs and replacements, significantly reduce the risk of in-flight malfunctions, thereby enhancing overall safety.

Answer:
"""

            answer = self._create_chat_completion(
                messages=[
                    {"role": "system", "content": "You are an expert aviation safety analyst specializing in systemic pattern recognition in maintenance data."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=250 if question_type == 'actionable' else 600,
                temperature=0.3
            )

            return {
                'question_id': question_data.get('id', ''),
                'question': question,
                'category': question_data.get('category', ''),
                'answer': answer,
                'method': method_label,
                'cross_record_summary': cross_record_summary,
                'context_used': len(dataset_context),
                'context_budget': budget,
                'chunks_retrieved': top_k,
                'controls': self.control_config(),
                **self._model_fields(),
            }

        return self._iter_generate(
            questions,
            desc=f"Generating {method_label} answers",
            produce=produce,
            method_label=method_label,
            prior_answers=prior_answers,
            on_progress=on_progress,
        )

    def _retrieve_textchunk_context(
        self,
        question: str,
        datasets: Dict[str, pd.DataFrame],
        max_chars: int = 2000,
        top_k: int = 3,
    ) -> str:
        """Lightweight chunk retrieval for hybrid ablation (reuses text-chunk path when possible)."""
        try:
            # Prefer the full textchunk path if chunks already cached via generate_textchunkrag_answers helpers
            chunks = []
            for name, df in datasets.items():
                for col in df.columns:
                    if df[col].dtype == object:
                        for val in df[col].dropna().astype(str).head(50):
                            chunks.append(f"[{name}] {val}")
            if not chunks:
                return ""
            q_emb = np.array(self.embedder.embed(question), dtype=float)
            scored = []
            for chunk in chunks[:200]:
                c_emb = self._get_embedding_cached(chunk)
                if c_emb is None:
                    continue
                sim = self._calculate_cosine_similarity(q_emb.tolist(), c_emb)
                scored.append((sim, chunk))
            scored.sort(key=lambda x: x[0], reverse=True)
            selected = [(s, c) for s, c in scored[:top_k]]
            chunk_ids = []
            for sim, c in selected:
                cid = hashlib.md5(c.encode()).hexdigest()[:12]
                chunk_ids.append({"id": cid, "sim": float(sim), "preview": c[:120]})
            # merge into evidence if present
            ev = dict(self._last_evidence or {})
            ev["chunks"] = chunk_ids
            if chunk_ids:
                ev["confidence"] = float(np.mean([c["sim"] for c in chunk_ids] + [ev.get("confidence", 0.0)]))
            self._last_evidence = ev
            text = "\n".join(f"- [{c['id']}] {c['preview']}" for c in chunk_ids)
            return text[:max_chars]
        except Exception as e:
            return f"(chunk retrieval unavailable: {e})"

    def generate_routed_answers(
        self,
        questions: List[Dict],
        datasets: Dict[str, pd.DataFrame],
        prefer_policy: Optional[str] = None,
        prior_answers: Optional[List[Dict]] = None,
        on_progress: Optional[Callable[[List[Dict]], None]] = None,
    ) -> List[Dict]:
        """Route each question to keo / textchunk / hybrid and generate accordingly."""
        print("Generating task-routed answers...")

        def _first_or_error(batch: List[Dict], method: str, q: Dict) -> Dict:
            if batch:
                return batch[0]
            return {
                "question_id": q.get("id", ""),
                "question": q.get("question", ""),
                "answer": "",
                "method": method,
                "error": True,
            }

        def produce(q: Dict) -> Dict:
            self._last_evidence = {}
            policy = route_question(q, prefer=prefer_policy)  # type: ignore
            if policy == "textchunk":
                batch = self.generate_textchunkrag_answers([q], datasets)
                ans = _first_or_error(batch, "routed_textchunk", q)
                ans["method"] = "routed_textchunk"
            elif policy == "hybrid":
                batch = self.generate_graphrag_answers(
                    [q], datasets,
                    include_graph=True, include_community=True, include_chunks=True,
                    method_label="routed_hybrid",
                )
                ans = _first_or_error(batch, "routed_hybrid", q)
            else:
                batch = self.generate_graphrag_answers(
                    [q], datasets,
                    include_graph=True, include_community=True, include_chunks=False,
                    method_label="routed_keo",
                )
                ans = _first_or_error(batch, "routed_keo", q)
            ans["route_policy"] = policy
            ans["route_reason"] = explain_route(q, policy)
            if self.emit_evidence and ans.get("evidence") is None and self._last_evidence:
                if policy != "textchunk":
                    ans["evidence"] = self._last_evidence
            return ans

        return self._iter_generate(
            questions,
            desc="Routed answers",
            produce=produce,
            method_label="routed",
            prior_answers=prior_answers,
            on_progress=on_progress,
        )

    def generate_comparative_analysis(self,
                                    vanilla_answers: List[Dict],
                                    graphrag_answers: List[Dict]) -> List[Dict]:
        """
        Generate comparative analysis between vanilla and GraphRAG answers
        
        Args:
            vanilla_answers: Answers from vanilla LLM
            graphrag_answers: Answers from GraphRAG
        
        Returns:
            List of comparative analyses
        """
        print("Generating comparative analysis...")
        comparisons = []
        
        # Match answers by question_id
        vanilla_dict = {ans['question_id']: ans for ans in vanilla_answers if not ans.get('error')}
        graphrag_dict = {ans['question_id']: ans for ans in graphrag_answers if not ans.get('error')}
        
        common_ids = set(vanilla_dict.keys()) & set(graphrag_dict.keys())
        
        for question_id in tqdm(common_ids, desc="Comparing answers"):
            try:
                vanilla_ans = vanilla_dict[question_id]
                graphrag_ans = graphrag_dict[question_id]
                
                prompt = f"""
Compare and analyze these two answers to the same aviation maintenance question:

Question: {vanilla_ans['question']}

Vanilla LLM Answer:
{vanilla_ans['answer']}

GraphRAG Answer:
{graphrag_ans['answer']}

Provide a detailed comparison analysis covering:

1. Comprehensiveness: Which answer is more complete?
2. Specificity: Which provides more specific, actionable insights?
3. Evidence Use: How well does each use supporting evidence?
4. Systemic Understanding: Which better captures complex relationships?
5. Practical Value: Which is more useful for aviation safety professionals?

Rate each answer on these dimensions (1-5 scale) and provide an overall assessment.

Analysis:
"""
                
                analysis = self._create_chat_completion(
                    messages=[
                        {"role": "system", "content": "You are an expert evaluator of aviation safety analysis, skilled in assessing the quality and utility of analytical responses."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=400,
                    temperature=0.2
                )
                
                comparisons.append({
                    'question_id': question_id,
                    'question': vanilla_ans['question'],
                    'category': vanilla_ans.get('category', ''),
                    'vanilla_answer': vanilla_ans['answer'],
                    'graphrag_answer': graphrag_ans['answer'],
                    'comparative_analysis': analysis,
                    **self._model_fields(),
                })
                
                # Rate limiting
                #time.sleep(0.5)
                
            except Exception as e:
                print(f"Error generating comparison for question {question_id}: {e}")
        
        return comparisons
    
    def _retrieve_basic_context(self, 
                              question: str, 
                              datasets: Dict[str, pd.DataFrame], 
                              max_chars: int,
                              top_k: int = 3,
                              chunk_size: int = 500) -> str:
        """Retrieve basic context relevant to the question using text chunk-based RAG
        
        Args:
            question: The question to find relevant context for
            datasets: Dictionary of datasets to search through
            max_chars: Maximum characters to return
            top_k: Number of most similar chunks to retrieve per dataset
            chunk_size: Size of text chunks for embedding
        """
        context_parts = []
        
        try:
            # Generate question embedding (with caching)
            question_embedding = self._get_embedding_cached(question)
            if question_embedding is None:
                raise Exception("Could not generate question embedding")
            
            # Check if we have cached chunks for this dataset combination
            dataset_cache_key = self._get_dataset_cache_key(datasets, chunk_size)
            
            if dataset_cache_key in self.chunk_cache:
                cached_data = self.chunk_cache[dataset_cache_key]
            else:
                print("→ Generating new text chunks and embeddings...")
                cached_data = {}
                
                # Text chunking and embedding for each dataset
                for dataset_name, df in datasets.items():
                    chunks = self._create_text_chunks(df, chunk_size=chunk_size)
                    
                    if not chunks:
                        continue
                    
                    # Generate embeddings for chunks (with caching)
                    chunk_embeddings = []
                    for chunk in tqdm(chunks, desc=f"Processing {dataset_name} chunks"):
                        embedding = self._get_embedding_cached(chunk)
                        chunk_embeddings.append(embedding)
                    
                    # Store in cache
                    cached_data[dataset_name] = {
                        'chunks': chunks,
                        'embeddings': chunk_embeddings
                    }
                
                # Save the dataset cache
                self.chunk_cache[dataset_cache_key] = cached_data
                self._save_cache()
            
            # Text chunking and retrieval for each dataset
            for dataset_name, dataset_data in cached_data.items():
                chunks = dataset_data['chunks']
                chunk_embeddings = dataset_data['embeddings']
                
                # Calculate similarities and find most relevant chunks
                chunk_similarities = []
                for i, chunk_embedding in enumerate(chunk_embeddings):
                    if chunk_embedding is not None:
                        similarity = self._calculate_cosine_similarity(
                            question_embedding, chunk_embedding
                        )
                        chunk_similarities.append((i, similarity, chunks[i]))
                
                # Sort by similarity and select top chunks
                chunk_similarities.sort(key=lambda x: x[1], reverse=True)
                if self.dedup_chunks:
                    chunk_similarities = self._dedup_scored_chunks(chunk_similarities)
                top_chunks = chunk_similarities[:top_k]  # Top k most relevant chunks
                
                if top_chunks:
                    context_parts.append(f"\nRelevant {dataset_name} context:")
                    for _, similarity, chunk in top_chunks:
                        context_parts.append(f"- {chunk[:200]}...")
                        
        except Exception as e:
            print(f"Error in text chunk retrieval: {e}")
            # Fallback to simple text extraction
            for dataset_name, df in datasets.items():
                text_data = []
                for col in df.columns:
                    if df[col].dtype == 'object':
                        text_data.extend(df[col].dropna().astype(str).head(3).tolist())
                
                if text_data:
                    context_parts.append(f"\nRelevant {dataset_name} records:")
                    for record in text_data[:3]:
                        context_parts.append(f"- {record[:150]}...")
        
        context = "\n".join(context_parts)
        return context[:max_chars] if len(context) > max_chars else context
    
    def _dedup_scored_chunks(
        self,
        scored: List[Tuple[int, float, str]],
        jaccard_threshold: float = 0.8,
    ) -> List[Tuple[int, float, str]]:
        """
        C-dedup control: greedily drop near-duplicate chunks (token-set Jaccard).

        Maintenance corpora repeat boilerplate heavily, so a plausible explanation
        for the graph advantage is redundancy removal rather than structure. This
        control removes the redundancy from the chunk arm directly.
        """
        kept: List[Tuple[int, float, str]] = []
        kept_tokens: List[set] = []
        for idx, sim, chunk in scored:
            tokens = set(str(chunk).lower().split())
            if not tokens:
                continue
            duplicate = False
            for prior in kept_tokens:
                union = len(tokens | prior)
                if union and len(tokens & prior) / union >= jaccard_threshold:
                    duplicate = True
                    break
            if duplicate:
                continue
            kept.append((idx, sim, chunk))
            kept_tokens.append(tokens)
        return kept

    def keo_context_chars(
        self,
        question: str,
        datasets: Dict[str, pd.DataFrame],
        max_path_length: int = 3,
        top_k_nodes: int = 10,
    ) -> int:
        """
        Characters of context the KEO arm would consume for this question.

        Used by the C-tok control to give chunk retrieval a matched budget. Costs
        no completion tokens — only (cached) embeddings and local graph work.
        """
        saved_evidence = self._last_evidence
        try:
            graph_context = self._retrieve_graph_context(question, max_path_length, top_k_nodes)
            community_context = self._get_community_summaries(question)
            dataset_context = self._retrieve_basic_context(question, datasets, 2000, top_k=3)
            return len(graph_context) + len(community_context) + len(dataset_context)
        except Exception as e:
            print(f"Warning: could not compute matched budget ({e}); using default")
            return 0
        finally:
            self._last_evidence = saved_evidence

    def _retrieve_graph_context(self, 
                              question: str, 
                              max_path_length: int, 
                              top_k_nodes: int) -> str:
        """
        Retrieve context using weighted graph structure with maximum spanning tree approach
        Also populates self._last_evidence with triple IDs and confidence.
        """
        # Returning a placeholder would splice "Graph retriever not available" into the
        # prompt, and the answer would still be judged as a graph arm — a control that
        # silently has no graph context is worse than a missing one.
        if self.graph_retriever is None:
            raise RuntimeError(
                "Graph retriever not available; cannot retrieve graph context."
            )

        try:
            question_embedding = self.embedder.embed(question)
            
            # Find most relevant nodes
            node_similarities = []
            for node in self.knowledge_graph.nodes():
                if node in self.graph_retriever.embeddings:
                    node_embedding = self.graph_retriever.embeddings[node]
                    similarity = self.graph_retriever.calculate_semantic_similarity(
                        question_embedding, node_embedding
                    )
                    node_similarities.append((node, similarity))
            
            # Get top-k most relevant nodes
            node_similarities.sort(key=lambda x: x[1], reverse=True)
            top_nodes = [node for node, _ in node_similarities[:top_k_nodes]]
            seed_sims = {n: s for n, s in node_similarities[:top_k_nodes]}
            
            # Step 1: Create subtrees containing these top-k nodes
            subtree_nodes = set(top_nodes)
            
            # Step 2: Expand with m-hop neighbors
            for hop in range(max_path_length):
                new_nodes = set()
                for node in subtree_nodes:
                    if node in self.knowledge_graph:
                        neighbors = list(self.knowledge_graph.neighbors(node))
                        new_nodes.update(neighbors)
                subtree_nodes.update(new_nodes)
            
            # Step 3: Extract subgraph with all relevant nodes
            relevant_subgraph = self.knowledge_graph.subgraph(subtree_nodes)
            
            # Step 4: Find connected components in the subgraph
            if relevant_subgraph.is_directed():
                undirected_subgraph = relevant_subgraph.to_undirected()
            else:
                undirected_subgraph = relevant_subgraph

            # Step 4b: choose which edges survive. The risk selector spends one global
            # token budget across the whole retrieved subgraph, so components compete
            # with each other; mst/subgraph keep every component and prune within it.
            risk_selection = None
            if self.selector == "risk":
                structure_graph, risk_selection = self._select_risk_budgeted(
                    undirected_subgraph, dict(node_similarities)
                )
            else:
                structure_graph = undirected_subgraph

            connected_components = list(nx.connected_components(structure_graph))

            context_narratives = []
            evidence_triples = []
            
            for i, component in enumerate(connected_components):
                if len(component) < 2:
                    continue
                    
                component_subgraph = relevant_subgraph.subgraph(component)
                component_undirected = structure_graph.subgraph(component)
                
                if self.selector == "risk":
                    # already selected under budget; nothing further to prune
                    structure = component_undirected
                elif self.use_mst:
                    structure = self._find_maximum_spanning_tree(component_undirected)
                else:
                    structure = component_undirected
                
                if structure and len(structure.edges()) > 0:
                    narrative = self._generate_narrative_from_mst(structure, component_subgraph)
                    if narrative:
                        # Same surface form as the MST arm on purpose: only the edge
                        # selection differs, so a judge cannot reward the format.
                        label = (
                            "Knowledge Subgraph"
                            if self.selector == "subgraph" and not self.use_mst
                            else "Knowledge Path"
                        )
                        context_narratives.append(f"{label} {i+1}:\n{narrative}")
                    for u, v, data in structure.edges(data=True):
                        if self.strip_relations:
                            rel = "related_to"
                        else:
                            rel = (
                                data.get("relation") or data.get("relationship")
                                or data.get("relations") or data.get("label")
                                or data.get("type") or "related_to"
                            )
                        w = float(data.get("weight", 1.0))
                        tid = hashlib.md5(f"{u}|{rel}|{v}".encode()).hexdigest()[:12]
                        evidence_triples.append({
                            "id": tid,
                            "head": str(u),
                            "relation": str(rel),
                            "tail": str(v),
                            "weight": w,
                        })
            
            conf = float(np.mean(list(seed_sims.values()))) if seed_sims else 0.0
            self._last_evidence = {
                "seed_nodes": [{"id": n, "sim": float(seed_sims.get(n, 0.0))} for n in top_nodes],
                "triples": evidence_triples[:50],
                "confidence": conf,
                "use_mst": self.use_mst,
                "edge_weight": self.edge_weight,
                "embedding_backend": self.embedding_backend_name,
                "controls": self.control_config(),
                "selection": self.selection_config(),
            }
            if risk_selection is not None:
                self._last_evidence["risk_selection"] = risk_selection

            if context_narratives:
                return "Graph-based Knowledge Context:\n\n" + "\n\n".join(context_narratives)
            else:
                return "No significant graph patterns found for this question."
                
        except Exception as e:
            print(f"Error retrieving graph context: {e}")
            self._last_evidence = {"error": str(e), "triples": [], "confidence": 0.0}
            return "Error retrieving graph context"
    
    def _select_risk_budgeted(
        self,
        undirected_subgraph: nx.Graph,
        node_similarities: Dict[str, float],
    ) -> Tuple[nx.Graph, Dict[str, Any]]:
        """
        Risk-weighted budgeted coverage selection (papers/ACL_STRATEGY.md §3 Pivot C).

        Maximises \\(\\sum_{c} \\max_{e \\ni c} \\mathrm{rel}(e,q)\\cdot\\mathrm{sev}(e)\\)
        over covered entities subject to a token budget, where rel(e,q) is the best
        question-similarity of the edge's two endpoints. Monotone submodular under a
        knapsack constraint, so the cost-benefit greedy carries the usual guarantee.

        Unlike MST this is query-dependent and budget-aware: it cannot be replaced by a
        fixed structural prune, and it is the reason the arm can beat MST on safety
        recall at equal context length.
        """
        records = []
        for u, v, data in undirected_subgraph.edges(data=True):
            if self.strip_relations:
                relation = "related_to"
            else:
                relation = (
                    data.get("relation") or data.get("relationship")
                    or data.get("relations") or data.get("label")
                    or data.get("type") or "related_to"
                )
            relevance = max(
                float(node_similarities.get(u, 0.0)),
                float(node_similarities.get(v, 0.0)),
            )
            records.append({
                "head": str(u),
                "tail": str(v),
                "relation": str(relation),
                "rel": max(relevance, 0.0),
                "severity": edge_severity(str(u), str(relation), str(v)),
                "tokens": edge_tokens(str(u), str(relation), str(v)),
                "weight": float(data.get("weight", 1.0)),
                "node_u": u,
                "node_v": v,
            })

        result = greedy_budgeted_coverage(
            records,
            budget_tokens=self.risk_budget_tokens,
            relevance=lambda rec: rec["rel"],
            use_severity=True,
            risk_floor=self.risk_floor,
            fill_budget=self.risk_fill_budget,
        )

        selected = nx.Graph()
        for rec in result["selected"]:
            u, v = rec["node_u"], rec["node_v"]
            attrs = dict(undirected_subgraph.get_edge_data(u, v) or {})
            selected.add_edge(u, v, **attrs)

        metrics = safety_metrics(result, records)
        stats = {
            "candidate_edges": len(records),
            "selected_edges": len(result["selected"]),
            "tokens_used": result["tokens_used"],
            "budget_tokens": self.risk_budget_tokens,
            "reserved_tokens_used": result["reserved_tokens_used"],
            "risk_floor": self.risk_floor,
            "filler_edges": result["filler_edges"],
            "fill_budget": self.risk_fill_budget,
            "entities_covered": result["entities_covered"],
            "high_severity_recall": metrics["high_severity_recall"],
            "high_severity_total": metrics["high_severity_total"],
            "mean_severity_of_selection": metrics["mean_severity_of_selection"],
        }
        self._risk_selection_stats.append(stats)
        return selected, stats

    def risk_selection_summary(self) -> Dict[str, Any]:
        """
        Aggregate risk-selector behaviour across a run.

        Reported alongside judged quality so the Pareto claim (judge-optimal is not
        safety-optimal) can be read off one file.
        """
        stats = self._risk_selection_stats
        if not stats:
            return {}
        recalls = [s["high_severity_recall"] for s in stats if s["high_severity_recall"] is not None]
        return {
            "questions": len(stats),
            "mean_selected_edges": sum(s["selected_edges"] for s in stats) / len(stats),
            "mean_tokens_used": sum(s["tokens_used"] for s in stats) / len(stats),
            "budget_tokens": self.risk_budget_tokens,
            "risk_floor": self.risk_floor,
            "fill_budget": self.risk_fill_budget,
            "mean_filler_edges": sum(s["filler_edges"] for s in stats) / len(stats),
            "mean_entities_covered": sum(s["entities_covered"] for s in stats) / len(stats),
            "mean_high_severity_recall": (sum(recalls) / len(recalls)) if recalls else None,
            "questions_with_high_severity_candidates": len(recalls),
        }

    def _find_maximum_spanning_tree(self, graph: nx.Graph) -> nx.Graph:
        """
        Find Maximum Spanning Tree using edge weights
        
        Args:
            graph: Undirected graph with edge weights
            
        Returns:
            Maximum spanning tree as a graph
        """
        try:
            # Ensure we have a connected graph
            if not nx.is_connected(graph):
                # Get the largest connected component
                largest_cc = max(nx.connected_components(graph), key=len)
                graph = graph.subgraph(largest_cc)
            
            # Check if edges have weights
            has_weights = any('weight' in data for _, _, data in graph.edges(data=True))
            
            if has_weights:
                # For maximum spanning tree, we need to negate weights for minimum spanning tree algorithm
                # or use maximum_spanning_tree if available
                try:
                    mst = nx.maximum_spanning_tree(graph, weight='weight')
                except:
                    # Fallback: negate weights and use minimum spanning tree
                    temp_graph = graph.copy()
                    for u, v, data in temp_graph.edges(data=True):
                        if 'weight' in data:
                            data['weight'] = -data['weight']
                    mst = nx.minimum_spanning_tree(temp_graph, weight='weight')
                    # Restore original weights
                    for u, v, data in mst.edges(data=True):
                        if 'weight' in data:
                            data['weight'] = -data['weight']
            else:
                # If no weights, treat all edges as weight 1
                for u, v in graph.edges():
                    graph[u][v]['weight'] = 1.0
                mst = nx.maximum_spanning_tree(graph, weight='weight')
            
            return mst
            
        except Exception as e:
            print(f"Error finding MST: {e}")
            return nx.Graph()
    
    def _generate_narrative_from_mst(self, mst: nx.Graph, original_graph: nx.Graph) -> str:
        """
        Generate textual narrative from MST using DFS traversal
        
        Args:
            mst: Maximum spanning tree
            original_graph: Original graph with all edge attributes
            
        Returns:
            Textual narrative describing the knowledge path
        """
        try:
            if len(mst.edges()) == 0:
                return ""
            
            # Find the edge with maximum weight to start DFS
            max_weight = -float('inf')
            start_edge = None
            
            for u, v, data in mst.edges(data=True):
                weight = data.get('weight', 0)
                if weight > max_weight:
                    max_weight = weight
                    start_edge = (u, v)
            
            if not start_edge:
                return ""
            
            # Start DFS from one node of the maximum weight edge
            start_node = start_edge[0]
            visited = set()
            narrative_parts = []
            
            def dfs_narrative(node, parent=None):
                if node in visited:
                    return
                
                visited.add(node)
                
                # Add node information
                node_attrs = original_graph.nodes.get(node, {})
                node_desc = self._format_node_description(node, node_attrs)
                
                if parent is not None:
                    # Add edge information
                    edge_data = original_graph.get_edge_data(parent, node) or original_graph.get_edge_data(node, parent)
                    relationship = self._format_relationship_description(parent, node, edge_data)
                    narrative_parts.append(f"{relationship} {node_desc}")
                else:
                    narrative_parts.append(f"{node_desc}")
                
                # Visit neighbors in order of edge weight (descending)
                neighbors = []
                for neighbor in mst.neighbors(node):
                    if neighbor not in visited:
                        edge_data = mst.get_edge_data(node, neighbor)
                        weight = edge_data.get('weight', 0) if edge_data else 0
                        neighbors.append((neighbor, weight))
                
                # Sort by weight (descending) for priority traversal
                neighbors.sort(key=lambda x: x[1], reverse=True)
                
                for neighbor, _ in neighbors:
                    dfs_narrative(neighbor, node)
            
            # Start DFS traversal
            dfs_narrative(start_node)
            
            # Join narrative parts into coherent text with natural flow
            if narrative_parts:
                # Create a more natural narrative flow
                narrative_text = narrative_parts[0]  # Start with first element
                for i in range(1, len(narrative_parts)):
                    narrative_text += f", which {narrative_parts[i]}"
                return narrative_text
            else:
                return ""
                
        except Exception as e:
            print(f"Error generating narrative: {e}")
            return ""
    
    def _format_node_description(self, node: str, attributes: Dict) -> str:
        """Format node information for narrative as simple text"""
        # Filter out technical IDs and unwanted attributes including all incident-related fields
        excluded_keys = {'incident_id', 'qid', 'id', 'node_id', 'type', 'incident_ids', 'incident ids'}
        
        if attributes:
            # Start with the node name (cleaned)
            clean_node = node.replace('_', ' ').replace('-', ' ').title()
            
            # Extract meaningful attributes, excluding technical IDs and incident references
            meaningful_attrs = []
            for key, value in attributes.items():
                key_lower = str(key).lower()
                if (key_lower not in excluded_keys and 
                    value and str(value).strip() and 
                    not key_lower.endswith('_id') and
                    not key_lower.endswith('_ids') and
                    not key_lower.startswith('id_') and
                    not key_lower.startswith('incident') and
                    'incident' not in key_lower):
                    
                    # Clean up the attribute value
                    clean_value = str(value).strip()
                    if len(clean_value) > 50:
                        clean_value = clean_value[:50] + "..."
                    
                    # Format as natural text
                    clean_key = key.replace('_', ' ').replace('-', ' ').title()
                    meaningful_attrs.append(f"{clean_key}: {clean_value}")
            
            # Return only the cleaned node name without any attributes to avoid incident IDs
            return clean_node
        else:
            # Just return cleaned node name
            return node.replace('_', ' ').replace('-', ' ').title()
    
    def _format_relationship_description(self, source: str, target: str, edge_data: Dict) -> str:
        """Format relationship information for narrative as simple text"""
        if self.strip_relations:
            # G-norel control: keep connectivity, remove relation semantics
            return "is connected to"
        if edge_data:
            relationship = (
                edge_data.get("relationship")
                or edge_data.get("relation")
                or edge_data.get("relations")
                or edge_data.get("label")
                or edge_data.get("type")
                or "relates to"
            )
            # Clean up the relationship text
            clean_relationship = relationship.replace('_', ' ').replace('-', ' ').lower()
            
            # # Make it more natural language
            # if clean_relationship in ['relates to', 'connected to', 'links to']:
            #     return "connects to"
            # elif clean_relationship in ['causes', 'leads to', 'results in']:
            #     return "causes"
            # elif clean_relationship in ['involves', 'includes', 'contains']:
            #     return "involves"
            # elif clean_relationship in ['affects', 'impacts', 'influences']:
            #     return "affects"
            # else:
            #     return clean_relationship
            return clean_relationship
        else:
            return "connects to"
    
    def _get_community_summaries(self, question: str) -> str:
        """Get community-based summaries using DFS traversal from highest weight edges"""
        # Same reasoning as _retrieve_graph_context: a placeholder here would be judged
        # as community context. `is None` because nx.Graph truthiness is node count.
        if self.knowledge_graph is None:
            raise RuntimeError(
                "No graph available for community analysis; load the KG first."
            )
        
        try:
            # Convert directed graph to undirected for community detection
            if self.knowledge_graph.is_directed():
                # Convert to undirected graph for community detection
                undirected_graph = self.knowledge_graph.to_undirected()
                communities = list(nx.connected_components(undirected_graph))
            else:
                # Use original graph if already undirected
                communities = list(nx.connected_components(self.knowledge_graph))
            
            context_parts = []
            context_parts.append("Community-based Knowledge Summary:")
            
            # Analyze largest communities
            largest_communities = sorted(communities, key=len, reverse=True)[:3]
            
            for i, community in enumerate(largest_communities):
                context_parts.append(f"\nCommunity {i+1} ({len(community)} nodes):")
                
                # Get subgraph for this community
                community_subgraph = self.knowledge_graph.subgraph(community)
                
                # Generate DFS-based summary for this community
                community_summary = self._generate_community_dfs_summary(community_subgraph)
                if community_summary:
                    context_parts.append(community_summary)
                else:
                    # Fallback to simple node listing
                    sample_nodes = list(community)[:5]
                    context_parts.append(f"Key entities: {', '.join(sample_nodes)}")
            
            return "\n".join(context_parts)
            
        except Exception as e:
            print(f"Error getting community summaries: {e}")
            return "Error analyzing communities"
    
    def _generate_community_dfs_summary(self, subgraph: nx.Graph) -> str:
        """Generate a DFS-based summary of a community, finding the best path with highest-weight edges"""
        try:
            if not subgraph.nodes() or not subgraph.edges():
                # No edges, just list nodes
                if subgraph.nodes():
                    nodes = list(subgraph.nodes())[:5]
                    clean_nodes = [node.replace('_', ' ').replace('-', ' ').title() for node in nodes]
                    return f"Isolated entities: {', '.join(clean_nodes)}"
                return ""
            
            # Strategy: Try multiple starting points and find the path that includes the most high-weight entities
            best_path = []
            best_score = -1
            
            # Get all edges sorted by weight (descending)
            all_edges = [(u, v, data.get('weight', 0)) for u, v, data in subgraph.edges(data=True)]
            all_edges.sort(key=lambda x: x[2], reverse=True)
            
            # Try starting from nodes of the top 3 highest-weight edges
            start_candidates = set()
            for u, v, weight in all_edges[:min(6, len(all_edges))]:  # Top 3 edges = up to 6 nodes
                start_candidates.add(u)
                start_candidates.add(v)
            
            for start_node in list(start_candidates)[:3]:  # Limit to 3 attempts for performance
                path = self._find_best_path_from_node(subgraph, start_node, max_nodes=5)
                if path:
                    # Score the path based on edge weights and diversity
                    score = self._score_path(subgraph, path)
                    if score > best_score:
                        best_score = score
                        best_path = path
            
            # If no good path found, fall back to simple highest-weight edge approach
            if not best_path:
                if all_edges:
                    start_edge = (all_edges[0][0], all_edges[0][1])
                    best_path = self._find_best_path_from_node(subgraph, start_edge[0], max_nodes=5)
                else:
                    # No edges with weights, just pick any path
                    start_node = list(subgraph.nodes())[0]
                    best_path = self._find_best_path_from_node(subgraph, start_node, max_nodes=5)
            
            # Create narrative from the best path
            return self._create_narrative_from_path(subgraph, best_path)
                
        except Exception as e:
            print(f"Error generating community DFS summary: {e}")
            return ""
    
    def _find_best_path_from_node(self, subgraph: nx.Graph, start_node: str, max_nodes: int = 5) -> List[str]:
        """Find the best path starting from a given node using greedy selection of highest-weight edges"""
        if start_node not in subgraph.nodes():
            return []
        
        path = [start_node]
        current_node = start_node
        visited = {start_node}
        
        while len(path) < max_nodes:
            # Get all unvisited neighbors with their edge weights
            neighbors = []
            for neighbor in subgraph.neighbors(current_node):
                if neighbor not in visited:
                    edge_data = subgraph.get_edge_data(current_node, neighbor)
                    weight = edge_data.get('weight', 0) if edge_data else 0
                    neighbors.append((neighbor, weight))
            
            if not neighbors:
                break
            
            # Choose the neighbor with the highest edge weight
            neighbors.sort(key=lambda x: x[1], reverse=True)
            next_node = neighbors[0][0]
            
            path.append(next_node)
            visited.add(next_node)
            current_node = next_node
        
        return path
    
    def _score_path(self, subgraph: nx.Graph, path: List[str]) -> float:
        """Score a path based on edge weights and entity diversity"""
        if len(path) < 2:
            return 0
        
        total_weight = 0
        for i in range(len(path) - 1):
            edge_data = subgraph.get_edge_data(path[i], path[i+1])
            weight = edge_data.get('weight', 0) if edge_data else 0
            total_weight += weight
        
        # Bonus for longer paths (more entities)
        length_bonus = len(path) * 0.1
        
        return total_weight + length_bonus
    
    def _create_narrative_from_path(self, subgraph: nx.Graph, path: List[str]) -> str:
        """Create a natural language narrative from a path of nodes"""
        if not path:
            return ""
        
        narrative_parts = []
        
        # Start with the first node
        first_node_attrs = subgraph.nodes.get(path[0], {})
        first_desc = self._format_node_description(path[0], first_node_attrs)
        narrative_parts.append(first_desc)
        
        # Add subsequent nodes with their relationships
        for i in range(1, len(path)):
            current_node = path[i]
            previous_node = path[i-1]
            
            # Get edge information
            edge_data = subgraph.get_edge_data(previous_node, current_node) or subgraph.get_edge_data(current_node, previous_node)
            relationship = self._format_relationship_description(previous_node, current_node, edge_data)
            
            # Format current node
            node_attrs = subgraph.nodes.get(current_node, {})
            node_desc = self._format_node_description(current_node, node_attrs)
            
            narrative_parts.append(f"{relationship} {node_desc}")
        
        # Create natural language narrative
        if len(narrative_parts) == 1:
            return f"Key path: {narrative_parts[0]}"
        else:
            narrative_text = narrative_parts[0]  # Start with first element
            for j in range(1, len(narrative_parts)):
                narrative_text += f", which {narrative_parts[j]}"
            return f"Key path: {narrative_text}"

    def save_answers(self, answers: List[Dict], output_path: str, format: str = 'json') -> None:
        """Save answers to file"""
        try:
            if format.lower() == 'json':
                with open(output_path, 'w') as f:
                    json.dump(answers, f, indent=2)
            elif format.lower() == 'csv':
                df = pd.DataFrame(answers)
                df.to_csv(output_path, index=False)
            else:
                raise ValueError("Format must be 'json' or 'csv'")
            
            print(f"Answers saved to {output_path}")
            
        except Exception as e:
            print(f"Error saving answers: {e}")
    
    def _create_text_chunks(self, df: pd.DataFrame, chunk_size: int = 500) -> List[str]:
        """Create text chunks from a narrative column (c119, narrative, text, …)."""
        chunks = []
        col = getattr(self, "text_column", None)
        if col and col in df.columns:
            text_col = col
        else:
            for candidate in ("c119", "narrative", "text", "content", "description"):
                if candidate in df.columns:
                    text_col = candidate
                    break
            else:
                print(f"Warning: No text column found. Available: {list(df.columns)}")
                return chunks

        for _, row in df.iterrows():
            if pd.notna(row[text_col]) and str(row[text_col]).strip():
                text_content = str(row[text_col]).strip()
                
                # Split into chunks if text is too long
                if len(text_content) <= chunk_size:
                    chunks.append(text_content)
                else:
                    # Split into smaller chunks
                    words = text_content.split()
                    current_chunk = []
                    current_length = 0
                    
                    for word in words:
                        if current_length + len(word) + 1 <= chunk_size:
                            current_chunk.append(word)
                            current_length += len(word) + 1
                        else:
                            if current_chunk:
                                chunks.append(" ".join(current_chunk))
                            current_chunk = [word]
                            current_length = len(word)
                    
                    if current_chunk:
                        chunks.append(" ".join(current_chunk))
        
        return chunks
    
    def _calculate_cosine_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Calculate cosine similarity between two embeddings"""
        try:
            # Convert to numpy arrays
            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)
            
            # Calculate cosine similarity
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            return dot_product / (norm1 * norm2)
            
        except Exception as e:
            print(f"Error calculating cosine similarity: {e}")
            return 0.0
    
    def _load_cache(self):
        """Load embedding cache from disk"""
        cache_file = os.path.join(self.cache_dir, "embedding_cache.pkl")
        chunk_cache_file = os.path.join(self.cache_dir, "chunk_cache.pkl")
        
        try:
            if os.path.exists(cache_file):
                with open(cache_file, 'rb') as f:
                    self.embedding_cache = pickle.load(f)
                print(f"✓ Loaded {len(self.embedding_cache)} cached embeddings")
            
            if os.path.exists(chunk_cache_file):
                with open(chunk_cache_file, 'rb') as f:
                    self.chunk_cache = pickle.load(f)
                print(f"✓ Loaded {len(self.chunk_cache)} cached chunk sets")
                
        except Exception as e:
            print(f"Warning: Could not load cache: {e}")
            self.embedding_cache = {}
            self.chunk_cache = {}
    
    def _save_cache(self):
        """Save embedding cache to disk"""
        try:
            cache_file = os.path.join(self.cache_dir, "embedding_cache.pkl")
            chunk_cache_file = os.path.join(self.cache_dir, "chunk_cache.pkl")
            
            with open(cache_file, 'wb') as f:
                pickle.dump(self.embedding_cache, f)
            
            with open(chunk_cache_file, 'wb') as f:
                pickle.dump(self.chunk_cache, f)
                
            print(f"✓ Saved cache with {len(self.embedding_cache)} embeddings and {len(self.chunk_cache)} chunk sets")
            
        except Exception as e:
            print(f"Warning: Could not save cache: {e}")
    
    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text"""
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def _get_dataset_cache_key(self, datasets: Dict[str, pd.DataFrame], chunk_size: int) -> str:
        """Generate cache key for dataset chunks (includes embedder identity)."""
        emb_model = getattr(self.embedder, "model", None) or ""
        dataset_info = [
            f"emb={self.embedding_backend_name}:{emb_model}",
            f"chunk_size={chunk_size}",
        ]
        for name, df in datasets.items():
            dataset_signature = f"{name}_{df.shape}"
            if len(df) > 0:
                sample_text = str(df.head(3).to_dict())[:200]
                dataset_signature += f"_{hashlib.md5(sample_text.encode()).hexdigest()[:8]}"
            dataset_info.append(dataset_signature)

        combined_signature = "_".join(sorted(dataset_info))
        return hashlib.md5(combined_signature.encode("utf-8")).hexdigest()
    
    def _get_embedding_cached(self, text: str) -> Optional[List[float]]:
        """Get embedding from cache or generate and cache it"""
        cache_key = self._get_cache_key(f"{self.embedding_backend_name}:{text}")
        
        if cache_key in self.embedding_cache:
            return self.embedding_cache[cache_key]
        
        try:
            embedding = self.embedder.embed(text)
            self.embedding_cache[cache_key] = embedding
            return embedding
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return None
    
    def cleanup(self):
        """Clean up and save cache before shutting down"""
        self._save_cache()
        print("✓ Cache saved and cleanup completed")


if __name__ == "__main__":
    # Example usage
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    
    # Initialize answer generator
    generator = SensemakingAnswerGenerator(openai_api_key)
    
    # Load knowledge graph if available
    graph_path = "../kg/output/knowledge_graph.gml"
    if os.path.exists(graph_path):
        generator.load_knowledge_graph(graph_path)
    
    # Load questions
    try:
        with open("aviation_sensemaking_questions.json", 'r') as f:
            questions = json.load(f)
        print(f"Loaded {len(questions)} questions")
    except FileNotFoundError:
        print("Questions file not found. Please run question_generator.py first.")
        questions = []
    
    # Load datasets (excluding files used for KG construction)
    data_paths = {
        'maintenance_remaining': "../../OMIn_dataset/data/FAA_data/sampled_for_kg/Maintenance_remaining_for_questions.csv",
        'aircraft_annotation': "../../OMIn_dataset/data/MaintNet_data/Aircraft_Annotation_DataFile.csv"
    }
    
    datasets = {}
    for name, path in data_paths.items():
        try:
            datasets[name] = pd.read_csv(path)
            print(f"Loaded {name}: {len(datasets[name])} records")
        except FileNotFoundError:
            print(f"Dataset {name} not found at {path}")
    
    if questions and datasets:
        # Sample questions for testing (use first 5)
        sample_questions = questions[:5]
        
        # Generate vanilla answers
        vanilla_answers = generator.generate_vanilla_answers(sample_questions)
        generator.save_answers(vanilla_answers, "vanilla_answers.json")
        
        # Generate GraphRAG answers if graph is available
        if generator.knowledge_graph:
            graphrag_answers = generator.generate_graphrag_answers(sample_questions, datasets)
            generator.save_answers(graphrag_answers, "graphrag_answers.json")
            
            # Generate comparative analysis
            comparisons = generator.generate_comparative_analysis(vanilla_answers, graphrag_answers)
            generator.save_answers(comparisons, "answer_comparisons.json")
        
        print("Answer generation complete!")
