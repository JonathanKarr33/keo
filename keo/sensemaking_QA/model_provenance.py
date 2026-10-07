"""
Exact model / package provenance for reproducible KEO runs.

Records requested IDs plus whatever the provider returns at resolve-time
(OpenAI snapshot IDs, HF revision SHAs, local /v1/models payload).
"""

from __future__ import annotations

import importlib.metadata
import os
import platform
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set


def _pkg_version(name: str) -> Optional[str]:
    try:
        return importlib.metadata.version(name)
    except Exception:
        return None


def package_versions() -> Dict[str, Optional[str]]:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "openai": _pkg_version("openai"),
        "anthropic": _pkg_version("anthropic"),
        "huggingface_hub": _pkg_version("huggingface_hub"),
        "sentence_transformers": _pkg_version("sentence-transformers"),
        "transformers": _pkg_version("transformers"),
        "torch": _pkg_version("torch"),
        "networkx": _pkg_version("networkx"),
    }


def resolve_openai_model(client: Any, model_id: str) -> Dict[str, Any]:
    """Best-effort OpenAI model card (snapshot id when API exposes it)."""
    out: Dict[str, Any] = {
        "requested_id": model_id,
        "resolved_id": model_id,
        "provider": "openai",
    }
    try:
        info = client.models.retrieve(model_id)
        # SDK object or dict
        rid = getattr(info, "id", None) or (info.get("id") if isinstance(info, dict) else None)
        created = getattr(info, "created", None) or (info.get("created") if isinstance(info, dict) else None)
        owned_by = getattr(info, "owned_by", None) or (info.get("owned_by") if isinstance(info, dict) else None)
        if rid:
            out["resolved_id"] = rid
        if created is not None:
            out["created"] = created
            try:
                out["created_iso"] = datetime.fromtimestamp(int(created), tz=timezone.utc).isoformat()
            except Exception:
                pass
        if owned_by:
            out["owned_by"] = owned_by
    except Exception as e:
        out["resolve_error"] = str(e)[:240]
    return out


def resolve_huggingface_model(model_id: str, api_provider: Optional[str] = None) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "requested_id": model_id,
        "resolved_id": model_id,
        "provider": "huggingface",
        "api_provider": api_provider,
    }
    try:
        from huggingface_hub import model_info

        token = os.getenv("HF_TOKEN")
        info = model_info(model_id, token=token)
        sha = getattr(info, "sha", None)
        out["resolved_id"] = getattr(info, "id", None) or model_id
        if sha:
            out["revision_sha"] = sha
            out["resolved_id_with_revision"] = f"{out['resolved_id']}@{sha[:12]}"
        out["pipeline_tag"] = getattr(info, "pipeline_tag", None)
        siblings = getattr(info, "siblings", None) or []
        # lightweight signal that weights exist
        out["n_files"] = len(siblings) if siblings else None
    except Exception as e:
        out["resolve_error"] = str(e)[:240]
    return out


def resolve_local_model(client: Any, model_id: str, base_url: Optional[str] = None) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "requested_id": model_id,
        "resolved_id": model_id,
        "provider": "local",
        "base_url": base_url,
    }
    try:
        listing = client.models.list()
        ids: List[str] = []
        data = getattr(listing, "data", None) or listing
        for m in data:
            mid = getattr(m, "id", None) or (m.get("id") if isinstance(m, dict) else None)
            if mid:
                ids.append(mid)
        out["available_models"] = ids
        # Prefer exact match, else first served model
        if model_id in ids:
            out["resolved_id"] = model_id
        elif ids:
            out["resolved_id"] = ids[0]
            out["note"] = f"requested '{model_id}' not in /v1/models; using '{ids[0]}'"
    except Exception as e:
        out["resolve_error"] = str(e)[:240]
    return out


def resolve_llm(
    provider: str,
    model_id: str,
    client: Any = None,
    api_provider: Optional[str] = None,
    base_url: Optional[str] = None,
) -> Dict[str, Any]:
    provider = (provider or "").lower()
    if provider == "openai" and client is not None:
        return resolve_openai_model(client, model_id)
    if provider == "anthropic":
        from anthropic_client import resolve_anthropic_model

        return resolve_anthropic_model(model_id)
    if provider in ("claude_cli", "claude-code", "claude_code"):
        from claude_cli import resolve_claude_cli_model

        return resolve_claude_cli_model(model_id)
    if provider == "huggingface":
        return resolve_huggingface_model(model_id, api_provider=api_provider)
    if provider == "local":
        return resolve_local_model(client, model_id, base_url=base_url)
    return {
        "requested_id": model_id,
        "resolved_id": model_id,
        "provider": provider,
    }


def embedding_provenance(embedder: Any) -> Dict[str, Any]:
    backend = getattr(embedder, "backend", None)
    model = getattr(embedder, "model", None)
    out: Dict[str, Any] = {
        "backend": backend,
        "requested_id": model,
        "resolved_id": model,
        "dim": getattr(embedder, "dim", None),
    }
    if backend == "openai" and model:
        client = getattr(embedder, "_openai", None)
        if client is not None:
            out.update({k: v for k, v in resolve_openai_model(client, model).items() if k not in ("provider",)})
            out["provider"] = "openai"
    elif backend == "local" and model:
        # Prefer HF revision for ST checkpoints
        hf = resolve_huggingface_model(model)
        if hf.get("revision_sha"):
            out["revision_sha"] = hf["revision_sha"]
            out["resolved_id_with_revision"] = hf.get("resolved_id_with_revision")
        st = getattr(embedder, "_st_model", None)
        if st is not None:
            try:
                out["st_model_name_or_path"] = getattr(st, "model_name_or_path", None) or str(st)
            except Exception:
                pass
    return out


def observe_completion_model(response: Any) -> Optional[str]:
    """Extract the exact model string returned on a chat completion."""
    if response is None:
        return None
    mid = getattr(response, "model", None)
    if mid:
        return str(mid)
    if isinstance(response, dict):
        return response.get("model")
    return None


def build_run_provenance(
    *,
    role: str,
    provider: str,
    model_id: str,
    client: Any = None,
    api_provider: Optional[str] = None,
    base_url: Optional[str] = None,
    embedding: Optional[Dict[str, Any]] = None,
    observed_model_ids: Optional[Set[str]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    llm = resolve_llm(
        provider=provider,
        model_id=model_id,
        client=client,
        api_provider=api_provider,
        base_url=base_url,
    )
    observed = sorted(observed_model_ids) if observed_model_ids else []
    # Prefer API-observed snapshot (e.g. gpt-4o-mini-2024-07-18) when present
    exact = llm.get("resolved_id_with_revision") or llm.get("resolved_id") or model_id
    if observed:
        # OpenAI often returns dated snapshot only on the completion object
        exact = observed[-1]
        llm["observed_ids"] = observed
        llm["exact_id"] = exact
    else:
        llm["exact_id"] = exact

    payload: Dict[str, Any] = {
        "role": role,  # answerer | judge | question_generator
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "llm": llm,
        "packages": package_versions(),
    }
    if embedding is not None:
        payload["embedding"] = embedding
    if extra:
        payload["extra"] = extra
    return payload
