#!/usr/bin/env python3
"""KEO human-validation server (answer prefs + severity).

  cd keo/sensemaking_QA/human_validation/pref_site
  python3 serve.py
  open http://127.0.0.1:8765/

Annotator 1 password: keo
Annotator 2 password: eok
Adjudicator (3) password: adj

Answer prefs (rate.html / adjudicate.html):
  ../ratings/human_pref_votes_annotator{1,2,3}.json
  ../ratings/human_pref_labels.csv
  ../ratings/human_pref_consensus.json

Severity / risk triples (severity.html) — lexicon prior NOT shown:
  ../ratings/severity_votes_annotator{1,2,3}.json
  ../ratings/severity_labels.csv
  ../ratings/severity_consensus.json
  also mirrors classes into severity_expert_n100.json (ann a/b)
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import re
import secrets
import threading
import time
from difflib import SequenceMatcher
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent
RATINGS = ROOT.parent / "ratings"
BLIND_PATH = RATINGS / "answer_calibration_n40.json"
KEY_PATH = RATINGS / "answer_calibration_n40_key.json"
LABELS_CSV = RATINGS / "human_pref_labels.csv"
CONSENSUS_PATH = RATINGS / "human_pref_consensus.json"
SEVERITY_PACK = RATINGS / "severity_expert_n100.json"
SEVERITY_LABELS_CSV = RATINGS / "severity_labels.csv"
SEVERITY_CONSENSUS = RATINGS / "severity_consensus.json"
SESSIONS: Dict[str, Dict[str, Any]] = {}
LOCK = threading.Lock()

PASSWORDS = {"1": "keo", "2": "eok", "3": "adj"}
PRIMARY = ("1", "2")
SEVERITY_CLASSES = (
    "catastrophic",
    "hazardous",
    "major",
    "minor",
    "none",
)
STOP = set(
    """a an the and or but if then of to in on for with from by as is are was were be been being
    this that these those it its their his her our your you we they them at not no so than into
    over under about after before between through during without within can could should would
    may might will shall do does did done have has had having more most less least also just
    only very when where which who whom what how why such other per via etc""".split()
)

_BLIND: Optional[dict] = None
_KEY_BY_ID: Dict[str, dict] = {}


def load_data() -> None:
    global _BLIND, _KEY_BY_ID
    _BLIND = json.loads(BLIND_PATH.read_text())
    key = json.loads(KEY_PATH.read_text())
    _KEY_BY_ID = {it["item_id"]: it for it in key.get("items") or []}


def votes_path(annotator: str) -> Path:
    return RATINGS / f"human_pref_votes_annotator{annotator}.json"


def load_votes(annotator: str) -> dict:
    path = votes_path(annotator)
    if not path.exists():
        return {
            "annotator": annotator,
            "updated_at": None,
            "n_rated": 0,
            "votes": {},
        }
    return json.loads(path.read_text())


def save_votes(annotator: str, blob: dict) -> None:
    RATINGS.mkdir(parents=True, exist_ok=True)
    blob["annotator"] = annotator
    blob["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    blob["n_rated"] = sum(
        1 for v in (blob.get("votes") or {}).values() if v.get("preference_ab")
    )
    path = votes_path(annotator)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(blob, indent=2))
    tmp.replace(path)
    rewrite_labels_csv()
    write_consensus()


def rewrite_labels_csv() -> None:
    rows = []
    for who in ("1", "2", "3"):
        blob = load_votes(who)
        for item_num, v in sorted(
            (blob.get("votes") or {}).items(), key=lambda kv: int(kv[0])
        ):
            rows.append(
                {
                    "annotator": who,
                    "role": "adjudicator" if who == "3" else "primary",
                    "item_num": item_num,
                    "item_id": v.get("item_id", ""),
                    "preference_ab": v.get("preference_ab", ""),
                    "preference_system": v.get("preference_system", ""),
                    "resolution": v.get("resolution", ""),
                    "note": v.get("note", ""),
                    "a_system": v.get("a_system", ""),
                    "b_system": v.get("b_system", ""),
                    "swapped_from_pack": v.get("swapped_from_pack", ""),
                    "updated_at": v.get("updated_at", ""),
                }
            )
    RATINGS.mkdir(parents=True, exist_ok=True)
    with LABELS_CSV.open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "annotator",
            "role",
            "item_num",
            "item_id",
            "preference_ab",
            "preference_system",
            "resolution",
            "note",
            "a_system",
            "b_system",
            "swapped_from_pack",
            "updated_at",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def system_pref(annotator: str, item_num: str) -> Optional[str]:
    v = (load_votes(annotator).get("votes") or {}).get(str(item_num))
    if not v:
        return None
    return v.get("preference_system") or None


def disagreement_nums() -> List[str]:
    """Items both primaries rated where preference_system differs."""
    v1 = load_votes("1").get("votes") or {}
    v2 = load_votes("2").get("votes") or {}
    out = []
    for num in sorted(set(v1) & set(v2), key=lambda x: int(x)):
        s1 = v1[num].get("preference_system")
        s2 = v2[num].get("preference_system")
        if s1 and s2 and s1 != s2:
            out.append(num)
    return out


def write_consensus() -> None:
    """Final labels: agree → consensus; else adjudicator/conference if present."""
    assert _BLIND is not None
    v1 = load_votes("1").get("votes") or {}
    v2 = load_votes("2").get("votes") or {}
    v3 = load_votes("3").get("votes") or {}
    items = []
    n_agree = n_adj = n_conf = n_open = 0
    for i, raw in enumerate(_BLIND.get("items") or [], 1):
        num = str(i)
        a1 = v1.get(num)
        a2 = v2.get(num)
        row = {
            "item_num": i,
            "item_id": raw["item_id"],
            "ann1_system": (a1 or {}).get("preference_system"),
            "ann2_system": (a2 or {}).get("preference_system"),
            "final_system": None,
            "source": None,
        }
        if a1 and a2:
            s1 = a1.get("preference_system")
            s2 = a2.get("preference_system")
            if s1 and s2 and s1 == s2:
                row["final_system"] = s1
                row["source"] = "consensus"
                n_agree += 1
            else:
                a3 = v3.get(num)
                if a3 and a3.get("preference_system"):
                    row["final_system"] = a3["preference_system"]
                    res = a3.get("resolution") or "adjudicator"
                    row["source"] = res
                    if res == "conference":
                        n_conf += 1
                    else:
                        n_adj += 1
                else:
                    row["source"] = "open_disagreement"
                    n_open += 1
        elif a1 or a2:
            row["source"] = "incomplete_primaries"
        else:
            row["source"] = "unrated"
        items.append(row)

    blob = {
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "n_items": len(items),
        "n_consensus_agree": n_agree,
        "n_adjudicated": n_adj,
        "n_conference": n_conf,
        "n_open_disagreement": n_open,
        "items": items,
    }
    RATINGS.mkdir(parents=True, exist_ok=True)
    tmp = CONSENSUS_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(blob, indent=2))
    tmp.replace(CONSENSUS_PATH)


def severity_votes_path(annotator: str) -> Path:
    return RATINGS / f"severity_votes_annotator{annotator}.json"


def load_severity_votes(annotator: str) -> dict:
    path = severity_votes_path(annotator)
    if not path.exists():
        return {
            "annotator": annotator,
            "task": "severity",
            "updated_at": None,
            "n_rated": 0,
            "votes": {},
        }
    return json.loads(path.read_text())


def load_severity_pack() -> dict:
    return json.loads(SEVERITY_PACK.read_text())


def save_severity_votes(annotator: str, blob: dict) -> None:
    RATINGS.mkdir(parents=True, exist_ok=True)
    blob["annotator"] = annotator
    blob["task"] = "severity"
    blob["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    blob["n_rated"] = sum(
        1 for v in (blob.get("votes") or {}).values() if v.get("severity_class")
    )
    path = severity_votes_path(annotator)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(blob, indent=2))
    tmp.replace(path)
    rewrite_severity_labels_csv()
    write_severity_consensus()
    mirror_severity_into_pack()


def rewrite_severity_labels_csv() -> None:
    rows = []
    for who in ("1", "2", "3"):
        blob = load_severity_votes(who)
        for item_num, v in sorted(
            (blob.get("votes") or {}).items(), key=lambda kv: int(kv[0])
        ):
            rows.append(
                {
                    "annotator": who,
                    "role": "adjudicator" if who == "3" else "primary",
                    "item_num": item_num,
                    "head": v.get("head", ""),
                    "relation": v.get("relation", ""),
                    "tail": v.get("tail", ""),
                    "severity_class": v.get("severity_class", ""),
                    "resolution": v.get("resolution", ""),
                    "note": v.get("note", ""),
                    "updated_at": v.get("updated_at", ""),
                }
            )
    with SEVERITY_LABELS_CSV.open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "annotator",
            "role",
            "item_num",
            "head",
            "relation",
            "tail",
            "severity_class",
            "resolution",
            "note",
            "updated_at",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def severity_disagreement_nums() -> List[str]:
    v1 = load_severity_votes("1").get("votes") or {}
    v2 = load_severity_votes("2").get("votes") or {}
    out = []
    for num in sorted(set(v1) & set(v2), key=lambda x: int(x)):
        c1 = v1[num].get("severity_class")
        c2 = v2[num].get("severity_class")
        if c1 and c2 and c1 != c2:
            out.append(num)
    return out


def write_severity_consensus() -> None:
    pack = load_severity_pack()
    v1 = load_severity_votes("1").get("votes") or {}
    v2 = load_severity_votes("2").get("votes") or {}
    v3 = load_severity_votes("3").get("votes") or {}
    items = []
    n_agree = n_adj = n_conf = n_open = 0
    for raw in pack.get("items") or []:
        num = str(raw["i"])
        a1 = v1.get(num)
        a2 = v2.get(num)
        row = {
            "item_num": raw["i"],
            "head": raw.get("head"),
            "relation": raw.get("relation"),
            "tail": raw.get("tail"),
            "ann1_class": (a1 or {}).get("severity_class"),
            "ann2_class": (a2 or {}).get("severity_class"),
            "final_class": None,
            "source": None,
        }
        if a1 and a2:
            c1 = a1.get("severity_class")
            c2 = a2.get("severity_class")
            if c1 and c2 and c1 == c2:
                row["final_class"] = c1
                row["source"] = "consensus"
                n_agree += 1
            else:
                a3 = v3.get(num)
                if a3 and a3.get("severity_class"):
                    row["final_class"] = a3["severity_class"]
                    res = a3.get("resolution") or "adjudicator"
                    row["source"] = res
                    if res == "conference":
                        n_conf += 1
                    else:
                        n_adj += 1
                else:
                    row["source"] = "open_disagreement"
                    n_open += 1
        elif a1 or a2:
            row["source"] = "incomplete_primaries"
        else:
            row["source"] = "unrated"
        items.append(row)

    blob = {
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "n_items": len(items),
        "n_consensus_agree": n_agree,
        "n_adjudicated": n_adj,
        "n_conference": n_conf,
        "n_open_disagreement": n_open,
        "classes": list(SEVERITY_CLASSES),
        "items": items,
    }
    tmp = SEVERITY_CONSENSUS.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(blob, indent=2))
    tmp.replace(SEVERITY_CONSENSUS)


def mirror_severity_into_pack() -> None:
    """Write primary votes into severity_expert_n100.json without exposing priors."""
    pack = load_severity_pack()
    v1 = load_severity_votes("1").get("votes") or {}
    v2 = load_severity_votes("2").get("votes") or {}
    for raw in pack.get("items") or []:
        num = str(raw["i"])
        raw["annotator_a_class"] = (v1.get(num) or {}).get("severity_class")
        raw["annotator_b_class"] = (v2.get(num) or {}).get("severity_class")
    tmp = SEVERITY_PACK.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(pack, indent=2))
    tmp.replace(SEVERITY_PACK)


def side_random_swap(annotator: str, item_id: str) -> bool:
    digest = hashlib.sha256(f"keo-pref|{annotator}|{item_id}".encode()).hexdigest()
    return int(digest[:8], 16) % 2 == 1


def tokenize_words(text: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)?|[^\sA-Za-z0-9]+|\s+", text)


def word_diff_html(a: str, b: str) -> Tuple[str, str, List[str], List[str]]:
    ta = tokenize_words(a)
    tb = tokenize_words(b)
    sm = SequenceMatcher(None, [t.lower() for t in ta], [t.lower() for t in tb])
    out_a: List[str] = []
    out_b: List[str] = []
    only_a: List[str] = []
    only_b: List[str] = []

    def is_word(t: str) -> bool:
        return bool(re.match(r"^[A-Za-z0-9]", t))

    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        chunk_a = ta[i1:i2]
        chunk_b = tb[j1:j2]
        if tag == "equal":
            out_a.extend(html.escape(t) for t in chunk_a)
            out_b.extend(html.escape(t) for t in chunk_b)
        elif tag == "replace":
            out_a.append('<mark class="chg">')
            out_a.extend(html.escape(t) for t in chunk_a)
            out_a.append("</mark>")
            out_b.append('<mark class="chg">')
            out_b.extend(html.escape(t) for t in chunk_b)
            out_b.append("</mark>")
            only_a.extend(t.lower() for t in chunk_a if is_word(t) and t.lower() not in STOP)
            only_b.extend(t.lower() for t in chunk_b if is_word(t) and t.lower() not in STOP)
        elif tag == "delete":
            out_a.append('<mark class="only-a">')
            out_a.extend(html.escape(t) for t in chunk_a)
            out_a.append("</mark>")
            only_a.extend(t.lower() for t in chunk_a if is_word(t) and t.lower() not in STOP)
        elif tag == "insert":
            out_b.append('<mark class="only-b">')
            out_b.extend(html.escape(t) for t in chunk_b)
            out_b.append("</mark>")
            only_b.extend(t.lower() for t in chunk_b if is_word(t) and t.lower() not in STOP)

    def uniq_keep(words: List[str], k: int = 14) -> List[str]:
        seen = set()
        out = []
        for w in words:
            if w not in seen:
                seen.add(w)
                out.append(w)
            if len(out) >= k:
                break
        return out

    html_a = "".join(out_a).replace("\n", "<br>\n")
    html_b = "".join(out_b).replace("\n", "<br>\n")
    return html_a, html_b, uniq_keep(only_a), uniq_keep(only_b)


def build_item_payload(annotator: str, item_num: int, raw: dict) -> dict:
    item_id = raw["item_id"]
    left = raw.get("answer_left") or ""
    right = raw.get("answer_right") or ""
    key = _KEY_BY_ID.get(item_id) or {}
    left_sys = key.get("left_system", "unknown")
    right_sys = key.get("right_system", "unknown")

    swap = side_random_swap(annotator, item_id)
    if swap:
        a_text, b_text = right, left
        a_sys, b_sys = right_sys, left_sys
    else:
        a_text, b_text = left, right
        a_sys, b_sys = left_sys, right_sys

    html_a, html_b, chips_a, chips_b = word_diff_html(a_text, b_text)
    return {
        "item_num": item_num,
        "item_id": item_id,
        "question": raw.get("question") or "",
        "answer_a_html": html_a,
        "answer_b_html": html_b,
        "chips_a": chips_a,
        "chips_b": chips_b,
        "_meta": {
            "a_system": a_sys,
            "b_system": b_sys,
            "swapped_from_pack": swap,
        },
    }


def map_pref_to_system(pref_ab: str, a_sys: str, b_sys: str) -> str:
    if pref_ab == "A":
        return a_sys
    if pref_ab == "B":
        return b_sys
    if pref_ab == "neither":
        return "tie"
    return ""


def system_to_ab(pref_sys: Optional[str], a_sys: str, b_sys: str) -> str:
    if not pref_sys:
        return "—"
    if pref_sys == "tie":
        return "Neither"
    if pref_sys == a_sys:
        return "Prefer A"
    if pref_sys == b_sys:
        return "Prefer B"
    return "—"


class Handler(BaseHTTPRequestHandler):
    server_version = "KEOPref/4.0"

    def _cors(self) -> None:
        self.send_header("Cache-Control", "no-store")

    def _json(self, code: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self._cors()
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            return json.loads(raw.decode("utf-8") or "{}")
        except json.JSONDecodeError:
            return {}

    def _session(self) -> Optional[dict]:
        auth = self.headers.get("Authorization") or ""
        if auth.startswith("Bearer "):
            token = auth[7:].strip()
            with LOCK:
                return SESSIONS.get(token)
        return None

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(204)
        self._cors()
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/api/me":
            sess = self._session()
            if not sess:
                return self._json(401, {"error": "login required"})
            who = sess["annotator"]
            votes = load_votes(who)
            return self._json(
                200,
                {
                    "annotator": who,
                    "role": "adjudicator" if who == "3" else "primary",
                    "n_rated": votes.get("n_rated", 0),
                    "votes": {
                        k: v.get("preference_ab")
                        for k, v in (votes.get("votes") or {}).items()
                    },
                },
            )

        if path == "/api/items":
            sess = self._session()
            if not sess:
                return self._json(401, {"error": "login required"})
            if sess["annotator"] == "3":
                return self._json(
                    403,
                    {"error": "adjudicator uses /api/disagreements and adjudicate.html"},
                )
            assert _BLIND is not None
            who = sess["annotator"]
            items = []
            meta = {}
            for i, raw in enumerate(_BLIND.get("items") or [], 1):
                payload = build_item_payload(who, i, raw)
                meta[str(i)] = payload.pop("_meta")
                items.append(payload)
            with LOCK:
                sess["item_meta"] = meta
            votes = load_votes(who)
            return self._json(
                200,
                {
                    "annotator": who,
                    "n": len(items),
                    "items": items,
                    "votes": {
                        k: v.get("preference_ab")
                        for k, v in (votes.get("votes") or {}).items()
                    },
                    "note": "A/B sides are randomized per item; systems are not shown.",
                },
            )

        if path == "/api/disagreements":
            sess = self._session()
            if not sess:
                return self._json(401, {"error": "login required"})
            if sess["annotator"] != "3":
                return self._json(403, {"error": "adjudicator login required"})
            assert _BLIND is not None
            who = "3"
            nums = disagreement_nums()
            v1 = load_votes("1").get("votes") or {}
            v2 = load_votes("2").get("votes") or {}
            v3 = load_votes("3").get("votes") or {}
            items = []
            meta = {}
            raws = _BLIND.get("items") or []
            for num in nums:
                idx = int(num) - 1
                raw = raws[idx]
                payload = build_item_payload(who, int(num), raw)
                m = payload.pop("_meta")
                meta[num] = m
                a_sys, b_sys = m["a_system"], m["b_system"]
                s1 = (v1.get(num) or {}).get("preference_system")
                s2 = (v2.get(num) or {}).get("preference_system")
                adj = v3.get(num) or {}
                payload["ann1_label"] = system_to_ab(s1, a_sys, b_sys)
                payload["ann2_label"] = system_to_ab(s2, a_sys, b_sys)
                # stay blind: do not expose system names
                payload["ann1_system"] = None
                payload["ann2_system"] = None
                payload["prior_resolution"] = adj.get("resolution")
                payload["prior_pref"] = adj.get("preference_ab")
                payload["prior_note"] = adj.get("note") or ""
                items.append(payload)
            with LOCK:
                sess["item_meta"] = meta
            both_done = (
                load_votes("1").get("n_rated", 0) >= 40
                and load_votes("2").get("n_rated", 0) >= 40
            )
            return self._json(
                200,
                {
                    "annotator": who,
                    "role": "adjudicator",
                    "n_disagreements": len(items),
                    "primaries_complete": both_done,
                    "n_adjudicated": sum(
                        1 for n in nums if (v3.get(n) or {}).get("preference_ab")
                    ),
                    "items": items,
                    "note": "Only contested items. Labels are remapped to your A/B. Systems stay hidden.",
                },
            )

        if path == "/api/labels":
            rows = []
            if LABELS_CSV.exists():
                with LABELS_CSV.open(newline="", encoding="utf-8") as handle:
                    rows = list(csv.DictReader(handle))
            write_consensus()
            consensus = {}
            if CONSENSUS_PATH.exists():
                consensus = json.loads(CONSENSUS_PATH.read_text())
            summary = {
                "n_rows": len(rows),
                "by_annotator": {},
                "disagreements": len(disagreement_nums()),
                "consensus": {
                    "n_agree": consensus.get("n_consensus_agree", 0),
                    "n_adjudicated": consensus.get("n_adjudicated", 0),
                    "n_conference": consensus.get("n_conference", 0),
                    "n_open": consensus.get("n_open_disagreement", 0),
                },
            }
            for who in ("1", "2", "3"):
                blob = load_votes(who)
                summary["by_annotator"][who] = {
                    "n_rated": blob.get("n_rated", 0),
                    "updated_at": blob.get("updated_at"),
                    "role": "adjudicator" if who == "3" else "primary",
                    "path": str(votes_path(who)),
                }
            return self._json(
                200, {"summary": summary, "rows": rows, "consensus": consensus}
            )

        if path == "/api/severity/items":
            sess = self._session()
            if not sess:
                return self._json(401, {"error": "login required"})
            who = sess["annotator"]
            pack = load_severity_pack()
            votes = load_severity_votes(who)
            items = []
            for raw in pack.get("items") or []:
                num = str(raw["i"])
                items.append(
                    {
                        "item_num": raw["i"],
                        "head": raw.get("head") or "",
                        "relation": raw.get("relation") or "",
                        "tail": raw.get("tail") or "",
                    }
                )
            # Adjudicator only needs contested; still return all for status nav
            contested = severity_disagreement_nums() if who == "3" else []
            return self._json(
                200,
                {
                    "annotator": who,
                    "role": "adjudicator" if who == "3" else "primary",
                    "n": len(items),
                    "classes": list(SEVERITY_CLASSES),
                    "instructions": pack.get("instructions")
                    or "Assign one severity class per triple.",
                    "items": items,
                    "votes": {
                        k: v.get("severity_class")
                        for k, v in (votes.get("votes") or {}).items()
                    },
                    "contested": contested,
                    "n_rated": votes.get("n_rated", 0),
                    "note": "Lexicon prior is hidden. Do not open severity_expert_n100_key.json.",
                },
            )

        if path == "/api/severity/labels":
            rewrite_severity_labels_csv()
            write_severity_consensus()
            rows = []
            if SEVERITY_LABELS_CSV.exists():
                with SEVERITY_LABELS_CSV.open(newline="", encoding="utf-8") as handle:
                    rows = list(csv.DictReader(handle))
            consensus = {}
            if SEVERITY_CONSENSUS.exists():
                consensus = json.loads(SEVERITY_CONSENSUS.read_text())
            contested = severity_disagreement_nums()
            summary = {
                "n_rows": len(rows),
                "n_items": consensus.get("n_items", 100),
                "disagreements": len(contested),
                "by_annotator": {},
                "consensus": {
                    "n_agree": consensus.get("n_consensus_agree", 0),
                    "n_adjudicated": consensus.get("n_adjudicated", 0),
                    "n_conference": consensus.get("n_conference", 0),
                    "n_open": consensus.get("n_open_disagreement", 0),
                },
            }
            for who in ("1", "2", "3"):
                blob = load_severity_votes(who)
                summary["by_annotator"][who] = {
                    "n_rated": blob.get("n_rated", 0),
                    "updated_at": blob.get("updated_at"),
                    "role": "adjudicator" if who == "3" else "primary",
                    "path": str(severity_votes_path(who)),
                }
            return self._json(
                200,
                {
                    "summary": summary,
                    "rows": rows,
                    "consensus": consensus,
                    "contested": contested,
                },
            )

        rel = path.lstrip("/") or "index.html"
        if ".." in rel or rel.startswith("api/"):
            self.send_error(404)
            return
        file_path = (ROOT / rel).resolve()
        if not str(file_path).startswith(str(ROOT)) or not file_path.is_file():
            self.send_error(404)
            return
        data = file_path.read_bytes()
        ctype = {
            ".html": "text/html; charset=utf-8",
            ".js": "text/javascript; charset=utf-8",
            ".css": "text/css; charset=utf-8",
            ".json": "application/json",
            ".svg": "image/svg+xml",
        }.get(file_path.suffix, "application/octet-stream")
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self._cors()
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path
        body = self._read_json()

        if path == "/api/login":
            who = str(body.get("annotator") or "")
            pw = str(body.get("password") or "")
            if PASSWORDS.get(who) != pw:
                return self._json(401, {"error": "wrong password"})
            token = secrets.token_urlsafe(24)
            with LOCK:
                SESSIONS[token] = {"annotator": who, "item_meta": {}, "at": time.time()}
            role = "adjudicator" if who == "3" else "primary"
            return self._json(
                200, {"token": token, "annotator": who, "role": role}
            )

        if path == "/api/vote":
            sess = self._session()
            if not sess:
                return self._json(401, {"error": "login required"})
            if sess["annotator"] == "3":
                return self._json(
                    403, {"error": "use /api/adjudicate for contested items"}
                )
            who = sess["annotator"]
            item_num = str(body.get("item_num") or "")
            pref = str(body.get("preference_ab") or "")
            if pref not in {"A", "B", "neither"}:
                return self._json(400, {"error": "preference_ab must be A, B, or neither"})
            if not item_num.isdigit():
                return self._json(400, {"error": "bad item_num"})

            meta = (sess.get("item_meta") or {}).get(item_num)
            if not meta:
                assert _BLIND is not None
                idx = int(item_num) - 1
                raws = _BLIND.get("items") or []
                if idx < 0 or idx >= len(raws):
                    return self._json(400, {"error": "item out of range"})
                payload = build_item_payload(who, int(item_num), raws[idx])
                meta = payload["_meta"]
                with LOCK:
                    sess.setdefault("item_meta", {})[item_num] = meta

            a_sys = meta["a_system"]
            b_sys = meta["b_system"]
            pref_sys = map_pref_to_system(pref, a_sys, b_sys)
            assert _BLIND is not None
            raw = (_BLIND.get("items") or [])[int(item_num) - 1]

            with LOCK:
                blob = load_votes(who)
                votes = blob.setdefault("votes", {})
                votes[item_num] = {
                    "item_id": raw["item_id"],
                    "preference_ab": pref,
                    "preference_system": pref_sys,
                    "a_system": a_sys,
                    "b_system": b_sys,
                    "swapped_from_pack": bool(meta.get("swapped_from_pack")),
                    "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                }
                save_votes(who, blob)

            return self._json(
                200,
                {
                    "ok": True,
                    "item_num": int(item_num),
                    "preference_ab": pref,
                    "n_rated": load_votes(who).get("n_rated", 0),
                    "saved_to": str(votes_path(who)),
                    "labels_csv": str(LABELS_CSV),
                },
            )

        if path == "/api/adjudicate":
            sess = self._session()
            if not sess:
                return self._json(401, {"error": "login required"})
            if sess["annotator"] != "3":
                return self._json(403, {"error": "adjudicator login required"})
            who = "3"
            item_num = str(body.get("item_num") or "")
            pref = str(body.get("preference_ab") or "")
            resolution = str(body.get("resolution") or "adjudicator")
            note = str(body.get("note") or "")[:500]
            if pref not in {"A", "B", "neither"}:
                return self._json(400, {"error": "preference_ab must be A, B, or neither"})
            if resolution not in {"adjudicator", "conference"}:
                return self._json(
                    400, {"error": "resolution must be adjudicator or conference"}
                )
            if item_num not in disagreement_nums():
                return self._json(
                    400, {"error": "item is not a disagreement (or primaries incomplete)"}
                )

            meta = (sess.get("item_meta") or {}).get(item_num)
            if not meta:
                assert _BLIND is not None
                idx = int(item_num) - 1
                payload = build_item_payload(who, int(item_num), (_BLIND.get("items") or [])[idx])
                meta = payload["_meta"]
                with LOCK:
                    sess.setdefault("item_meta", {})[item_num] = meta

            a_sys = meta["a_system"]
            b_sys = meta["b_system"]
            pref_sys = map_pref_to_system(pref, a_sys, b_sys)
            assert _BLIND is not None
            raw = (_BLIND.get("items") or [])[int(item_num) - 1]

            with LOCK:
                blob = load_votes(who)
                votes = blob.setdefault("votes", {})
                votes[item_num] = {
                    "item_id": raw["item_id"],
                    "preference_ab": pref,
                    "preference_system": pref_sys,
                    "resolution": resolution,
                    "note": note,
                    "a_system": a_sys,
                    "b_system": b_sys,
                    "swapped_from_pack": bool(meta.get("swapped_from_pack")),
                    "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                }
                save_votes(who, blob)

            open_left = len(
                [
                    n
                    for n in disagreement_nums()
                    if not (load_votes("3").get("votes") or {}).get(n, {}).get("preference_ab")
                ]
            )
            return self._json(
                200,
                {
                    "ok": True,
                    "item_num": int(item_num),
                    "preference_ab": pref,
                    "resolution": resolution,
                    "n_open_disagreements": open_left,
                    "saved_to": str(votes_path(who)),
                    "consensus_path": str(CONSENSUS_PATH),
                },
            )

        if path == "/api/severity/vote":
            sess = self._session()
            if not sess:
                return self._json(401, {"error": "login required"})
            who = sess["annotator"]
            item_num = str(body.get("item_num") if body.get("item_num") is not None else "")
            sev = str(body.get("severity_class") or "").strip().lower()
            resolution = str(body.get("resolution") or "")
            note = str(body.get("note") or "")[:500]
            if sev not in SEVERITY_CLASSES:
                return self._json(
                    400,
                    {"error": f"severity_class must be one of {list(SEVERITY_CLASSES)}"},
                )
            if not item_num.isdigit():
                return self._json(400, {"error": "bad item_num"})

            pack = load_severity_pack()
            by_i = {str(it["i"]): it for it in (pack.get("items") or [])}
            raw = by_i.get(item_num)
            if not raw:
                return self._json(400, {"error": "item out of range"})

            if who == "3":
                if item_num not in severity_disagreement_nums():
                    return self._json(
                        400,
                        {
                            "error": "item is not a severity disagreement "
                            "(or primaries incomplete)"
                        },
                    )
                if resolution not in {"adjudicator", "conference"}:
                    resolution = "adjudicator"
            else:
                resolution = ""

            with LOCK:
                blob = load_severity_votes(who)
                votes = blob.setdefault("votes", {})
                row = {
                    "item_num": int(item_num),
                    "head": raw.get("head"),
                    "relation": raw.get("relation"),
                    "tail": raw.get("tail"),
                    "severity_class": sev,
                    "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                }
                if who == "3":
                    row["resolution"] = resolution
                    row["note"] = note
                votes[item_num] = row
                save_severity_votes(who, blob)

            open_left = (
                len(
                    [
                        n
                        for n in severity_disagreement_nums()
                        if not (load_severity_votes("3").get("votes") or {})
                        .get(n, {})
                        .get("severity_class")
                    ]
                )
                if who == "3"
                else None
            )
            return self._json(
                200,
                {
                    "ok": True,
                    "item_num": int(item_num),
                    "severity_class": sev,
                    "n_rated": load_severity_votes(who).get("n_rated", 0),
                    "n_open_disagreements": open_left,
                    "saved_to": str(severity_votes_path(who)),
                    "consensus_path": str(SEVERITY_CONSENSUS),
                },
            )

        if path == "/api/logout":
            auth = self.headers.get("Authorization") or ""
            if auth.startswith("Bearer "):
                token = auth[7:].strip()
                with LOCK:
                    SESSIONS.pop(token, None)
            return self._json(200, {"ok": True})

        self.send_error(404)

    def log_message(self, fmt: str, *args) -> None:
        print("[%s] %s" % (self.log_date_time_string(), fmt % args))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    if not BLIND_PATH.exists() or not KEY_PATH.exists():
        raise SystemExit(f"Missing pack files under {RATINGS}")
    load_data()
    rewrite_labels_csv()
    write_consensus()

    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"KEO preference + severity site: http://{args.host}:{args.port}/")
    print("Annotator 1 / keo  |  Annotator 2 / eok  |  Adjudicator 3 / adj")
    print("Prefs: /rate.html  ·  Severity: /severity.html  ·  Status: /status.html")
    print(f"Severity pack: {SEVERITY_PACK.name} ({len(load_severity_pack().get('items') or [])} triples)")
    print("Ctrl+C to stop.")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")


if __name__ == "__main__":
    main()
