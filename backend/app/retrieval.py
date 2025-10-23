# app/retrieval.py
from __future__ import annotations
from typing import List, Tuple, Dict, Optional
import re
from dataclasses import dataclass
from rank_bm25 import BM25Okapi
from langchain.schema import Document

# ---------------------------
# BM25 index over split docs
# ---------------------------

@dataclass
class _BM25State:
    bm25: Optional[BM25Okapi] = None
    corpus_tokens: List[List[str]] = None
    docs: List[Document] = None

_STATE = _BM25State(bm25=None, corpus_tokens=[], docs=[])

_TOKEN_RE = re.compile(r"[A-Za-z0-9%µ.-]+")

def _tok(text: str) -> List[str]:
    # minimal tokenizer (keeps numbers/units)
    return [t.lower() for t in _TOKEN_RE.findall(text)]

def init_bm25(split_docs: List[Document]) -> None:
    """Call once after you split + embed docs (tx + SOAP)."""
    _STATE.docs = split_docs
    _STATE.corpus_tokens = [_tok(d.page_content) for d in split_docs]
    if _STATE.corpus_tokens:
        _STATE.bm25 = BM25Okapi(_STATE.corpus_tokens)

def bm25_search(query: str, k: int = 6) -> List[Document]:
    if not _STATE.bm25 or not _STATE.docs:
        return []
    scores = _STATE.bm25.get_scores(_tok(query))
    # argsort top-k
    idxs = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return [_STATE.docs[i] for i in idxs if scores[i] > 0]

# ---------------------------
# Hierarchical (section-aware) booster
# ---------------------------

def _section_key(d: Document) -> str:
    # prefer explicit SOAP section else tx
    sec = d.metadata.get("section")
    did = d.metadata.get("doc_id", "tx")
    return f"{did}:{sec or 'NA'}"

def hierarchical_boost(candidates: List[Document], extra_pool: List[Document], per_section: int = 2) -> List[Document]:
    """
    Take initial candidates (e.g., from BM25) and expand with more docs
    from the same sections to preserve local context.
    """
    by_section: Dict[str, List[Document]] = {}
    for d in candidates + extra_pool:
        by_section.setdefault(_section_key(d), [])
        if d not in by_section[_section_key(d)]:
            by_section[_section_key(d)].append(d)

    # order sections by how early they appeared in the seed candidates
    seen_order = []
    for d in candidates:
        sk = _section_key(d)
        if sk not in seen_order:
            seen_order.append(sk)

    out: List[Document] = []
    for sk in seen_order:
        pack = by_section.get(sk, [])
        out.extend(pack[:per_section])  # take a couple from that section
    # de-dup while keeping order
    uniq = []
    seen = set()
    for d in out:
        key = (d.metadata.get("doc_id", ""), d.metadata.get("section"), d.page_content[:120])
        if key not in seen:
            seen.add(key)
            uniq.append(d)
    return uniq

# ---------------------------
# Ensemble: dense + chat + fact + BM25
# ---------------------------

def ensemble_retrieve(
    question: str,
    dense_main_docs: List[Document],
    fact_mem_docs: List[Document],
    chat_docs: List[Document],
    bm25_k: int = 6,
    final_k: int = 8,
) -> List[Document]:
    bm25_docs = bm25_search(question, k=bm25_k)

    # hierarchical context: expand around BM25 hits using nearby-in-section docs
    # (we piggyback on dense_main_docs as 'extra_pool' since they came from the same corpus)
    h_docs = hierarchical_boost(bm25_docs, extra_pool=dense_main_docs, per_section=2)

    # Merge: prefer chat/fact (session) first, then BM25/hier, then dense
    merged = _merge_dedup([*chat_docs, *fact_mem_docs, *h_docs, *dense_main_docs])
    return merged[:final_k]

def _merge_dedup(lists: List[Document]) -> List[Document]:
    out: List[Document] = []
    seen = set()
    for d in lists:
        key = (d.metadata.get("doc_id", ""), d.metadata.get("section"), d.metadata.get("role"), d.page_content[:160])
        if key in seen:
            continue
        seen.add(key)
        out.append(d)
    return out

# ---------------------------
# Answerability check
# ---------------------------

def answerability_check(question: str, docs: List[Document], min_docs: int = 1, min_unique_sources: int = 1) -> Tuple[bool, str]:
    """
    Simple gate: require at least N docs and M unique sources before answering.
    Returns (is_answerable, reason).
    """
    if not docs:
        return False, "no_retrieval"
    sources = {d.metadata.get("doc_id", "doc") for d in docs}
    if len(docs) < min_docs:
        return False, "insufficient_docs"
    if len(sources) < min_unique_sources:
        return False, "insufficient_sources"
    return True, "ok"