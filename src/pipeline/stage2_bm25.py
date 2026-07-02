"""
stage2_bm25.py — BM25 lexical pre-filter.

Uses BM25Okapi to rank candidates by keyword relevance to the JD.
Hardcoded keyword expansion replaces LLM query expansion for speed.
IDF naturally rewards rare technical terms (PyTorch, FAISS) over generic words.
"""

import re
import numpy as np
from src.utils.logger import logger


def _tokenize(text: str) -> list[str]:
    """Fast lowercasing tokenizer — splits on non-alphanumeric chars."""
    return re.findall(r"[a-z0-9][\w\-']*", text.lower())


def _build_candidate_text(c: dict) -> str:
    """Build a single searchable text blob per candidate."""
    p = c.get("profile", {})
    parts = [
        p.get("current_title", "") or "",
        p.get("headline", "") or "",
        (p.get("summary", "") or "")[:500],
        p.get("current_industry", "") or "",
    ]
    parts += [s.get("name", "") or "" for s in c.get("skills", [])]
    for role in c.get("career_history", [])[:3]:
        parts.append(role.get("title", "") or "")
        parts.append((role.get("description", "") or "")[:300])
    parts += [cert.get("name", "") or "" for cert in c.get("certifications", [])]
    return " ".join(x for x in parts if x)


def stage2_bm25(candidates: list[dict], keywords: list[str], top_k: int) -> list[dict]:
    """
    Args:
        candidates: Candidates from Stage 1.
        keywords: Expanded keyword list from jd_config.BM25_KEYWORDS.
        top_k: Hard cap on output (e.g., 5000).

    Returns:
        Top-k candidates sorted by BM25 relevance score.
    """
    if len(candidates) <= top_k:
        logger.info(f"Stage 2 | BM25 skipped — {len(candidates):,} <= {top_k:,}")
        return candidates

    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        logger.warning("rank_bm25 not installed. Run: pip install rank-bm25. Skipping Stage 2.")
        return candidates

    logger.info(f"Stage 2 | Building BM25 corpus from {len(candidates):,} candidates...")
    tokenized_corpus = [_tokenize(_build_candidate_text(c)) for c in candidates]
    bm25 = BM25Okapi(tokenized_corpus)

    query_tokens = _tokenize(" ".join(keywords))
    scores = bm25.get_scores(query_tokens)

    ranked_indices = np.argsort(scores)[::-1][:top_k]
    result = [candidates[i] for i in ranked_indices]

    logger.info(
        f"Stage 2 | Passed {len(result):,} | Top BM25 score: {scores[ranked_indices[0]]:.2f}"
    )
    return result
