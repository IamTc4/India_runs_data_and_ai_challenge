"""
stage4_crossencoder.py — Cross-encoder reranking using ms-marco-MiniLM-L-6-v2.

Model: cross-encoder/ms-marco-MiniLM-L-6-v2 (~80MB)
Method: Joint query-document scoring — far more accurate than bi-encoders
        because query and candidate are read TOGETHER by the model.
Speed:  Scores 300 candidate-query pairs in ~30s on CPU.
Why:    Unlike MiniLM/BGE which encode separately, cross-encoders catch
        nuanced matches like "built production vector search at scale" vs
        "researched embedding methods academically."
"""

import math
import os

from src.utils.logger import logger

# Force fully offline operation
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")


def _sigmoid(x: float) -> float:
    """Normalise raw cross-encoder logits → [0, 1]."""
    return 1.0 / (1.0 + math.exp(-x))


def _build_candidate_text(c: dict) -> str:
    """Concise candidate text for cross-encoder passage input."""
    p = c.get("profile", {})
    parts = [
        p.get("current_title", "") or "",
        p.get("headline", "") or "",
        (p.get("summary", "") or "")[:400],
    ]

    skill_names = [s.get("name", "") for s in c.get("skills", [])[:25]]
    parts.append("Skills: " + ", ".join(filter(None, skill_names)))

    for role in c.get("career_history", [])[:3]:
        t = role.get("title", "") or ""
        d = (role.get("description", "") or "")[:300]
        if t:
            parts.append(t)
        if d:
            parts.append(d)

    for edu in c.get("education", [])[:2]:
        deg = edu.get("degree", "") or ""
        field = edu.get("field_of_study", "") or ""
        if deg or field:
            parts.append(f"{deg} {field}".strip())

    return " | ".join(p for p in parts if p)


def stage4_crossencoder(
    candidates_with_scores: list[tuple[dict, float]],
    jd_text: str,
    model_name: str,
    cache_dir,
    batch_size: int,
    top_k: int,
) -> list[tuple[dict, float, float]]:
    """
    Cross-encoder reranking — replaces the BGE-Large + FAISS Stage 4.

    Args:
        candidates_with_scores: Output of Stage 3 — (candidate, stage3_score) pairs.
        jd_text: Full job description text (used as cross-encoder query).
        model_name: HuggingFace cross-encoder model ID.
        cache_dir: Path for model caching.
        batch_size: Scoring batch size (32 is safe on 8GB RAM).
        top_k: Hard cap on output (e.g., 100).

    Returns:
        List of (candidate, stage3_score, cross_encoder_score_normalised),
        sorted by cross-encoder score descending.
    """
    stage3_scores = {c.get("candidate_id"): s for c, s in candidates_with_scores}
    candidates = [c for c, _ in candidates_with_scores]

    # Auto-skip if already at or below the target
    if len(candidates) <= top_k:
        logger.info(
            f"Stage 4 | Cross-encoder skipped — {len(candidates):,} <= {top_k:,}"
        )
        return [
            (c, stage3_scores.get(c.get("candidate_id"), 0.5), 0.5)
            for c in candidates
        ]

    try:
        from sentence_transformers import CrossEncoder
    except ImportError as e:
        logger.warning(
            f"Stage 4 | Missing dependency: {e}. Falling back to Stage 3 top-{top_k}."
        )
        return [
            (c, stage3_scores.get(c.get("candidate_id"), 0.5), 0.5)
            for c in candidates[:top_k]
        ]

    logger.info(f"Stage 4 | Loading cross-encoder: {model_name}")
    model = CrossEncoder(model_name, max_length=512, cache_folder=str(cache_dir), local_files_only=True)

    # Build (query, passage) pairs — truncate to avoid exceeding 512 tokens
    # JD is used as the query; candidate profile is the passage
    query = jd_text[:800]  # ~200 tokens — plenty for the JD summary
    logger.info(
        f"Stage 4 | Scoring {len(candidates):,} (query, candidate) pairs (batch={batch_size})..."
    )
    pairs = [(query, _build_candidate_text(c)) for c in candidates]

    raw_scores = model.predict(pairs, batch_size=batch_size, show_progress_bar=True)

    # Normalise logits → [0, 1] via sigmoid
    scored = [
        (idx, _sigmoid(float(raw_scores[idx])))
        for idx in range(len(candidates))
    ]
    scored.sort(key=lambda x: x[1], reverse=True)
    top = scored[:top_k]

    result = [
        (
            candidates[idx],
            stage3_scores.get(candidates[idx].get("candidate_id"), 0.5),
            norm_score,
        )
        for idx, norm_score in top
    ]

    logger.info(
        f"Stage 4 | Passed {len(result):,} | "
        f"Top cross-encoder score: {result[0][2]:.4f} | "
        f"Bottom: {result[-1][2]:.4f}"
    )
    return result
