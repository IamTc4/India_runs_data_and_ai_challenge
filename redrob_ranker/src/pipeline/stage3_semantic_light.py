"""
stage3_semantic_light.py — Light semantic scoring using MiniLM-L6-v2.

Model: sentence-transformers/all-MiniLM-L6-v2  (~80MB, fast CPU)
Method: Bi-encoder cosine similarity between JD and candidate embeddings.
Returns top-k by similarity for Stage 4 deep processing.
"""

import os
import numpy as np
from src.utils.logger import logger

# Force fully offline operation — prevents HuggingFace from attempting
# network calls when models are already cached locally.
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")


def _build_semantic_text(c: dict) -> str:
    """
    Compact text representation for semantic encoding.
    Keeps the most semantically rich sections per candidate.
    """
    p = c.get("profile", {})
    parts = [
        p.get("current_title", "") or "",
        p.get("headline", "") or "",
        (p.get("summary", "") or "")[:400],
    ]

    skill_names = [s.get("name", "") for s in c.get("skills", [])[:25]]
    parts.append("Skills: " + ", ".join(filter(None, skill_names)))

    history = c.get("career_history", [])
    if history:
        latest = history[0]
        parts.append(latest.get("title", "") or "")
        parts.append((latest.get("description", "") or "")[:250])

    return " | ".join(p for p in parts if p)


def stage3_light_semantic(
    candidates: list[dict],
    jd_text: str,
    model_name: str,
    cache_dir,
    batch_size: int,
    top_k: int,
) -> list[tuple[dict, float]]:
    """
    Args:
        candidates: Candidates from Stage 2.
        jd_text: Full job description text.
        model_name: HuggingFace model ID (e.g., all-MiniLM-L6-v2).
        cache_dir: Path to pre-downloaded model cache.
        batch_size: Encoding batch size (tune for CPU RAM).
        top_k: Hard cap on output (e.g., 1000).

    Returns:
        List of (candidate, similarity_score) tuples, sorted descending.
    """
    if len(candidates) <= top_k:
        logger.info(f"Stage 3 | Light semantic skipped — {len(candidates):,} <= {top_k:,}")
        return [(c, 0.5) for c in candidates]

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        logger.warning("sentence-transformers not installed. Run: pip install sentence-transformers")
        return [(c, 0.5) for c in candidates[:top_k]]

    logger.info(f"Stage 3 | Loading model: {model_name}")
    model = SentenceTransformer(model_name, cache_folder=str(cache_dir), local_files_only=True)

    logger.info("Stage 3 | Encoding Job Description...")
    jd_embedding = model.encode(
        [jd_text],
        convert_to_numpy=True,
        show_progress_bar=False,
        normalize_embeddings=True,
    )[0]

    logger.info(f"Stage 3 | Encoding {len(candidates):,} candidates (batch={batch_size})...")
    texts = [_build_semantic_text(c) for c in candidates]

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    # Cosine similarity via dot product (vectors are L2-normalized)
    similarities = embeddings @ jd_embedding   # shape: (N,)

    ranked_indices = np.argsort(similarities)[::-1][:top_k]
    result = [(candidates[i], float(similarities[i])) for i in ranked_indices]

    logger.info(
        f"Stage 3 | Passed {len(result):,} | "
        f"Top similarity: {result[0][1]:.4f} | Bottom: {result[-1][1]:.4f}"
    )
    return result
