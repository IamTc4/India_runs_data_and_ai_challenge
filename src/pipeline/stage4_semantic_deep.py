"""
stage4_semantic_deep.py — Deep representation matching using BGE-Large + FAISS.

Model: BAAI/bge-large-en-v1.5  (~1.3GB, 1024-dim, top MTEB leaderboard)
Method: High-dimensional cosine similarity via FAISS IndexFlatIP.
Catches candidates who described ML work without exact keyword matches.
"""

import numpy as np
from src.utils.logger import logger


def _build_deep_text(c: dict, instruction: str) -> str:
    """
    Richer text for deep encoding — includes full career descriptions.
    BGE instruction prefix improves asymmetric retrieval quality.
    """
    p = c.get("profile", {})
    parts = [
        p.get("current_title", "") or "",
        p.get("headline", "") or "",
        (p.get("summary", "") or "")[:500],
    ]

    skill_names = [s.get("name", "") for s in c.get("skills", [])[:30]]
    parts.append("Skills: " + ", ".join(filter(None, skill_names)))

    for role in c.get("career_history", [])[:4]:
        parts.append(role.get("title", "") or "")
        parts.append((role.get("description", "") or "")[:350])

    for edu in c.get("education", []):
        parts.append(
            f"{edu.get('degree', '')} in {edu.get('field_of_study', '')} "
            f"from {edu.get('institution', '')}"
        )

    for cert in c.get("certifications", []):
        parts.append(cert.get("name", "") or "")

    full_text = " | ".join(p for p in parts if p)
    return instruction + full_text


def stage4_deep_semantic(
    candidates_with_scores: list[tuple[dict, float]],
    jd_text: str,
    model_name: str,
    cache_dir,
    batch_size: int,
    top_k: int,
    instruction: str,
    jd_instruction: str,
) -> list[tuple[dict, float, float]]:
    """
    Args:
        candidates_with_scores: Output of Stage 3 — (candidate, stage3_score) pairs.
        jd_text: Full job description text.
        model_name: HuggingFace model ID (BAAI/bge-large-en-v1.5).
        cache_dir: Path to pre-downloaded model cache.
        batch_size: Encoding batch size for CPU.
        top_k: Hard cap on output (e.g., 200).
        instruction: Candidate encoding prefix (BGE retrieval instruction).
        jd_instruction: JD encoding prefix (BGE query instruction).

    Returns:
        List of (candidate, stage3_score, stage4_score), sorted by Stage 4 score desc.
    """
    candidates = [c for c, _ in candidates_with_scores]
    stage3_scores = {c.get("candidate_id"): s for c, s in candidates_with_scores}

    if len(candidates) <= top_k:
        logger.info(f"Stage 4 | Deep semantic skipped — {len(candidates):,} <= {top_k:,}")
        return [(c, stage3_scores.get(c.get("candidate_id"), 0.5), 0.5) for c in candidates]

    try:
        from sentence_transformers import SentenceTransformer
        import faiss
    except ImportError as e:
        logger.warning(f"Missing dependency: {e}. Run: pip install sentence-transformers faiss-cpu")
        return [
            (c, stage3_scores.get(c.get("candidate_id"), 0.5), 0.5)
            for c in candidates[:top_k]
        ]

    logger.info(f"Stage 4 | Loading model: {model_name}")
    model = SentenceTransformer(model_name, cache_folder=str(cache_dir))

    logger.info("Stage 4 | Encoding Job Description with BGE instruction...")
    jd_embedding = model.encode(
        [jd_instruction + jd_text],
        convert_to_numpy=True,
        show_progress_bar=False,
        normalize_embeddings=True,
    )[0].astype(np.float32)

    logger.info(f"Stage 4 | Encoding {len(candidates):,} candidates (batch={batch_size})...")
    texts = [_build_deep_text(c, instruction) for c in candidates]

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        show_progress_bar=True,
        normalize_embeddings=True,
    ).astype(np.float32)

    # Build FAISS inner-product index (= cosine similarity for normalized vectors)
    dim = embeddings.shape[1]
    logger.info(f"Stage 4 | Building FAISS index (dim={dim}, n={len(candidates):,})...")
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    logger.info("Stage 4 | Searching FAISS index...")
    jd_vec = jd_embedding.reshape(1, -1)
    similarities, indices = index.search(jd_vec, top_k)

    result = [
        (
            candidates[indices[0][j]],
            stage3_scores.get(candidates[indices[0][j]].get("candidate_id"), 0.5),
            float(similarities[0][j]),
        )
        for j in range(len(indices[0]))
    ]

    logger.info(
        f"Stage 4 | Passed {len(result):,} | "
        f"Top BGE similarity: {result[0][2]:.4f} | Bottom: {result[-1][2]:.4f}"
    )
    return result
