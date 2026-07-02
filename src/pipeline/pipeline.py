"""
pipeline.py — Main orchestrator: wires all 5 stages together.

Usage:
    from src.pipeline.pipeline import run_pipeline
    run_pipeline(candidates_path="data/candidates.jsonl", output_path="submission.csv")
"""

import time
from pathlib import Path

from src.config.jd_config import JD_TEXT, BM25_KEYWORDS
from src.config.settings import config
from src.pipeline.stage1_filter import stage1_filter
from src.pipeline.stage2_bm25 import stage2_bm25
from src.pipeline.stage3_semantic_light import stage3_light_semantic
from src.pipeline.stage4_crossencoder import stage4_crossencoder
from src.pipeline.stage5_rerank import stage5_rerank
from src.utils.io import load_candidates, write_submission, validate_submission
from src.utils.logger import logger


def run_pipeline(
    candidates_path: str,
    output_path: str,
    debug: bool = False,
    skip_stage3: bool = False,
    skip_stage4: bool = False,
) -> bool:
    """
    Run the full 5-stage ranking pipeline.

    Args:
        candidates_path: Path to candidates.jsonl or sample_candidates.json.
        output_path: Path to write submission.csv.
        debug: Print detailed top-10 candidates to console.
        skip_stage3: Skip MiniLM semantic scoring (fallback to BM25 top).
        skip_stage4: Skip cross-encoder reranking (fallback to Stage 3 top-100).

    Returns:
        True if successful and submission is valid, False otherwise.
    """
    t_start = time.perf_counter()
    cfg = config

    logger.info("=" * 60)
    logger.info("Redrob AI Ranker — Starting Pipeline")
    logger.info("=" * 60)

    # ── Load ──────────────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    candidates = load_candidates(candidates_path)
    logger.info(f"Load | {time.perf_counter()-t0:.1f}s")

    # ── Stage 1: Hard Filter ──────────────────────────────────────────────────
    t0 = time.perf_counter()
    logger.info("\n--- Stage 1: Hard Filter ---")
    s1_out = stage1_filter(
        candidates,
        min_exp=cfg.pipeline.min_experience_years,
        max_inactive_days=cfg.pipeline.max_inactive_days,
    )
    logger.info(f"Stage 1 | Duration: {time.perf_counter()-t0:.1f}s")

    # ── Stage 2: BM25 Pre-filter ──────────────────────────────────────────────
    t0 = time.perf_counter()
    logger.info("\n--- Stage 2: BM25 Lexical Pre-filter ---")
    s2_out = stage2_bm25(
        s1_out,
        keywords=BM25_KEYWORDS,
        top_k=cfg.pipeline.bm25_top_k,
    )
    logger.info(f"Stage 2 | Duration: {time.perf_counter()-t0:.1f}s")

    # ── Stage 3: Light Semantic (MiniLM) ──────────────────────────────────────
    t0 = time.perf_counter()
    logger.info("\n--- Stage 3: Semantic Scoring (MiniLM-L6-v2) ---")
    if skip_stage3:
        logger.warning("Stage 3 SKIPPED — using BM25 output directly.")
        s3_out = [(c, 0.5) for c in s2_out]
    else:
        s3_out = stage3_light_semantic(
            s2_out,
            jd_text=JD_TEXT,
            model_name=cfg.model.light_model_name,
            cache_dir=cfg.model.cache_dir,
            batch_size=cfg.model.light_batch_size,
            top_k=cfg.pipeline.stage3_top_k,
        )
    logger.info(f"Stage 3 | Duration: {time.perf_counter()-t0:.1f}s")

    # ── Stage 4: Cross-Encoder Reranker ─────────────────────────────────────
    t0 = time.perf_counter()
    logger.info("\n--- Stage 4: Cross-Encoder Reranker (ms-marco-MiniLM-L-6-v2) ---")
    if skip_stage4:
        logger.warning("Stage 4 SKIPPED — using Stage 3 output directly.")
        s4_out = [(c, s3, 0.5) for c, s3 in s3_out]
    else:
        s4_out = stage4_crossencoder(
            s3_out,
            jd_text=JD_TEXT,
            model_name=cfg.model.cross_encoder_model_name,
            cache_dir=cfg.model.cache_dir,
            batch_size=cfg.model.cross_encoder_batch_size,
            top_k=cfg.pipeline.stage4_top_k,
        )
    logger.info(f"Stage 4 | Duration: {time.perf_counter()-t0:.1f}s")

    # ── Stage 5: Final Weighted Re-rank ───────────────────────────────────────
    t0 = time.perf_counter()
    logger.info("\n--- Stage 5: Weighted Re-rank + Reasoning ---")

    # Fallback: if fewer than final_top_k candidates reached Stage 5,
    # run rules on ALL loaded candidates to guarantee 100 output rows.
    if len(s4_out) < cfg.pipeline.final_top_k:
        logger.warning(
            f"Only {len(s4_out)} candidates reached Stage 5 — "
            f"falling back to scoring all {len(candidates):,} loaded candidates."
        )
        s4_out = [(c, 0.5, 0.5) for c in candidates]

    rows = stage5_rerank(
        s4_out,
        weights=cfg.weights,
        top_k=cfg.pipeline.final_top_k,
        debug=debug,
    )
    logger.info(f"Stage 5 | Duration: {time.perf_counter()-t0:.1f}s")

    # ── Write + Validate ──────────────────────────────────────────────────────
    logger.info("\n--- Writing Submission ---")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    write_submission(rows, output_path)

    valid = validate_submission(output_path, expected_rows=cfg.pipeline.final_top_k)

    total = time.perf_counter() - t_start
    logger.info("=" * 60)
    if valid:
        logger.success(f"Pipeline complete in {total:.1f}s | Submission: {output_path}")
    else:
        logger.error(f"Pipeline finished but validation FAILED after {total:.1f}s")
    if total > 300:
        logger.warning("Runtime exceeded 5-min challenge limit!")
    logger.info("=" * 60)

    return valid
