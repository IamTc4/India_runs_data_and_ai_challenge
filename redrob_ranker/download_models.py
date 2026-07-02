#!/usr/bin/env python3
"""
download_models.py — Pre-download and cache all ML models for offline use.

Run ONCE before ranking. Models cached in ./models/ directory.
Total download size: ~1.5GB
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config.settings import config
from src.utils.logger import logger


def download_all():
    from sentence_transformers import SentenceTransformer, CrossEncoder

    cfg = config.model
    cache = str(cfg.cache_dir)
    cfg.cache_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Downloading Stage 3 model: all-MiniLM-L6-v2 (~80MB)...")
    model_light = SentenceTransformer(cfg.light_model_name, cache_folder=cache)
    logger.success(f"  Saved to: {cache}")

    logger.info("Downloading Stage 4 model: cross-encoder/ms-marco-MiniLM-L-6-v2 (~80MB)...")
    model_cross = CrossEncoder(cfg.cross_encoder_model_name, cache_folder=cache)
    logger.success(f"  Cross-encoder cached in: {cache}")

    # Quick smoke tests
    test_cand = "Senior ML Engineer with 6 years experience. Skills: Python, FAISS, embeddings, semantic search, RAG, transformers, PyTorch."
    test_jd = "Looking for ML engineer with RAG, vector search, embeddings experience."

    logger.info("Running smoke test on Stage 3 model...")
    e1 = model_light.encode([test_cand])
    logger.success(f"  MiniLM embedding shape: {e1.shape}")

    logger.info("Running smoke test on Stage 4 cross-encoder...")
    score = model_cross.predict([(test_jd, test_cand)])
    logger.success(f"  Cross-encoder score: {score[0]:.4f}")

    logger.success("All models downloaded and verified. Ready for offline ranking.")
    logger.info(
        "\nExpected pipeline timing on 100K candidates:\n"
        "  Load:           ~10s\n"
        "  Stage 1 filter:  ~1s\n"
        "  Stage 2 BM25:   ~22s (100K -> 1000)\n"
        "  Stage 3 MiniLM: ~90s (1000 -> 300)\n"
        "  Stage 4 Cross:  ~30s (300 -> 100)\n"
        "  Stage 5 Rules:   ~1s\n"
        "  ---------------------\n"
        "  Total:          ~155s  (~2.5 min) [OK]"
    )


if __name__ == "__main__":
    download_all()
