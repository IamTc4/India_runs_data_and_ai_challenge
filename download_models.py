"""
download_models.py — Pre-download all ML models before ranking.

Run ONCE before rank.py:
    python download_models.py

This caches models to disk so rank.py runs fully offline.
Challenge constraint: has_network_during_ranking = false
"""

import os
import sys

print("=" * 60)
print("Redrob Challenge — Model Pre-downloader")
print("=" * 60)
print("This script downloads and caches all required models.")
print("After this completes, rank.py will run fully offline.\n")

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("ERROR: sentence-transformers not installed.")
    print("Run: pip install -r requirements.txt")
    sys.exit(1)

MODELS = [
    ("sentence-transformers/all-MiniLM-L6-v2",  "Stage 3 - Light Semantic Model (~80MB)"),
    ("BAAI/bge-large-en-v1.5",                   "Stage 4 - Deep Representation Model (~1.3GB)"),
]

for model_name, description in MODELS:
    print(f"\n[DOWNLOADING] {description}")
    print(f"  Model: {model_name}")
    try:
        model = SentenceTransformer(model_name)
        # Quick test encode
        _ = model.encode(["test sentence"], show_progress_bar=False)
        print(f"  ✅ Cached successfully.")
        del model
    except Exception as e:
        print(f"  ❌ FAILED: {e}")
        sys.exit(1)

print("\n" + "=" * 60)
print("✅ All models cached. You can now run rank.py offline.")
print("=" * 60)
