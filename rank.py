#!/usr/bin/env python3
"""
rank.py — CLI entry point for the Redrob AI Ranker.

Usage:
  # Full run (all 5 stages with ML models):
  python rank.py --candidates ../data/candidates.jsonl --out submission.csv

  # Fast run (BM25 + rules only, no ML models needed):
  python rank.py --candidates ../data/candidates.jsonl --out submission.csv --no-models

  # Test on sample data:
  python rank.py --candidates ../data/sample_candidates.json --out test_submission.csv --debug
"""

import argparse
import sys
from pathlib import Path

# Add project root to PYTHONPATH so src.* imports work from CLI
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.pipeline.pipeline import run_pipeline
from src.utils.logger import logger


def main():
    parser = argparse.ArgumentParser(
        prog="rank",
        description="Redrob AI Ranker — Multi-stage hybrid candidate retrieval",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python rank.py --candidates ../data/candidates.jsonl
  python rank.py --candidates ../data/sample_candidates.json --out test_sub.csv --debug
  python rank.py --candidates ../data/candidates.jsonl --no-models  # BM25+rules only
        """,
    )
    parser.add_argument(
        "--candidates",
        default="../data/candidates.jsonl",
        help="Path to candidates.jsonl or sample_candidates.json",
    )
    parser.add_argument(
        "--out",
        default="submission.csv",
        help="Output CSV path (default: submission.csv)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print top-10 candidates to console",
    )
    parser.add_argument(
        "--no-models",
        action="store_true",
        help="Skip ML model stages (BM25 + rules only). Faster, no GPU/download needed.",
    )
    args = parser.parse_args()

    success = run_pipeline(
        candidates_path=args.candidates,
        output_path=args.out,
        debug=args.debug,
        skip_stage3=args.no_models,
        skip_stage4=args.no_models,
    )
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
