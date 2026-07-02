"""I/O utilities — candidate loading and submission writing."""

import csv
import json
from pathlib import Path
from typing import Iterator

from src.utils.logger import logger


def stream_candidates(path: str) -> Iterator[dict]:
    """Memory-efficient streaming loader for large JSONL files."""
    p = Path(path)
    if p.suffix == ".jsonl":
        logger.info(f"Streaming JSONL: {path}")
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)
    elif p.suffix == ".json":
        logger.info(f"Loading JSON: {path}")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        items = data if isinstance(data, list) else list(data.values())
        yield from items
    else:
        raise ValueError(f"Unsupported format: {p.suffix}. Use .jsonl or .json")


def load_candidates(path: str) -> list[dict]:
    """Load all candidates into memory (use stream_candidates for 100K+)."""
    candidates = list(stream_candidates(path))
    logger.info(f"Loaded {len(candidates):,} candidates.")
    return candidates


def write_submission(rows: list[dict], output_path: str) -> None:
    """Write final submission CSV in challenge format."""
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["candidate_id", "rank", "score", "reasoning"]
        )
        writer.writeheader()
        writer.writerows(rows)
    logger.success(f"Submission written -> {output_path}  ({len(rows)} rows)")


def validate_submission(output_path: str, expected_rows: int = 100) -> bool:
    """Built-in submission validator (mirrors official validate_submission.py logic)."""
    errors = []
    ranks_seen, ids_seen = set(), set()
    prev_score = float("inf")
    row_count = 0

    with open(output_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row_count += 1
            r, s, cid = int(row["rank"]), float(row["score"]), row["candidate_id"]
            if r in ranks_seen:
                errors.append(f"Duplicate rank {r}")
            if cid in ids_seen:
                errors.append(f"Duplicate candidate_id {cid}")
            if s > prev_score + 1e-9:
                errors.append(f"Score not monotonic at rank {r}")
            ranks_seen.add(r)
            ids_seen.add(cid)
            prev_score = s

    if row_count != expected_rows:
        errors.append(f"Expected {expected_rows} rows, got {row_count}")
    if ranks_seen != set(range(1, expected_rows + 1)):
        errors.append(f"Ranks 1-{expected_rows} not all present")

    if errors:
        for e in errors:
            logger.error(f"Validation: {e}")
        return False

    logger.success(f"Validation passed — {row_count} rows, scores non-increasing")
    return True
