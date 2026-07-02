# Redrob AI Ranker — Enterprise Candidate Intelligence Platform

## Architecture

```
redrob_ranker/
├── src/
│   ├── pipeline/          # 5-stage cascade pipeline
│   │   ├── stage1_filter.py
│   │   ├── stage2_bm25.py
│   │   ├── stage3_semantic_light.py
│   │   ├── stage4_semantic_deep.py
│   │   └── stage5_rerank.py
│   ├── config/            # JD config, settings, model paths
│   │   ├── jd_config.py
│   │   ├── settings.py
│   │   └── skills_taxonomy.py
│   ├── scoring/           # Rule-based scoring engine
│   │   └── rules.py
│   └── utils/             # Shared utilities
│       ├── io.py
│       └── logger.py
├── api/                   # FastAPI enterprise server
│   ├── main.py
│   └── routes.py
├── models/                # Pre-downloaded model cache
├── rank.py                # CLI entry point
├── download_models.py     # One-time model setup
└── requirements.txt
```

## Pipeline Flow

```
100,000 Resumes + 1 JD
        │
  Stage 1: Hardcoded Filter (Pandas)       100K → ~90K   [<2s]
        │
  Stage 2: BM25 Lexical Pre-rank           ~90K → 5K     [~25s]
        │
  Stage 3: MiniLM Semantic Scoring         5K   → 1K     [~60s]
        │
  Stage 4: BGE-Large + FAISS Deep Match    1K   → 200    [~30s]
        │
  Stage 5: Weighted Re-rank + Reasoning    200  → 100    [<1s]
        │
     submission.csv ✅
```

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download ML models (one-time, ~1.5GB)
python download_models.py

# 3. Run on full dataset
python rank.py --candidates data/candidates.jsonl --out submission.csv

# 4. Run enterprise API server
uvicorn api.main:app --reload --port 8000
```

## Enterprise API

```bash
POST /api/rank
{
  "candidates_path": "data/candidates.jsonl",
  "job_description": "...",
  "top_k": 100
}
```
The ranking pipeline can be run in two different ways depending on your needs. Below are the commands and details for each option:

Option 1: Highly Optimized Fast Winning Ranker (Recommended)
This uses a fast BM25 lexical pre-filter followed by multi-signal rules (title, career history, skill taxonomy, experience fit, location, and behavioral metrics) and honeypot detection.

Execution Time: ~33 seconds on the full 100,000 candidate dataset.
Why use it: Designed to easily pass the challenge's 5-minute runtime limit on CPU, using minimal memory and resources.
powershell
# Run on the full dataset
python rank.py --candidates candidates.jsonl --out team_antigravity.csv
# Run on the sample dataset with debug prints
python rank.py --candidates sample_candidates.json --out test_submission.csv --debug
Option 2: Hybrid Multi-Stage ML Ranker
This utilizes deep semantic matching with MiniLM-L6-v2 and BGE-Large-en-v1.5 models over 5 stages.

Execution Time: ~18 minutes (due to heavy CPU encoding of candidate profile blobs).
Why use it: Provides deep semantic matches that do not rely on exact keyword matches, but note it warns about exceeding the 5-minute limit on CPU.
powershell
# 1. Download and cache the ML models (one-time, ~1.5GB)
python redrob_ranker/download_models.py
# 2. Run the deep ML pipeline on the full dataset
python redrob_ranker/rank.py --candidates candidates.jsonl --out submission.csv
# 3. Run the deep pipeline skipping ML models (fast mode, runs in ~30s)
python redrob_ranker/rank.py --candidates candidates.jsonl --out submission.csv --no-models
Running the API Server
You can also run the enterprise FastAPI server to rank candidates interactively:

powershell
cd redrob_ranker
uvicorn api.main:app --reload --port 8000
Verification
Both pipelines have been run on the full candidates.jsonl dataset and their submissions have been validated using the official verification script (validate_submission.py). Both outputs (team_antigravity.csv and submission.csv) are verified as fully valid and compliant with the challenge rules:

powershell
python validate_submission.py team_antigravity.csv
# Output: Submission is valid.
python validate_submission.py submission.csv
# Output: Submission is valid.