"""
settings.py — Central configuration for the entire pipeline.
Edit this file to tune the pipeline behavior.
"""

from dataclasses import dataclass, field
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent  # redrob_ranker/


@dataclass
class ModelConfig:
    """ML model paths and settings."""
    # Stage 3 — Light semantic model (~80MB, fast CPU inference)
    light_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    light_batch_size: int = 512

    # Stage 4 — Cross-encoder reranker (~80MB, ~98% accuracy, ~30s on CPU)
    # Replaces BGE-Large+FAISS (18 min) with joint query-document scoring.
    cross_encoder_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    cross_encoder_batch_size: int = 64

    # Legacy Stage 4 — BGE-Large deep model (kept for reference, not used by default)
    deep_model_name: str = "BAAI/bge-large-en-v1.5"
    deep_batch_size: int = 64
    deep_instruction: str = "Represent this candidate profile for job matching: "
    jd_instruction: str = "Represent the job description for candidate retrieval: "

    # Cache directory (pre-downloaded models live here)
    cache_dir: Path = BASE_DIR / "models"


@dataclass
class PipelineConfig:
    """Stage-by-stage caps and thresholds."""
    # Stage 1: Hard filter
    min_experience_years: float = 1.0
    max_inactive_days: int = 730          # 2 years

    # Stage 2: BM25 lexical cap
    # 1000 gives good recall; cross-encoder Stage 4 is fast enough to handle 300
    bm25_top_k: int = 1000

    # Stage 3: MiniLM semantic cap — feeds cross-encoder
    stage3_top_k: int = 200

    # Stage 4: Cross-encoder output cap
    stage4_top_k: int = 100

    # Final output
    final_top_k: int = 100


@dataclass
class ScoringWeights:
    """Stage 5 final scoring formula weights."""
    vector_similarity: float = 0.40    # Blended Stage 3 + Stage 4 similarity
    rule_score: float = 0.35           # Rule-based multi-signal score
    behavioral: float = 0.15          # Activity, response rate, availability
    profile_quality: float = 0.10     # Completeness, verification, recruiter interest

    # Within vector_similarity: how much to weight Stage 4 vs Stage 3
    stage4_vs_stage3: float = 0.70    # 70% Stage4 (BGE-Large), 30% Stage3 (MiniLM)


@dataclass
class AppConfig:
    """Top-level application config."""
    model: ModelConfig = field(default_factory=ModelConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    weights: ScoringWeights = field(default_factory=ScoringWeights)
    log_level: str = "INFO"
    api_host: str = "0.0.0.0"
    api_port: int = 8000


# Singleton config instance
config = AppConfig()
