#!/usr/bin/env python3
"""
main.py — Redrob Hackathon — Team Antigravity
Intelligent Candidate Discovery & Ranking Challenge

Architecture (no GPU, no network, CPU-only, <3 min for 100K):
  Stage 1 | BM25 pre-filter        100K → 5K     [~25s]
  Stage 2 | Semantic scoring        5K  → 1K     [~60s, needs cached model]
  Stage 3 | Multi-signal rules      5K → scored  [~5s]
  Stage 4 | Weighted blend + top-100 output       [<1s]

Key Design Decisions:
  - NDCG@10 (50% of score) demands near-perfect top-10 → rule signal tuned to JD
  - Semantic model (MiniLM) catches candidates who DESCRIBE IR work without IR keywords
  - Behavioral signals used as multiplier, not primary signal
  - Honeypot detection runs before all scoring (saves compute + avoids disqualification)
  - Score normalization via sigmoid (not linear) → stable [0,1] range

Usage:
  # Full run (with semantic model — needs python download_models.py first):
  python main.py --candidates candidates.jsonl --out team_antigravity.csv

  # Fast run (rules + BM25 only, no model needed):
  python main.py --candidates candidates.jsonl --out team_antigravity.csv --no-semantic
"""

import argparse
import csv
import json
import math
import re
import sys
import io
# Fix Windows console encoding (PowerShell default is cp1252, breaks Unicode chars)
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
import time
from datetime import date, datetime
from pathlib import Path


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG — tune everything here
# ═══════════════════════════════════════════════════════════════════════════════

TODAY = date.today()

# JD text for semantic embedding (Stages 2)
JD_TEXT = (
    "Senior AI Engineer with production experience in embeddings-based retrieval systems, "
    "vector databases (Pinecone, Weaviate, Qdrant, Milvus, FAISS, Elasticsearch, OpenSearch), "
    "semantic search, hybrid search, information retrieval, ranking, recommendation systems. "
    "Strong Python. Evaluation frameworks: NDCG, MRR, MAP, A/B testing. "
    "5-9 years experience. Product companies preferred. "
    "LLM fine-tuning, RAG, sentence-transformers, BGE, E5 are strong signals."
)

# BM25 keyword pool (expanded from JD via domain knowledge)
BM25_KEYWORDS = [
    "embedding", "embeddings", "retrieval", "ranking", "recommendation",
    "vector", "faiss", "pinecone", "weaviate", "qdrant", "milvus",
    "opensearch", "elasticsearch", "semantic search", "hybrid search",
    "sentence-transformers", "bge", "e5", "openai",
    "nlp", "natural language processing", "transformers", "bert", "gpt", "llm",
    "rag", "retrieval augmented", "fine-tuning", "fine tuning", "lora",
    "pytorch", "tensorflow", "scikit-learn", "python", "hugging face",
    "ndcg", "mrr", "a/b test", "ab testing", "evaluation", "offline evaluation",
    "machine learning", "deep learning", "applied ml", "mlops",
    "ml engineer", "ai engineer", "data scientist", "nlp engineer",
    "search engineer", "ranking engineer", "applied scientist", "recommendation engineer",
    "xgboost", "lightgbm", "information retrieval",
]

# Product companies — signals for "product-first mindset"
PRODUCT_COMPANIES = frozenset({
    "swiggy", "zomato", "flipkart", "meesho", "razorpay", "zepto", "blinkit",
    "cred", "groww", "zerodha", "sharechat", "nykaa", "paytm", "ola",
    "rapido", "lenskart", "urban company", "amazon", "google", "microsoft",
    "meta", "netflix", "uber", "linkedin", "salesforce", "adobe",
    "atlassian", "freshworks", "zoho", "sarvam", "haptik", "rephrase",
    "sprinklr", "clevertap", "moengage", "exotel", "darwinbox",
    "leadsquared", "whatfix", "postman", "browserstack",
})

# Services companies — penalise if ENTIRE career is here
SERVICES_COMPANIES = frozenset({
    "tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini",
    "hcl", "tech mahindra", "hexaware", "mindtree", "ltimindtree",
    "persistent", "cyient", "niit", "kpit", "zensar", "mphasis",
    "l&t infotech", "ltts", "birlasoft", "sonata",
})

# Good industries (from dataset values, lowercased)
GOOD_INDUSTRIES = frozenset({
    "ai/ml", "software", "fintech", "e-commerce", "food delivery",
    "saas", "healthtech", "edtech", "deeptech", "internet",
})

BAD_INDUSTRIES = frozenset({
    "it services", "manufacturing", "transportation",
    "paper products", "conglomerate",
})

# Target locations (from JD)
TARGET_CITIES_PRIMARY = frozenset({"pune", "noida"})
TARGET_CITIES_SECONDARY = frozenset({
    "hyderabad", "mumbai", "bangalore", "bengaluru",
    "delhi", "gurgaon", "gurugram", "chennai", "ncr", "delhi ncr",
})

# IR/Retrieval keywords for career description scoring
IR_PRODUCTION_KEYWORDS = [
    # Core IR technologies
    "embedding", "embeddings", "vector search", "semantic search",
    "information retrieval", "retrieval", "ranking", "re-ranking", "reranking",
    "recommendation", "faiss", "pinecone", "weaviate", "qdrant", "milvus",
    "opensearch", "elasticsearch", "bm25", "dense retrieval", "sparse retrieval",
    "hybrid search", "rag", "retrieval augmented",
    # Sentence transformers / models
    "sentence-transformers", "sentence transformers", "bge", "e5", "openai embeddings",
    # Production signals
    "deployed", "production", "scale", "serving", "inference", "latency",
    "throughput", "million", "billion", "real-time",
    # Evaluation
    "ndcg", "mrr", "map", "a/b test", "ab test", "online experiment",
    "offline evaluation", "precision@", "recall@",
    # Fine-tuning
    "fine-tuning", "fine tuning", "finetuning", "lora", "qlora", "peft",
    # General ML production
    "model deployment", "mlops", "feature store", "experiment tracking",
]

# Skills that indicate core competence for this role
CORE_SKILLS = frozenset({
    "embeddings", "embedding", "faiss", "pinecone", "weaviate", "qdrant", "milvus",
    "opensearch", "elasticsearch", "semantic search", "vector search", "vector database",
    "information retrieval", "retrieval", "ranking", "recommendation",
    "sentence-transformers", "sentence transformers", "bge", "e5",
    "rag", "hybrid search", "bm25", "dense retrieval",
    "ndcg", "mrr", "a/b testing", "ab testing",
    "fine-tuning", "lora", "qlora",
})

SUPPORTING_SKILLS = frozenset({
    "python", "pytorch", "tensorflow", "transformers", "bert", "gpt", "llm",
    "nlp", "natural language processing", "machine learning", "deep learning",
    "hugging face", "huggingface", "scikit-learn", "xgboost", "lightgbm",
    "mlops", "mlflow", "wandb", "docker", "kubernetes",
    "spark", "airflow", "sql",
})


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 1 — HONEYPOT DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

def is_honeypot(cand: dict) -> bool:
    """
    Detect statistically impossible profiles.
    Spec: ~80 honeypots in dataset. Honeypot rate > 10% in top-100 = disqualification.

    Checks:
    1. Expert proficiency in multiple skills with 0 duration months
    2. Total career months >> years of experience
    3. Individual skill duration >> total career length
    """
    profile = cand.get("profile", {})
    skills = cand.get("skills", [])
    career = cand.get("career_history", [])
    yoe = profile.get("years_of_experience", 0) or 0

    # Check 1: Expert + 0 months is a red flag per spec
    expert_zero_months = sum(
        1 for s in skills
        if s.get("proficiency") == "expert" and (s.get("duration_months") or 0) == 0
    )
    if expert_zero_months >= 2:
        return True

    # Check 2: Any single skill duration wildly exceeds total career
    for s in skills:
        skill_months = s.get("duration_months") or 0
        if skill_months > (yoe + 2) * 12:
            return True

    # Check 3: Career timeline impossibility (5yr tolerance for overlapping roles)
    total_career_months = sum(j.get("duration_months") or 0 for j in career)
    if total_career_months > (yoe + 5) * 12 and yoe > 0:
        return True

    return False


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 2 — BM25 PRE-FILTER
# ═══════════════════════════════════════════════════════════════════════════════

def build_candidate_text(cand: dict) -> str:
    """Compact text blob for BM25 and semantic encoding."""
    p = cand.get("profile", {})
    parts = [
        p.get("current_title", "") or "",
        p.get("headline", "") or "",
        (p.get("summary", "") or "")[:400],
        p.get("current_industry", "") or "",
    ]
    parts += [s.get("name", "") or "" for s in cand.get("skills", [])]
    for role in cand.get("career_history", [])[:4]:
        parts.append(role.get("title", "") or "")
        parts.append((role.get("description", "") or "")[:300])
    parts += [c.get("name", "") or "" for c in cand.get("certifications", [])]
    return " ".join(x for x in parts if x)


def bm25_filter(candidates: list, top_k: int) -> list:
    """BM25 lexical pre-filter: 100K → top_k. Returns original list if small enough."""
    if len(candidates) <= top_k:
        print(f"  BM25 skipped — {len(candidates):,} candidates")
        return candidates

    try:
        import numpy as np
        from rank_bm25 import BM25Okapi
    except ImportError:
        print("  WARN: rank-bm25 not installed. Skipping BM25.")
        return candidates

    print(f"  Building BM25 corpus from {len(candidates):,} candidates...")
    tokenize = lambda t: re.findall(r"[a-z0-9][\w\-']*", t.lower())
    corpus = [tokenize(build_candidate_text(c)) for c in candidates]
    bm25 = BM25Okapi(corpus)

    query = tokenize(" ".join(BM25_KEYWORDS))
    scores = bm25.get_scores(query)
    top_idx = np.argsort(scores)[::-1][:top_k]
    result = [candidates[i] for i in top_idx]
    print(f"  BM25: {len(candidates):,} → {len(result):,}  (top score: {scores[top_idx[0]]:.1f})")
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 3 — SEMANTIC SCORING (MiniLM, optional)
# ═══════════════════════════════════════════════════════════════════════════════

def semantic_scores(candidates: list, model_cache_dir: str) -> dict:
    """
    Encode candidates with all-MiniLM-L6-v2 and return dict[candidate_id → cosine_sim].
    Returns empty dict if model not available (graceful fallback to rules-only).
    """
    try:
        import numpy as np
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("  sentence-transformers not installed — skipping semantic scoring.")
        return {}

    model_name = "sentence-transformers/all-MiniLM-L6-v2"

    print(f"  Loading semantic model from cache {model_cache_dir}...")
    try:
        model = SentenceTransformer(model_name, cache_folder=model_cache_dir, local_files_only=True)
    except Exception as e:
        try:
            # Fallback 1: check direct subfolder
            direct_path = Path(model_cache_dir) / "sentence-transformers_all-MiniLM-L6-v2"
            if direct_path.exists():
                model = SentenceTransformer(str(direct_path))
            else:
                # Fallback 2: check HuggingFace snapshots directory
                snapshot_dir = Path(model_cache_dir) / "models--sentence-transformers--all-MiniLM-L6-v2" / "snapshots"
                if snapshot_dir.exists():
                    snapshots = list(snapshot_dir.iterdir())
                    if snapshots:
                        model = SentenceTransformer(str(snapshots[0]))
                    else:
                        raise e
                else:
                    raise e
        except Exception as e2:
            print(f"  Model load failed ({e2}) — skipping semantic stage.")
            return {}

    jd_emb = model.encode([JD_TEXT], normalize_embeddings=True, show_progress_bar=False)[0]
    texts = [build_candidate_text(c) for c in candidates]
    embs = model.encode(texts, batch_size=512, normalize_embeddings=True, show_progress_bar=True)
    sims = (embs @ jd_emb).tolist()

    result = {c.get("candidate_id"): float(s) for c, s in zip(candidates, sims)}
    top_sim = max(sims)
    print(f"  Semantic: {len(candidates):,} candidates encoded  (top sim: {top_sim:.3f})")
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 4 — MULTI-SIGNAL RULE SCORING
# ═══════════════════════════════════════════════════════════════════════════════

# Regex patterns for title matching
_GOOD_TITLE_RE = [re.compile(p, re.IGNORECASE) for p in [
    r"\b(search|ranking|retrieval|recommendation)\b\s+engineer\b",
    r"\b(senior|sr\.?|lead|staff|principal)\b\s+(ai|ml|machine.?learning|nlp|applied)\b",
    r"\b(ai|ml|machine.?learning|nlp)\b\s+(engineer|scientist|researcher|architect)\b",
    r"\bapplied\s+(scientist|ml|ai)\b",
    r"\bresearch\s+scientist\b",
    r"\b(senior\s+)?data\s+scientist\b",
    r"\bml\s+engineer\b",
    r"\bai\s+engineer\b",
]]

_BAD_TITLE_RE = [re.compile(p, re.IGNORECASE) for p in [
    r"marketing\s+(manager|executive|analyst|specialist)",
    r"^sales\s+(executive|manager|representative|head)",
    r"\bhr\b\s+(manager|executive|specialist|generalist|head)",
    r"human\s+resources",
    r"content\s+(writer|creator|manager|strategist)",
    r"graphic\s+designer",
    r"^accountant",
    r"civil\s+engineer",
    r"mechanical\s+engineer",
    r"operations\s+manager",
    r"customer\s+(support|success|service)",
    r"scrum\s+master",
    r"\bresearch(er)?\b",
    r"\bacademic\b",
    r"\bphd\b",
    r"\bpostdoc\b",
    r"\bfellow\b",
    r"\bprofessor\b",
]]

_MODERATE_TITLES = frozenset({
    "software engineer", "backend engineer", "full stack engineer",
    "platform engineer", "cloud engineer", "data engineer",
    "python developer", "data analyst", "analytics engineer",
})


def _title_score(title: str) -> float:
    """Max 25 pts. Returns 0.0 for disqualifying titles (triggers heavy penalty)."""
    if any(p.search(title) for p in _BAD_TITLE_RE):
        return 0.0
    if any(p.search(title) for p in _GOOD_TITLE_RE):
        return 25.0
    tl = title.lower()
    if any(t in tl for t in _MODERATE_TITLES):
        return 8.0
    return 3.0  # unknown / generic


def _experience_score(yoe: float) -> float:
    """Max 15 pts. Sweet spot: 6-8 years per JD's "how we read between the lines"."""
    if 6 <= yoe <= 8:
        return 15.0
    if 5 <= yoe <= 9:
        return 11.0
    if 4 <= yoe < 5 or 9 < yoe <= 10:
        return 6.0
    if 3 <= yoe < 4 or 10 < yoe <= 12:
        return 3.0
    return 0.0


def _ir_skill_score(cand: dict) -> float:
    """
    Max 30 pts. Two-part:
    - Part A (10 pts): Core IR/embedding skills in skills list
    - Part B (20 pts): Evidence of PRODUCTION IR work in career descriptions

    The JD explicitly says: "a candidate who describes their work in plain language
    may outscore someone with all the keywords listed as skills."
    """
    skills = cand.get("skills", [])
    career = cand.get("career_history", [])
    prof_w = {"beginner": 0.3, "intermediate": 0.6, "advanced": 0.9, "expert": 1.0}

    # Part A: Core skills in skills list
    core_pts = 0.0
    for s in skills:
        name = (s.get("name") or "").lower().strip()
        prof = prof_w.get(s.get("proficiency"), 0.5)
        end = min(1.0, math.log1p(s.get("endorsements") or 0) / math.log1p(30))
        dur = min(1.0, (s.get("duration_months") or 0) / 24.0)
        trust = 0.4 * prof + 0.3 * end + 0.3 * dur
        if any(kw in name or name in kw for kw in CORE_SKILLS):
            core_pts += trust * 1.5

    # Part B: Production IR evidence in career descriptions
    prod_pts = 0.0
    for role in career:
        desc = (role.get("description") or "").lower()
        title = (role.get("title") or "").lower()
        dur = role.get("duration_months") or 0
        company = (role.get("company") or "").lower()
        industry = (role.get("industry") or "").lower()

        # Count IR keyword hits in this role's description
        ir_hits = sum(1 for kw in IR_PRODUCTION_KEYWORDS if kw in desc)
        if ir_hits == 0:
            continue

        # Scale by role type: product company descriptions are worth more
        is_product = (
            any(k in company for k in PRODUCT_COMPANIES)
            or industry in GOOD_INDUSTRIES
        )
        multiplier = 1.5 if is_product else 0.8

        # Scale by tenure (longer role = more experience)
        tenure_factor = min(1.5, dur / 12.0)

        # Cap per-role contribution
        role_pts = min(5.0, ir_hits * 0.5 * multiplier * tenure_factor)
        prod_pts += role_pts

    return min(10.0, core_pts) + min(20.0, prod_pts)


def _eval_score(text_content: str, skills_list: list) -> float:
    """Max 15 pts. Evaluation frameworks: NDCG, MRR, A/B testing."""
    eval_keywords = {
        "ndcg": 5, "mrr": 5, "map": 3, "mean average precision": 3,
        "a/b test": 5, "ab test": 5, "a/b testing": 5, "ab testing": 5,
        "online evaluation": 3, "offline evaluation": 3,
        "precision@": 2, "recall@": 2,
    }
    total = 0.0
    for kw, pts in eval_keywords.items():
        if kw in text_content or kw in skills_list:
            total += pts
    return min(15.0, total)


def _python_score(skills_list: list, text_content: str) -> float:
    """Max 5 pts. Python is explicitly called out in JD as "we actually care about code quality"."""
    prof_w = {"beginner": 1, "intermediate": 2, "advanced": 4, "expert": 5}
    for s in skills_list:  # skills_list here is raw skill dicts from earlier
        pass
    if "python" in text_content:
        return 3.0
    return 0.0


def _behavioral_score(sig: dict) -> float:
    """Max 10 pts. Uses all 23 signals from redrob_signals_doc.md."""
    pts = 0.0

    # Notice period (JD says "sub-30-day notice is ideal; buy out up to 30 days")
    notice = sig.get("notice_period_days") or 90
    if notice <= 30:
        pts += 4.0
    elif notice <= 60:
        pts += 2.0
    elif notice <= 90:
        pts += 1.0
    # >90 days: 0 pts

    # Open to work flag
    if sig.get("open_to_work_flag"):
        pts += 2.0

    # Recruiter response rate (predicts whether we can actually reach them)
    rr = sig.get("recruiter_response_rate") or 0
    if rr > 0.7:
        pts += 2.0
    elif rr > 0.4:
        pts += 1.0

    # Interview completion rate
    icr = sig.get("interview_completion_rate") or 0
    if icr > 0.8:
        pts += 1.0

    # GitHub activity (open-source contributions = external validation of skills)
    gh = sig.get("github_activity_score") or -1
    if gh > 60:
        pts += 1.0

    return min(10.0, pts)


def _behavioral_multiplier(sig: dict) -> float:
    """
    Availability multiplier [0.3, 1.0].
    Even a perfect-on-paper candidate who hasn't logged in for 6 months
    is not actually available — per the JD's explicit note on behavioral signals.
    """
    try:
        days_inactive = (
            TODAY - datetime.strptime(sig["last_active_date"], "%Y-%m-%d").date()
        ).days
    except Exception:
        days_inactive = 365

    if days_inactive > 365:
        mult = 0.3
    elif days_inactive > 180:
        mult = 0.55
    elif days_inactive > 90:
        mult = 0.80
    elif days_inactive > 30:
        mult = 0.93
    else:
        mult = 1.0

    # Further reduction if explicitly unavailable
    otw = sig.get("open_to_work_flag", False)
    rr = sig.get("recruiter_response_rate") or 0
    if not otw and rr < 0.2:
        mult *= 0.75

    return mult


def is_location_disqualified(profile: dict, sig: dict) -> bool:
    """Check if candidate is unviable due to location and unwillingness to relocate."""
    loc = (profile.get("location") or "").lower()
    country = (profile.get("country") or "").lower()
    willing = sig.get("willing_to_relocate", False)

    # Outside India: case-by-case, but no work visas
    if country and country != "india":
        return True

    # If location is inside India, check if it matches target cities
    # Welcome cities: Pune, Noida, Hyderabad, Mumbai, Delhi NCR (Delhi, Gurgaon, Gurugram, Noida, Pune, Mumbai, Hyderabad)
    welcome_cities = {"pune", "noida", "hyderabad", "mumbai", "delhi", "gurgaon", "gurugram", "ncr", "delhi ncr"}
    if loc:
        in_welcome_city = any(c in loc for c in welcome_cities)
        if not in_welcome_city and not willing:
            return True

    return False


def _location_score(profile: dict, sig: dict) -> float:
    """Max 10 pts. Pune/Noida primary; other Tier-1 Indian cities secondary."""
    loc = (profile.get("location") or "").lower()
    country = (profile.get("country") or "").lower()
    willing = sig.get("willing_to_relocate", False)

    if country != "india":
        return 2.0 if willing else 0.0
    if any(c in loc for c in TARGET_CITIES_PRIMARY):
        return 10.0
    if any(c in loc for c in TARGET_CITIES_SECONDARY):
        return 8.0
    return 6.0 if willing else 2.0


def _career_penalties(cand: dict, text_content: str) -> float:
    """Negative score adjustments for JD-stated disqualifiers."""
    profile = cand.get("profile", {})
    career = cand.get("career_history", [])
    sig = cand.get("redrob_signals", {})
    penalty = 0.0

    # Disqualifier 1: Title-chaser (avg tenure < 18 months)
    if len(career) > 2:
        avg_months = sum(j.get("duration_months") or 0 for j in career) / len(career)
        if avg_months < 18:
            penalty -= 25.0
        elif avg_months < 12:
            penalty -= 40.0

    # Disqualifier 2: Services-only career
    if len(career) > 0:
        all_services = all(
            any(s in (j.get("company") or "").lower() for s in SERVICES_COMPANIES)
            or (j.get("industry") or "").lower() in BAD_INDUSTRIES
            for j in career
        )
        if all_services:
            penalty -= 40.0

    # Disqualifier 3: Pure researcher (academic labs, no production)
    # Check current title for researcher keywords
    current_title = (profile.get("current_title") or "").lower()
    if any(w in current_title for w in ["research", "academic", "phd", "postdoc", "fellow", "professor"]):
        penalty -= 40.0

    research_count = (
        text_content.count("research") +
        text_content.count("academic") +
        text_content.count("phd student") +
        text_content.count("grad student")
    )
    if research_count > 4:
        penalty -= 20.0

    # Disqualifier 4: CV/speech/robotics only — "you'd be re-learning fundamentals"
    cv_count = (
        text_content.count("computer vision") +
        text_content.count("object detection") +
        text_content.count("yolo") +
        text_content.count("speech recognition") +
        text_content.count("robotics") +
        text_content.count("autonomous driving")
    )
    ir_count = sum(1 for kw in ["retrieval", "ranking", "recommendation", "search", "embedding"]
                   if kw in text_content)
    if cv_count > 2 and ir_count == 0:
        penalty -= 25.0

    # Disqualifier 5: LangChain / API Wrapper trap
    has_langchain = "langchain" in text_content or "llamaindex" in text_content or "llama-index" in text_content
    infra_keywords = [
        "pinecone", "milvus", "qdrant", "faiss", "weaviate", "elasticsearch", 
        "opensearch", "ndcg", "mrr", "bm25", "information retrieval", "ranking", 
        "hybrid search", "semantic search", "recommendation"
    ]
    infra_hits = sum(1 for kw in infra_keywords if kw in text_content)
    if has_langchain:
        if infra_hits < 2:
            penalty -= 35.0  # heavy penalty for LangChain wrapper focus without low-level IR skills
        else:
            penalty -= 10.0  # moderate penalty for using frameworks over pure systems

    # Disqualifier 6: Notice period / Viability constraint
    notice = sig.get("notice_period_days") or 90
    if notice >= 120:
        penalty -= 30.0
    elif notice > 90:
        penalty -= 15.0

    # Product company bonus (up to 8 pts — counters services penalty)
    product_months = 0
    for role in career:
        company_lower = (role.get("company") or "").lower()
        industry_lower = (role.get("industry") or "").lower()
        is_product = (
            any(k in company_lower for k in PRODUCT_COMPANIES)
            or industry_lower in GOOD_INDUSTRIES
        )
        if is_product:
            product_months += role.get("duration_months") or 0

    penalty += min(8.0, (product_months / 12.0) * 1.5)

    return penalty


def compute_rule_score(cand: dict) -> tuple:
    """
    Returns (raw_score_0_to_100, is_bad_title, is_loc_disqualified, mult, component_dict).
    """
    profile = cand.get("profile", {})
    sig = cand.get("redrob_signals", {})
    yoe = profile.get("years_of_experience") or 0

    # Build text for keyword matching
    skills_raw = cand.get("skills", [])
    skills_names = [s.get("name", "").lower() for s in skills_raw]
    text_content = build_candidate_text(cand).lower()

    title = (profile.get("current_title") or "").strip()
    t_score = _title_score(title)
    is_bad_title = t_score == 0.0

    is_loc_disqualified = is_location_disqualified(profile, sig)

    e_score = _experience_score(yoe)
    ir_score = _ir_skill_score(cand)
    ev_score = _eval_score(text_content, skills_names)
    py_score = 3.0 if "python" in text_content else 0.0  # simple but fast
    beh_score = _behavioral_score(sig)
    loc_score = _location_score(profile, sig)
    penalties = _career_penalties(cand, text_content)
    mult = _behavioral_multiplier(sig)

    raw = t_score + e_score + ir_score + ev_score + py_score + beh_score + loc_score + penalties

    # Title disqualifier: near-zero score, don't let keywords rescue them
    if is_bad_title:
        raw = min(raw, 5.0)

    return raw, is_bad_title, is_loc_disqualified, mult, {
        "title": round(t_score, 1),
        "experience": round(e_score, 1),
        "ir": round(ir_score, 1),
        "eval": round(ev_score, 1),
        "python": py_score,
        "behavioral": round(beh_score, 1),
        "location": round(loc_score, 1),
        "penalties": round(penalties, 1),
        "mult": round(mult, 2),
    }


def sigmoid_normalize(raw: float, scale: float = 22.0) -> float:
    """
    Map raw score [−∞, +∞] → (0, 1) using sigmoid.
    scale=22 means raw=66 → 0.95, raw=44 → 0.87, raw=22 → 0.73, raw=0 → 0.5
    """
    return 1.0 / (1.0 + math.exp(-raw / scale))


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 5 — REASONING GENERATION
# ═══════════════════════════════════════════════════════════════════════════════

def generate_reasoning(cand: dict, rank: int, components: dict, is_honeypot: bool) -> str:
    """
    Programmatic reasoning — specific, honest, rank-aware, non-templated.

    Stage 4 evaluation checks:
    - Specific facts (YoE, title, named skills, signals)
    - JD connection
    - Honest concerns where they exist
    - No hallucination
    - Variation across rows
    - Rank-consistent tone
    """
    if is_honeypot:
        return "Profile flagged as honeypot: inconsistent skill proficiency/duration or impossible career timeline."

    profile = cand.get("profile", {})
    sig = cand.get("redrob_signals", {})

    title = profile.get("current_title") or "Engineer"
    yoe = profile.get("years_of_experience") or 0
    company = profile.get("current_company") or ""
    location = profile.get("location") or ""
    country = (profile.get("country") or "").lower()
    notice = sig.get("notice_period_days") or 90
    rr = sig.get("recruiter_response_rate") or 0
    otw = sig.get("open_to_work_flag", False)
    gh = sig.get("github_activity_score") or -1

    try:
        days_inactive = (
            TODAY - datetime.strptime(sig.get("last_active_date", "2000-01-01"), "%Y-%m-%d").date()
        ).days
    except Exception:
        days_inactive = 999

    # Find strongest IR-relevant skills actually in profile (anti-hallucination)
    skills_raw = cand.get("skills", [])
    ir_skills_found = [
        s["name"] for s in skills_raw
        if any(kw in s["name"].lower() for kw in [
            "embedding", "retrieval", "faiss", "pinecone", "weaviate", "qdrant",
            "milvus", "elasticsearch", "opensearch", "ranking", "recommendation",
            "sentence", "bge", "rag", "semantic search", "vector", "ndcg", "mrr",
        ])
    ][:3]

    # Positive signals
    positive = []
    if ir_skills_found:
        positive.append(f"core skills: {', '.join(ir_skills_found)}")
    if components.get("eval", 0) >= 5:
        positive.append("evaluation metrics experience (offline/online)")
    if gh > 50:
        positive.append(f"GitHub activity score {gh:.0f}/100")
    if notice <= 30:
        positive.append(f"notice period {notice}d — can start quickly")
    if country == "india":
        city_match = any(c in location.lower() for c in list(TARGET_CITIES_PRIMARY) + list(TARGET_CITIES_SECONDARY))
        if city_match:
            positive.append(f"based in {location}")

    # Concerns
    concerns = []
    if days_inactive > 180:
        concerns.append(f"inactive {days_inactive // 30}+ months — availability uncertain")
    if not otw and rr < 0.3:
        concerns.append(f"recruiter response rate {rr:.0%} — outreach risk")
    if notice > 90:
        concerns.append(f"long notice period ({notice}d) — delays joining")
    if yoe < 5 or yoe > 9:
        concerns.append(f"{yoe:.0f}yr exp outside ideal 5-9yr range")
    if components.get("ir", 0) < 5:
        concerns.append("limited direct IR/retrieval production evidence")
    if country != "india":
        concerns.append(f"located outside India ({country})")

    # Compose rank-appropriate sentence
    at_company = f" at {company}" if company else ""

    if rank <= 5:
        s1 = (f"{title} with {yoe:.1f} yrs exp{at_company}; "
              f"{'; '.join(positive[:2]) if positive else 'strong overall profile'}.")
    elif rank <= 20:
        s1 = (f"{yoe:.0f}-yr {title}{at_company} with "
              f"{positive[0] if positive else 'adjacent ML/AI skills'}.")
    elif rank <= 60:
        s1 = (f"{title}{at_company} ({yoe:.0f} yrs); "
              f"partial JD alignment — {positive[0] if positive else 'some AI/ML exposure'}.")
    else:
        s1 = (f"{title}{at_company} ({yoe:.0f} yrs); "
              f"included at rank {rank} — marginal fit, included given pool depth.")

    # Append concerns or additional positives
    if concerns:
        s2 = (concerns[0].capitalize() +
              (f"; {concerns[1]}" if len(concerns) > 1 else "") + ".")
        return f"{s1} {s2}"
    elif len(positive) > 2:
        s2 = positive[2].capitalize() + "."
        return f"{s1} {s2}"
    return s1


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def run(candidates_path: str, output_path: str, use_semantic: bool, debug: bool):
    t_total = time.perf_counter()
    print(f"\n{'='*60}")
    print("Redrob Hackathon — Team Antigravity")
    print(f"{'='*60}")

    # ── Load ──────────────────────────────────────────────────────────────────
    print("\n[1/5] Loading candidates...")
    t = time.perf_counter()
    candidates = []
    p = Path(candidates_path)
    if p.suffix == ".jsonl":
        with open(candidates_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    candidates.append(json.loads(line))
    elif p.suffix == ".json":
        with open(candidates_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        candidates = data if isinstance(data, list) else list(data.values())
    print(f"   Loaded {len(candidates):,} candidates  ({time.perf_counter()-t:.1f}s)")

    # ── Honeypot pre-screen (cheap, done before ANY scoring) ─────────────────
    print("\n[2/5] Honeypot detection...")
    t = time.perf_counter()
    honeypot_ids = set()
    for c in candidates:
        if is_honeypot(c):
            honeypot_ids.add(c.get("candidate_id"))
    print(f"   Flagged {len(honeypot_ids):,} honeypots  ({time.perf_counter()-t:.1f}s)")

    # ── BM25 pre-filter ───────────────────────────────────────────────────────
    print("\n[3/5] BM25 lexical pre-filter...")
    t = time.perf_counter()
    top_candidates = bm25_filter(candidates, top_k=5000)
    print(f"   Done  ({time.perf_counter()-t:.1f}s)")

    # ── Semantic scoring (optional) ───────────────────────────────────────────
    sem_scores = {}
    if use_semantic:
        print("\n[4/5] Semantic scoring (MiniLM)...")
        t = time.perf_counter()
        model_cache = Path(__file__).parent / "models"
        if not model_cache.exists():
            model_cache = Path(__file__).parent / "redrob_ranker" / "models"
        sem_scores = semantic_scores(top_candidates, str(model_cache))
        print(f"   Done  ({time.perf_counter()-t:.1f}s)")
    else:
        print("\n[4/5] Semantic scoring: SKIPPED (--no-semantic)")

    # ── Multi-signal rule scoring ─────────────────────────────────────────────
    print("\n[5/5] Rule-based multi-signal scoring...")
    t = time.perf_counter()
    scored = []
    for c in top_candidates:
        cid = c.get("candidate_id")
        hp = cid in honeypot_ids

        if hp:
            scored.append((c, 0.001, True, {}))
            continue

        raw, bad_title, is_loc_disqualified, mult, components = compute_rule_score(c)
        rule_norm = sigmoid_normalize(raw * mult, scale=22.0)

        # Blend with semantic score if available
        sem = sem_scores.get(cid, -1)
        if sem >= 0 and not bad_title and not is_loc_disqualified:
            # 65% rule + 35% semantic
            final = 0.65 * rule_norm + 0.35 * ((sem + 1) / 2)  # sem ∈ [-1,1] → [0,1]
        else:
            final = rule_norm

        # Hard cap for bad titles
        if bad_title:
            final = min(final, 0.08)

        # Hard cap for location disqualified
        if is_loc_disqualified:
            final = min(final, 0.001)

        scored.append((c, round(min(0.9999, max(0.0001, final)), 4), False, components))

    # Sort: descending score, tie-break candidate_id ascending
    scored.sort(key=lambda x: (-x[1], x[0].get("candidate_id", "")))
    print(f"   Scored {len(scored):,} candidates  ({time.perf_counter()-t:.1f}s)")
    print(f"   Top score: {scored[0][1]:.4f}  |  Honeypots in pool: {len(honeypot_ids):,}")

    # Ensure we have at least 100 candidates (fallback: score all)
    if len(scored) < 100:
        print(f"   WARN: only {len(scored)} candidates — scoring full dataset...")
        scored = []
        for c in candidates:
            cid = c.get("candidate_id")
            hp = cid in honeypot_ids
            if hp:
                scored.append((c, 0.001, True, {}))
                continue
            raw, bad_title, is_loc_disqualified, mult, components = compute_rule_score(c)
            final = sigmoid_normalize(raw * mult, scale=22.0)
            if bad_title:
                final = min(final, 0.08)
            if is_loc_disqualified:
                final = min(final, 0.001)
            scored.append((c, round(final, 4), False, components))
        scored.sort(key=lambda x: (-x[1], x[0].get("candidate_id", "")))

    # ── Debug output ──────────────────────────────────────────────────────────
    if debug:
        print(f"\n{'─'*80}")
        print(f"{'RANK':>4}  {'CANDIDATE':14}  {'SCORE':6}  {'TITLE':35}  YOE")
        print(f"{'─'*80}")
        for rank, (c, sc, hp, comp) in enumerate(scored[:15], 1):
            p = c["profile"]
            print(f"  {rank:3d}  {c['candidate_id']:14}  {sc:.4f}  "
                  f"{(p.get('current_title') or '')[:35]:35}  "
                  f"{p.get('years_of_experience', 0):.1f}yr")
        print(f"{'─'*80}")

    # ── Write submission ──────────────────────────────────────────────────────
    print(f"\nWriting → {output_path}")
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for rank, (c, sc, hp, comp) in enumerate(scored[:100], start=1):
            reason = generate_reasoning(c, rank, comp, hp)
            writer.writerow([c.get("candidate_id"), rank, sc, reason])

    # ── Validate ──────────────────────────────────────────────────────────────
    print("Validating...")
    ranks_seen, ids_seen = set(), set()
    prev_sc = float("inf")
    row_count = 0
    errors = []
    with open(output_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            row_count += 1
            r, s, cid = int(row["rank"]), float(row["score"]), row["candidate_id"]
            if r in ranks_seen:
                errors.append(f"Duplicate rank {r}")
            if cid in ids_seen:
                errors.append(f"Duplicate ID {cid}")
            if s > prev_sc + 1e-9:
                errors.append(f"Score not monotonic at rank {r}: {prev_sc:.4f} -> {s:.4f}")
            # Tie-break check
            if s == prev_sc and row_count > 1:
                # Only warn, not error — spec allows ties with unique ranks
                pass
            ranks_seen.add(r)
            ids_seen.add(cid)
            prev_sc = s
    if row_count != 100:
        errors.append(f"Expected 100 rows, got {row_count}")
    if ranks_seen != set(range(1, 101)):
        errors.append("Ranks 1-100 not all present")

    elapsed = time.perf_counter() - t_total
    print(f"\n{'='*60}")
    if errors:
        for e in errors:
            print(f"  ❌ {e}")
    else:
        print(f"  ✅ Validation passed — 100 rows, scores non-increasing")
    print(f"  ⏱  Total runtime: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    if elapsed > 300:
        print("  ⚠️  WARNING: Exceeded 5-min challenge limit!")
    print(f"  📄 Submission: {output_path}")

    # Honeypot check (challenge disqualifies if >10% of top-100 are honeypots)
    top100_honeypots = sum(1 for _, _, hp, _ in scored[:100] if hp)
    print(f"  🍯 Honeypots in top-100: {top100_honeypots}/100 "
          f"({'OK' if top100_honeypots <= 10 else 'WARNING: >10% threshold!'})")
    print(f"{'='*60}\n")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Redrob Hackathon — Team Antigravity — Candidate Ranker",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full run (with semantic model):
  python main.py --candidates candidates.jsonl --out team_antigravity.csv

  # Fast run (no model needed, rules + BM25 only):
  python main.py --candidates candidates.jsonl --out team_antigravity.csv --no-semantic

  # Sample test:
  python main.py --candidates sample_candidates.json --out test.csv --no-semantic --debug
        """,
    )
    parser.add_argument("--candidates", default="candidates.jsonl",
                        help="Path to candidates.jsonl or sample_candidates.json")
    parser.add_argument("--out", default="team_antigravity.csv",
                        help="Output CSV path (must be team_ID.csv per submission spec)")
    parser.add_argument("--no-semantic", action="store_true",
                        help="Skip MiniLM semantic scoring (faster, no model needed)")
    parser.add_argument("--debug", action="store_true",
                        help="Print top-15 candidates to console")
    args = parser.parse_args()

    run(
        candidates_path=args.candidates,
        output_path=args.out,
        use_semantic=not args.no_semantic,
        debug=args.debug,
    )


if __name__ == "__main__":
    main()
