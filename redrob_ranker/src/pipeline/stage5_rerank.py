"""
stage5_rerank.py — Final weighted re-ranking and reasoning generation.

Combines:
  - Semantic similarity scores (Stage 3 + Stage 4)
  - Rule-based multi-signal scoring (title, career, skills, experience, location)
  - Behavioral signals (activity, response rate, notice period, GitHub)
  - Profile quality (completeness, verification, recruiter interest)

No network calls. Fully deterministic. Programmatic reasoning text.
"""

import math
import re
from datetime import date, datetime

from src.config.skills_taxonomy import (
    TIER1_SKILLS, TIER2_SKILLS, TIER_WEIGHTS,
    PRODUCT_COMPANIES, CONSULTING_COMPANIES,
    AI_INDUSTRIES, BAD_INDUSTRIES,
    TARGET_CITIES,
    JD_TITLE_KEYWORDS, JD_SUMMARY_KEYWORDS,
)
from src.utils.logger import logger

TODAY = date.today()

# ── Title patterns ────────────────────────────────────────────────────────────
GOOD_TITLE_PATTERNS = [
    re.compile(p, re.IGNORECASE) for p in [
        r"(senior|sr\.?|lead|staff|principal)\s+(ai|ml|machine\s*learning|nlp|data\s*science)",
        r"(ai|ml|machine\s*learning|nlp)\s+(engineer|scientist|researcher|architect)",
        r"applied\s+(scientist|ml|ai)",
        r"(search|ranking|retrieval|recommendation)\s+engineer",
        r"(senior\s+)?data\s+scientist",
        r"research\s+scientist",
        r"(junior\s+)?(ml|ai)\s+engineer",
        r"machine\s+learning\s+engineer",
    ]
]

BAD_TITLE_PATTERNS = [
    re.compile(p, re.IGNORECASE) for p in [
        r"marketing\s+(manager|executive|analyst)",
        r"^sales\s+(executive|manager|representative)",
        r"hr\s+(manager|executive|specialist|generalist)",
        r"content\s+(writer|creator|manager)",
        r"graphic\s+designer",
        r"^accountant$",
        r"civil\s+engineer",
        r"mechanical\s+engineer",
        r"operations\s+manager",
        r"customer\s+(support|success)",
    ]
]

MODERATE_TITLES = frozenset({
    "software engineer", "backend engineer", "full stack engineer",
    "platform engineer", "cloud engineer", "data engineer",
    "analytics engineer", "python developer", "data analyst",
})

CAREER_ML_KEYWORDS = [
    "embedding", "vector", "retrieval", "ranking", "recommendation",
    "nlp", "fine-tun", "transformer", "bert", "gpt", "llm",
    "search", "semantic", "similarity", "a/b test", "evaluation",
    "production", "deployed", "inference", "model", "training", "pipeline",
]


# ── Honeypot Detection ────────────────────────────────────────────────────────

def _detect_honeypot(candidate: dict) -> bool:
    """Flag statistically impossible profiles."""
    skills = candidate.get("skills", [])
    yoe: float = candidate["profile"].get("years_of_experience", 0) or 0
    career = candidate.get("career_history", [])

    expert_count = sum(1 for s in skills if s.get("proficiency") == "expert")
    if expert_count >= 6 and yoe < 3:
        return True

    total_skill_months = sum(s.get("duration_months", 0) or 0 for s in skills)
    yoe_months = yoe * 12
    if yoe_months > 0 and (total_skill_months / yoe_months) > 5.0 and yoe < 3:
        return True

    past_months = sum(
        r.get("duration_months", 0) or 0
        for r in career if not r.get("is_current", False)
    )
    if past_months > (yoe + 2) * 12 and past_months > 36:
        return True

    return False


# ── Component Scorers ─────────────────────────────────────────────────────────

def _score_title(title: str) -> float:
    """30 pts max."""
    if any(p.search(title) for p in BAD_TITLE_PATTERNS):
        return 0.0
    if any(p.search(title) for p in GOOD_TITLE_PATTERNS):
        return 30.0
    if any(t in title.lower() for t in MODERATE_TITLES):
        return 10.0
    return 5.0


def _score_career(candidate: dict) -> float:
    """25 pts max — scores actual work done, not just claimed skills."""
    career = candidate.get("career_history", [])
    if not career:
        return 0.0

    score = 0.0
    total_product_months = 0
    total_ml_hits = 0
    is_consulting_only = True

    for role in career:
        company_lower = (role.get("company") or "").lower()
        title_lower = (role.get("title") or "").lower()
        desc_lower = (role.get("description") or "").lower()
        duration = role.get("duration_months", 0) or 0
        industry_lower = (role.get("industry") or "").lower()

        is_product = any(k in company_lower for k in PRODUCT_COMPANIES)
        is_consulting = any(k in company_lower for k in CONSULTING_COMPANIES)

        if not is_product and industry_lower in AI_INDUSTRIES:
            is_product = True
        if not is_consulting and industry_lower in BAD_INDUSTRIES:
            is_consulting = True

        if is_product and not is_consulting:
            total_product_months += duration

        if not is_consulting:
            is_consulting_only = False

        if any(p.search(title_lower) for p in GOOD_TITLE_PATTERNS):
            score += min(5.0, (duration / 12.0) * 2.5)

        ml_hits = sum(1 for kw in CAREER_ML_KEYWORDS if kw in desc_lower)
        total_ml_hits += ml_hits

    score += min(10.0, (total_product_months / 12.0) * 1.5)
    score += min(10.0, total_ml_hits * 0.5)

    if is_consulting_only:
        score *= 0.5

    return min(25.0, score)


def _score_skills(candidate: dict) -> float:
    """20 pts max — trust-weighted: proficiency × endorsements × duration."""
    skills = candidate.get("skills", [])
    prof_weights = {"beginner": 0.3, "intermediate": 0.6, "advanced": 0.9, "expert": 1.0}
    core_score = bonus_score = domain_penalty = 0.0

    # Category matching to avoid double-crediting
    # (e.g. Pinecone + Vector Database)
    SPECIFIC_VECTOR = {"pinecone", "weaviate", "qdrant", "milvus", "faiss", "opensearch", "elasticsearch"}
    GENERIC_VECTOR = {"vector database", "vector search", "vector db", "vector", "embeddings", "embedding"}

    SPECIFIC_TUNING = {"lora", "qlora", "peft"}
    GENERIC_TUNING = {"fine-tuning", "fine tuning", "model fine-tuning", "transfer learning"}

    SPECIFIC_LLM = {"gpt", "llama", "mistral", "gemini", "claude", "langchain", "openai", "chatgpt", "bert", "sentence-transformers", "sentence transformers", "bge", "e5"}
    GENERIC_LLM = {"llm", "llms", "large language model", "large language models", "generative ai", "gen ai", "genai", "nlp", "natural language processing", "transformers", "transformer"}

    # Track if candidate has specific tools in their profile
    has_specific_vector = False
    has_specific_tuning = False
    has_specific_llm = False

    def matches_any(name: str, skill_set: set | frozenset) -> bool:
        return any(kw in name or name in kw for kw in skill_set)

    # First pass: check for specific tools
    for skill in skills:
        name = (skill.get("name") or "").lower().strip()
        if not name:
            continue
        if matches_any(name, SPECIFIC_VECTOR):
            has_specific_vector = True
        if matches_any(name, SPECIFIC_TUNING):
            has_specific_tuning = True
        if matches_any(name, SPECIFIC_LLM):
            has_specific_llm = True

    # Filter out generic terms from global sets to avoid redundant matching
    all_generic = GENERIC_VECTOR | GENERIC_TUNING | GENERIC_LLM
    tier1_cleaned = frozenset(TIER1_SKILLS - all_generic)
    tier2_cleaned = frozenset(TIER2_SKILLS - all_generic)

    # Track whether generic matcher has already been credited
    credited_generic_vector = False
    credited_generic_tuning = False
    credited_generic_llm = False

    for skill in skills:
        name = (skill.get("name") or "").lower().strip()
        if not name:
            continue
        prof = prof_weights.get(skill.get("proficiency"), 0.5)
        end = min(1.0, math.log1p(skill.get("endorsements", 0) or 0) / math.log1p(30))
        dur = min(1.0, (skill.get("duration_months", 0) or 0) / 24.0)
        trust = 0.4 * prof + 0.3 * end + 0.3 * dur

        # Check if the skill matches any generic concepts
        is_generic_vector = matches_any(name, GENERIC_VECTOR)
        is_generic_tuning = matches_any(name, GENERIC_TUNING)
        is_generic_llm = matches_any(name, GENERIC_LLM)

        if is_generic_vector:
            if not has_specific_vector and not credited_generic_vector:
                core_score += trust * 2.5
                credited_generic_vector = True
        elif is_generic_tuning:
            if not has_specific_tuning and not credited_generic_tuning:
                core_score += trust * 2.5
                credited_generic_tuning = True
        elif is_generic_llm:
            if not has_specific_llm and not credited_generic_llm:
                core_score += trust * 2.5
                credited_generic_llm = True
        else:
            # Match standard skills
            if matches_any(name, tier1_cleaned):
                core_score += trust * 2.5
            elif matches_any(name, tier2_cleaned):
                bonus_score += trust * 1.0

    return max(0.0, min(14.0, core_score) + min(6.0, bonus_score) - domain_penalty)


def _score_experience(candidate: dict) -> float:
    """10 pts max — 5-9yr sweet spot, penalise job-hopping."""
    yoe: float = candidate["profile"].get("years_of_experience", 0) or 0
    career = candidate.get("career_history", [])
    if 5 <= yoe <= 9:
        base = 10.0
    elif 4 <= yoe < 5 or 9 < yoe <= 11:
        base = 7.0
    elif 3 <= yoe < 4 or 11 < yoe <= 13:
        base = 4.0
    else:
        base = 1.5
    if career:
        longest = max((r.get("duration_months", 0) or 0) for r in career)
        if longest < 12:
            base *= 0.6
    return base


def _score_location(candidate: dict) -> float:
    """10 pts max — India + target cities preferred."""
    location = (candidate["profile"].get("location") or "").lower()
    country = (candidate["profile"].get("country") or "").lower()
    willing = candidate.get("redrob_signals", {}).get("willing_to_relocate", False)

    if country != "india":
        return 1.0 if willing else 0.0
    if any(city in location for city in TARGET_CITIES):
        return 10.0
    return 7.0 if willing else 4.0


def _score_behavioral_additive(sig: dict) -> float:
    """5 pts additive — activity, open-to-work, response rate, notice."""
    try:
        days_inactive = (
            TODAY - datetime.strptime(sig["last_active_date"], "%Y-%m-%d").date()
        ).days
    except Exception:
        days_inactive = 365

    recency = 2.0 if days_inactive <= 14 else 1.5 if days_inactive <= 30 else \
              1.0 if days_inactive <= 90 else 0.3 if days_inactive <= 180 else 0.0

    otw = 1.0 if sig.get("open_to_work_flag", False) else 0.0
    response = (sig.get("recruiter_response_rate", 0) or 0) * 1.0
    notice = sig.get("notice_period_days", 90) or 90
    notice_score = 1.0 if notice <= 30 else 0.7 if notice <= 60 else \
                   0.4 if notice <= 90 else 0.1
    return min(5.0, recency + otw + response + notice_score)


def _behavioral_multiplier(sig: dict) -> float:
    """Availability multiplier 0.2–1.0 applied to final score."""
    try:
        days_inactive = (
            TODAY - datetime.strptime(sig["last_active_date"], "%Y-%m-%d").date()
        ).days
    except Exception:
        days_inactive = 365

    mult = 0.2 if days_inactive > 365 else 0.5 if days_inactive > 180 else \
           0.75 if days_inactive > 90 else 1.0

    if not sig.get("open_to_work_flag", False) and (sig.get("recruiter_response_rate", 0) or 0) < 0.2:
        mult *= 0.7

    return mult


def _rule_score(candidate: dict) -> tuple[float, bool]:
    """
    Combined rule-based score [0, 1].
    Returns (score, is_disqualified_title).
    """
    title = (candidate["profile"].get("current_title") or "").strip()
    sig = candidate.get("redrob_signals", {})

    t = _score_title(title)
    c = _score_career(candidate)
    s = _score_skills(candidate)
    e = _score_experience(candidate)
    loc = _score_location(candidate)
    b = _score_behavioral_additive(sig)
    mult = _behavioral_multiplier(sig)

    raw_100 = t + c + s + e + loc + b
    is_disqualified = t == 0.0
    raw_norm = 0.01 if is_disqualified else (raw_100 / 100.0)
    return min(1.0, max(0.0, raw_norm * mult)), is_disqualified


# ── Final Scoring Formula ─────────────────────────────────────────────────────

def _compute_final_score(
    candidate: dict,
    stage3_score: float,
    stage4_score: float,
    weights,
) -> tuple[float, dict]:
    """
    Final Score = vector_sim × w1 + rule_score × w2 + behavioral × w3 + profile × w4

    All weights from ScoringWeights in settings.py.
    """
    sig = candidate.get("redrob_signals", {})

    # Vector similarity (blended Stage 3 + Stage 4)
    w34 = weights.stage4_vs_stage3
    vector_sim = (1 - w34) * stage3_score + w34 * stage4_score

    # Rule-based score
    rule, is_disq = _rule_score(candidate)

    # Behavioral signals (response rate, GitHub, activity)
    github_raw = sig.get("github_activity_score", -1)
    github = 0.0 if github_raw < 0 else github_raw / 100.0
    try:
        days_inactive = (
            TODAY - datetime.strptime(sig["last_active_date"], "%Y-%m-%d").date()
        ).days
    except Exception:
        days_inactive = 365
    activity = max(0.0, 1.0 - days_inactive / 180.0)
    response = sig.get("recruiter_response_rate", 0) or 0
    interview = sig.get("interview_completion_rate", 0) or 0
    behavioral = 0.30 * response + 0.25 * interview + 0.25 * github + 0.20 * activity

    # Profile quality
    completeness = (sig.get("profile_completeness_score", 0) or 0) / 100.0
    verified = (
        int(sig.get("verified_email", False))
        + int(sig.get("verified_phone", False))
        + int(sig.get("linkedin_connected", False))
    ) / 3.0
    saved_30d = sig.get("saved_by_recruiters_30d", 0) or 0
    recruiter_interest = min(1.0, math.log1p(saved_30d) / math.log1p(10))
    profile_quality = 0.4 * completeness + 0.3 * verified + 0.3 * recruiter_interest

    final = (
        weights.vector_similarity * vector_sim
        + weights.rule_score * rule
        + weights.behavioral * behavioral
        + weights.profile_quality * profile_quality
    )
    if is_disq:
        final *= 0.05  # Near-zero for wrong-domain candidates

    return round(min(1.0, max(0.0, final)), 4), {
        "vector_sim": round(vector_sim, 4),
        "stage3": round(stage3_score, 4),
        "stage4": round(stage4_score, 4),
        "rule": round(rule, 4),
        "behavioral": round(behavioral, 4),
        "profile_quality": round(profile_quality, 4),
        "github": round(github, 4),
        "is_disqualified": is_disq,
    }


# ── Reasoning Generator ───────────────────────────────────────────────────────

def _generate_reasoning(candidate: dict, rank: int, components: dict) -> str:
    """Programmatic reasoning — specific, honest, rank-aware. No LLM."""
    p = candidate["profile"]
    sig = candidate.get("redrob_signals", {})

    title = p.get("current_title", "Professional")
    yoe = p.get("years_of_experience", 0)
    company = p.get("current_company", "")
    location = p.get("location", "")
    country = p.get("country", "")
    notice = sig.get("notice_period_days", 90) or 90
    response_rate = sig.get("recruiter_response_rate", 0) or 0
    otw = sig.get("open_to_work_flag", False)

    try:
        days_inactive = (
            TODAY - datetime.strptime(sig.get("last_active_date", "2000-01-01"), "%Y-%m-%d").date()
        ).days
    except Exception:
        days_inactive = 999

    # Relevant skills
    relevant_skills = [
        s["name"] for s in candidate.get("skills", [])
        if any(kw in s["name"].lower() for kw in list(TIER1_SKILLS)[:20])
    ][:3]
    skills_str = ", ".join(relevant_skills) if relevant_skills else "ML/AI adjacent skills"

    parts = []
    if rank <= 10:
        parts.append(f"{title} with {yoe:.1f} yrs exp at {company}; core skills: {skills_str}")
    elif rank <= 30:
        parts.append(f"{yoe:.0f}-yr {title} at {company} with {skills_str}")
    else:
        parts.append(f"{title} at {company} ({yoe:.0f} yrs); partial JD alignment on {skills_str}")

    if country.lower() == "india":
        city_match = any(city in location.lower() for city in TARGET_CITIES)
        if city_match:
            parts.append(f"based in {location}")
        elif sig.get("willing_to_relocate"):
            parts.append("willing to relocate to target city")
        else:
            parts.append(f"in {location}, not willing to relocate")
    else:
        parts.append(f"located in {country} — outside India")

    if days_inactive > 180:
        parts.append(f"inactive {days_inactive // 30}+ months — availability risk")
    elif not otw and response_rate < 0.3:
        parts.append(f"response rate {response_rate:.0%} — outreach risk")

    if notice <= 30:
        parts.append(f"notice {notice}d")
    elif notice > 90:
        parts.append(f"notice {notice}d — longer than preferred")

    if components.get("github", 0) > 0.5:
        parts.append(f"strong GitHub activity ({components['github']:.0%})")

    if rank > 60:
        parts.append("ranked lower: partial core skill or semantic alignment")

    s1 = parts[0] + (f" ({parts[1]})" if len(parts) > 1 else "") + "."
    if len(parts) > 2:
        s2 = "; ".join(parts[2:]).capitalize() + "."
        return f"{s1} {s2}"
    return s1


# ── Stage 5 Entry Point ───────────────────────────────────────────────────────

def stage5_rerank(
    candidates_with_scores: list[tuple[dict, float, float]],
    weights,
    top_k: int = 100,
    debug: bool = False,
) -> list[dict]:
    """
    Args:
        candidates_with_scores: List of (candidate, stage3_score, stage4_score).
        weights: ScoringWeights instance from settings.py.
        top_k: Final output count (challenge requires 100).
        debug: Print top-10 to console.

    Returns:
        List of dicts ready for write_submission().
    """
    logger.info(f"Stage 5 | Re-ranking {len(candidates_with_scores):,} candidates...")

    scored = []
    honeypots = 0
    for candidate, s3, s4 in candidates_with_scores:
        if _detect_honeypot(candidate):
            honeypots += 1
            final = 0.001
            components = {"honeypot": True}
        else:
            final, components = _compute_final_score(candidate, s3, s4, weights)
        scored.append((candidate, final, components))

    # Sort: descending score (rounded to 4dp), tie-break by candidate_id ascending
    scored.sort(key=lambda x: (-round(x[1], 4), x[0].get("candidate_id", "")))

    if debug:
        logger.info("Stage 5 | TOP 10:")
        for rank, (c, sc, _) in enumerate(scored[:10], 1):
            p = c["profile"]
            logger.info(
                f"  #{rank:3d} | {c['candidate_id']} | "
                f"{p['current_title']:<35} | {p['years_of_experience']:.1f}yr | "
                f"score={sc:.4f}"
            )

    logger.info(
        f"Stage 5 | Honeypots detected: {honeypots} | "
        f"Top score: {scored[0][1]:.4f}"
    )

    rows = []
    for rank, (candidate, score, components) in enumerate(scored[:top_k], start=1):
        reasoning = (
            "Profile flagged: impossible skill/career timeline."
            if components.get("honeypot")
            else _generate_reasoning(candidate, rank, components)
        )
        rows.append({
            "candidate_id": candidate.get("candidate_id", ""),
            "rank": rank,
            "score": score,
            "reasoning": reasoning,
        })

    return rows
