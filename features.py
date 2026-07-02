"""
features.py — Feature extraction for the Redrob candidate ranking challenge.

Each extractor returns a float in [0, 1] unless otherwise noted.
All functions are pure (no side-effects) so they can be parallelised.
"""

from __future__ import annotations

import math
import re
from datetime import date, datetime
from typing import Any

from ai_skills import (
    AI_INDUSTRIES,
    COMPANY_SIZE_SCORE,
    JD_SUMMARY_KEYWORDS,
    JD_TITLE_KEYWORDS,
    RELEVANT_DEGREES,
    RELEVANT_FIELDS,
    TIER1_SKILLS,
    TIER2_SKILLS,
    TIER3_SKILLS,
    TIER_WEIGHTS,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize(value: float, lo: float, hi: float) -> float:
    """Clamp and linearly normalize value to [0, 1]."""
    if hi <= lo:
        return 0.0
    return max(0.0, min(1.0, (value - lo) / (hi - lo)))


def _today() -> date:
    return datetime.utcnow().date()


def _days_since(date_str: str | None) -> int | None:
    if not date_str:
        return None
    try:
        d = datetime.strptime(date_str[:10], "%Y-%m-%d").date()
        return (_today() - d).days
    except Exception:
        return None


def _keyword_hit_rate(text: str, keywords: list[str]) -> float:
    """Fraction of keywords found in lowercased text."""
    if not text:
        return 0.0
    low = text.lower()
    hits = sum(1 for kw in keywords if kw in low)
    return hits / len(keywords)


# ---------------------------------------------------------------------------
# 1. AI / ML Skill Score  (weight 35%)
# ---------------------------------------------------------------------------

def extract_ai_skill_score(candidate: dict[str, Any]) -> dict[str, float]:
    """
    Returns:
        raw_score          — weighted skill sum (unbounded, used for normalization at batch level)
        core_skill_count   — number of Tier-1 AI skills found
        skill_score        — normalised [0,1]
        trust_multiplier   — anti-stuffing factor based on endorsements + duration
    """
    skills: list[dict] = candidate.get("skills", [])
    proficiency_weights = {"beginner": 0.4, "intermediate": 0.7, "advanced": 0.9, "expert": 1.0}

    tier1_hits = 0
    raw_score = 0.0
    total_endorsements_relevant = 0
    total_duration_relevant = 0
    count_relevant = 0

    for skill in skills:
        name_raw: str = skill.get("name", "") or ""
        name = name_raw.lower().strip()
        proficiency = skill.get("proficiency", "intermediate")
        endorsements: int = skill.get("endorsements", 0) or 0
        duration_months: int = skill.get("duration_months", 0) or 0
        prof_w = proficiency_weights.get(proficiency, 0.5)

        tier = 0
        if name in TIER1_SKILLS:
            tier = 1
        elif name in TIER2_SKILLS:
            tier = 2
        elif name in TIER3_SKILLS:
            tier = 3

        if tier == 0:
            continue

        tier_w = TIER_WEIGHTS[tier]
        # Duration trust: skills used < 3 months get partial credit
        dur_factor = min(1.0, (duration_months + 1) / 6.0)
        # Endorsement factor: log scale 0–1
        end_factor = min(1.0, math.log1p(endorsements) / math.log1p(50))
        # Combined trust
        trust = 0.6 * dur_factor + 0.4 * end_factor

        raw_score += tier_w * prof_w * trust

        if tier == 1:
            tier1_hits += 1
            total_endorsements_relevant += endorsements
            total_duration_relevant += duration_months
            count_relevant += 1

    # Anti-stuffing: if many T1 skills but zero endorsements + zero duration → penalise
    avg_end = total_endorsements_relevant / max(1, count_relevant)
    avg_dur = total_duration_relevant / max(1, count_relevant)
    trust_multiplier = 0.5 + 0.3 * min(1.0, math.log1p(avg_end) / math.log1p(20)) \
                           + 0.2 * min(1.0, avg_dur / 24.0)
    trust_multiplier = max(0.3, min(1.0, trust_multiplier))

    # Normalize raw_score: cap at ~30 (achievable with ~10 T1 expert skills)
    skill_score = min(1.0, raw_score / 30.0)

    return {
        "raw_score": raw_score,
        "core_skill_count": tier1_hits,
        "skill_score": skill_score,
        "trust_multiplier": trust_multiplier,
    }


# ---------------------------------------------------------------------------
# 2. Career Relevance Score  (weight 25%)
# ---------------------------------------------------------------------------

def extract_career_score(candidate: dict[str, Any]) -> dict[str, float]:
    """
    Measures how relevant the candidate's career is to an AI/ML role.
    Checks: current title match, role history, industry alignment, recency.
    """
    profile: dict = candidate.get("profile", {})
    career: list[dict] = candidate.get("career_history", [])

    current_title: str = (profile.get("current_title") or "").lower()
    current_industry: str = (profile.get("current_industry") or "").lower()

    # Current title relevance
    title_score = 0.0
    for kw in JD_TITLE_KEYWORDS:
        if kw in current_title:
            title_score = 1.0
            break
    if title_score == 0.0:
        # Partial match: data/engineer/scientist with some AI exposure
        partial_keywords = ["data", "engineer", "scientist", "analyst", "developer", "researcher"]
        for kw in partial_keywords:
            if kw in current_title:
                title_score = 0.4
                break

    # Industry alignment
    industry_score = 0.0
    for ai_ind in AI_INDUSTRIES:
        if ai_ind in current_industry:
            industry_score = 1.0
            break
    if industry_score == 0.0:
        industry_score = 0.3  # baseline for being in any tech-adjacent field

    # Career history relevance — reward AI/ML work in past roles
    history_score = 0.0
    history_total_weight = 0.0
    for i, role in enumerate(career[:6]):  # look at most recent 6 roles
        recency_weight = 1.0 / (i + 1)  # more recent = higher weight
        desc: str = (role.get("description") or "").lower()
        rtitle: str = (role.get("title") or "").lower()
        rindustry: str = (role.get("industry") or "").lower()

        role_score = 0.0
        # Title match
        for kw in JD_TITLE_KEYWORDS:
            if kw in rtitle:
                role_score += 0.5
                break
        # Description keyword hits
        role_score += 0.5 * _keyword_hit_rate(desc, JD_SUMMARY_KEYWORDS)
        # Industry match
        for ai_ind in AI_INDUSTRIES:
            if ai_ind in rindustry:
                role_score += 0.3
                break
        role_score = min(1.0, role_score)

        history_score += recency_weight * role_score
        history_total_weight += recency_weight

    history_score = history_score / max(1.0, history_total_weight)

    # Company size progression (signals career growth)
    sizes = [r.get("company_size", "51-200") for r in career]
    size_scores = [COMPANY_SIZE_SCORE.get(s, 0.5) for s in sizes]
    avg_company_score = sum(size_scores) / len(size_scores) if size_scores else 0.5

    career_score = (
        0.35 * title_score
        + 0.30 * history_score
        + 0.20 * industry_score
        + 0.15 * avg_company_score
    )

    return {
        "title_score": title_score,
        "industry_score": industry_score,
        "history_score": history_score,
        "career_score": min(1.0, career_score),
    }


# ---------------------------------------------------------------------------
# 3. Experience Quality Score  (weight 15%)
# ---------------------------------------------------------------------------

def extract_experience_score(candidate: dict[str, Any]) -> dict[str, float]:
    """
    Scores years of experience, tenure stability, and role progression.
    Sweet spot for AI/ML roles: 3–10 years.
    """
    profile: dict = candidate.get("profile", {})
    yoe: float = profile.get("years_of_experience", 0) or 0

    # Bell-curve scoring: 3-10 yrs is ideal, <2 and >15 get lower scores
    if yoe < 1:
        yoe_score = 0.1
    elif yoe < 2:
        yoe_score = 0.3
    elif yoe <= 5:
        yoe_score = 0.7 + 0.06 * (yoe - 2)  # peaks at ~0.88
    elif yoe <= 10:
        yoe_score = 0.9 + 0.02 * (yoe - 5)  # plateaus near 1.0
    elif yoe <= 15:
        yoe_score = 1.0 - 0.03 * (yoe - 10)  # slight decline
    else:
        yoe_score = max(0.5, 1.0 - 0.05 * (yoe - 15))

    yoe_score = max(0.0, min(1.0, yoe_score))

    # Tenure stability (reward long tenures, penalise job-hopping)
    career: list[dict] = candidate.get("career_history", [])
    durations = [r.get("duration_months", 0) or 0 for r in career]
    avg_tenure = sum(durations) / len(durations) if durations else 0
    # 18+ months average = stable
    tenure_score = min(1.0, avg_tenure / 24.0)

    # Certifications bonus
    certs: list[dict] = candidate.get("certifications", [])
    recent_year = _today().year
    cert_score = 0.0
    for cert in certs:
        yr = cert.get("year", 0) or 0
        if yr >= recent_year - 3:  # certs in last 3 years
            cert_score = min(1.0, cert_score + 0.2)

    experience_score = 0.50 * yoe_score + 0.30 * tenure_score + 0.20 * cert_score
    return {
        "yoe_score": yoe_score,
        "tenure_score": tenure_score,
        "cert_score": cert_score,
        "experience_score": min(1.0, experience_score),
    }


# ---------------------------------------------------------------------------
# 4. Behavioral Signals Score  (weight 15%)
# ---------------------------------------------------------------------------

def extract_behavioral_score(candidate: dict[str, Any]) -> dict[str, float]:
    """
    Uses redrob_signals to measure engagement, reliability, and readiness.
    """
    sig: dict = candidate.get("redrob_signals", {})

    # Activity recency
    days_inactive = _days_since(sig.get("last_active_date")) or 365
    activity_score = max(0.0, 1.0 - days_inactive / 90.0)  # decay over 90 days

    # Open to work
    otw_score = 1.0 if sig.get("open_to_work_flag", False) else 0.3

    # Response reliability
    response_rate: float = sig.get("recruiter_response_rate", 0) or 0
    interview_rate: float = sig.get("interview_completion_rate", 0) or 0
    reliability_score = 0.6 * response_rate + 0.4 * interview_rate

    # GitHub activity (AI/ML engineers should have GitHub)
    github: float = sig.get("github_activity_score", -1)
    github_score = 0.0 if github < 0 else github / 100.0

    # Profile quality
    completeness: float = sig.get("profile_completeness_score", 0) or 0
    profile_score = completeness / 100.0

    # Recruiter interest signals
    saved_30d: int = sig.get("saved_by_recruiters_30d", 0) or 0
    views_30d: int = sig.get("profile_views_received_30d", 0) or 0
    interest_score = min(1.0, (math.log1p(saved_30d) + 0.5 * math.log1p(views_30d))
                         / (math.log1p(20) + 0.5 * math.log1p(100)))

    # Notice period (lower is better — faster to join)
    notice: int = sig.get("notice_period_days", 60) or 60
    notice_score = max(0.0, 1.0 - notice / 90.0)

    # Verifications
    verified = int(sig.get("verified_email", False)) + int(sig.get("verified_phone", False)) \
               + int(sig.get("linkedin_connected", False))
    verified_score = verified / 3.0

    behavioral_score = (
        0.20 * activity_score
        + 0.20 * reliability_score
        + 0.15 * github_score
        + 0.15 * profile_score
        + 0.10 * interest_score
        + 0.10 * otw_score
        + 0.05 * notice_score
        + 0.05 * verified_score
    )

    return {
        "activity_score": activity_score,
        "otw_score": otw_score,
        "reliability_score": reliability_score,
        "github_score": github_score,
        "profile_score": profile_score,
        "interest_score": interest_score,
        "behavioral_score": min(1.0, behavioral_score),
    }


# ---------------------------------------------------------------------------
# 5. Education Score  (weight 10%)
# ---------------------------------------------------------------------------

def extract_education_score(candidate: dict[str, Any]) -> dict[str, float]:
    """
    Scores institution tier, degree relevance, and field of study.
    """
    education: list[dict] = candidate.get("education", [])

    if not education:
        return {"tier_score": 0.3, "field_score": 0.3, "education_score": 0.3}

    tier_map = {"tier_1": 1.0, "tier_2": 0.8, "tier_3": 0.6, "tier_4": 0.4, "unknown": 0.5}

    best_tier = 0.0
    best_field = 0.0
    best_degree = 0.0

    for edu in education:
        tier_str: str = (edu.get("tier") or "unknown").lower()
        t = tier_map.get(tier_str, 0.5)
        if t > best_tier:
            best_tier = t

        field: str = (edu.get("field_of_study") or "").lower()
        for rf in RELEVANT_FIELDS:
            if rf in field:
                best_field = 1.0
                break
        if best_field == 0.0:
            best_field = max(best_field, 0.4)

        degree: str = (edu.get("degree") or "").lower()
        for rd in RELEVANT_DEGREES:
            if rd in degree:
                if "ph" in rd or "doctor" in rd:
                    best_degree = 1.0
                elif rd.startswith("m"):
                    best_degree = max(best_degree, 0.85)
                else:
                    best_degree = max(best_degree, 0.70)
                break

    education_score = 0.40 * best_tier + 0.35 * best_field + 0.25 * best_degree
    return {
        "tier_score": best_tier,
        "field_score": best_field,
        "degree_score": best_degree,
        "education_score": min(1.0, education_score),
    }


# ---------------------------------------------------------------------------
# 6. Summary / Composite Feature Vector
# ---------------------------------------------------------------------------

COMPONENT_WEIGHTS = {
    "skill":        0.35,
    "career":       0.25,
    "experience":   0.15,
    "behavioral":   0.15,
    "education":    0.10,
}


def build_feature_vector(candidate: dict[str, Any]) -> dict[str, Any]:
    """
    Builds the complete feature vector for a candidate.
    Returns a dict with individual scores + composite final_score.
    """
    skill_feats    = extract_ai_skill_score(candidate)
    career_feats   = extract_career_score(candidate)
    exp_feats      = extract_experience_score(candidate)
    behav_feats    = extract_behavioral_score(candidate)
    edu_feats      = extract_education_score(candidate)

    # Weighted composite (0–1 scale)
    base_score = (
        COMPONENT_WEIGHTS["skill"]      * skill_feats["skill_score"]
        + COMPONENT_WEIGHTS["career"]   * career_feats["career_score"]
        + COMPONENT_WEIGHTS["experience"] * exp_feats["experience_score"]
        + COMPONENT_WEIGHTS["behavioral"] * behav_feats["behavioral_score"]
        + COMPONENT_WEIGHTS["education"]  * edu_feats["education_score"]
    )

    # Apply trust multiplier from skills (anti-stuffing)
    final_score = base_score * skill_feats["trust_multiplier"]
    final_score = min(1.0, max(0.0, final_score))

    return {
        "candidate_id": candidate.get("candidate_id", ""),
        # Component scores
        "skill_score":      round(skill_feats["skill_score"], 4),
        "core_skill_count": skill_feats["core_skill_count"],
        "trust_multiplier": round(skill_feats["trust_multiplier"], 4),
        "career_score":     round(career_feats["career_score"], 4),
        "title_score":      round(career_feats["title_score"], 4),
        "experience_score": round(exp_feats["experience_score"], 4),
        "behavioral_score": round(behav_feats["behavioral_score"], 4),
        "github_score":     round(behav_feats["github_score"], 4),
        "education_score":  round(edu_feats["education_score"], 4),
        # Final
        "base_score":   round(base_score, 4),
        "final_score":  round(final_score, 4),
        # Pass-through for reasoning
        "_profile": candidate.get("profile", {}),
        "_sig":     candidate.get("redrob_signals", {}),
    }
