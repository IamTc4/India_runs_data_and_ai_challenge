"""
stage1_filter.py — Hard filtering using structured metadata fields.

Removes candidates who are structurally ineligible with zero ML cost.
Only truly hard rules: experience floor, 2-year inactivity.
Wrong-domain titles handled via score_title() to preserve 100-row output guarantee.
"""

from datetime import date, datetime
from src.utils.logger import logger


def stage1_filter(candidates: list[dict], min_exp: float, max_inactive_days: int) -> list[dict]:
    """
    Args:
        candidates: Full list of loaded candidate dicts.
        min_exp: Minimum years of experience to pass (from PipelineConfig).
        max_inactive_days: Maximum days since last activity (from PipelineConfig).

    Returns:
        Filtered list of candidates.
    """
    today = date.today()
    passed = []
    dropped_exp = dropped_inactive = 0

    for c in candidates:
        profile = c.get("profile", {})
        sig = c.get("redrob_signals", {})
        yoe: float = profile.get("years_of_experience", 0) or 0

        # Rule 1: Absolute experience floor
        if yoe < min_exp:
            dropped_exp += 1
            continue

        # Rule 2: Hard inactivity — 2+ years inactive = not contactable
        last_active_str = sig.get("last_active_date")
        if last_active_str:
            try:
                last_active = datetime.strptime(last_active_str[:10], "%Y-%m-%d").date()
                if (today - last_active).days > max_inactive_days:
                    dropped_inactive += 1
                    continue
            except ValueError:
                pass  # Malformed date — let through, handled in scoring

        passed.append(c)

    logger.info(
        f"Stage 1 | Passed {len(passed):,} | "
        f"Dropped: exp<{min_exp}yr={dropped_exp:,}  inactive={dropped_inactive:,}"
    )
    return passed
