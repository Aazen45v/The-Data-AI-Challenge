"""
honeypot.py — detect subtly impossible profiles.

The challenge embeds ~80 honeypots with internally inconsistent data
(e.g. tenure longer than the company has existed, "expert" in skills with
0 months of use, dates that don't add up). They are forced to relevance
tier 0 in the hidden ground truth, and a submission with >10% honeypots
in its top 100 is disqualified.

We do NOT special-case "find the 80 honeypots". We score *internal
consistency* as a normal part of reading a profile. A profile that fails
several consistency checks gets a strong penalty, so it naturally falls
out of the top ranks. This is robust to honeypots we haven't seen.
"""
from datetime import date

_DATE_FMTS = ("%Y-%m-%d", "%Y-%m", "%Y")


def _parse_date(s):
    if not s:
        return None
    for fmt in _DATE_FMTS:
        try:
            return date.fromisoformat(s) if fmt == "%Y-%m-%d" else None
        except ValueError:
            continue
    # tolerant manual parse
    try:
        parts = str(s).split("-")
        y = int(parts[0])
        m = int(parts[1]) if len(parts) > 1 else 1
        d = int(parts[2]) if len(parts) > 2 else 1
        return date(y, max(1, min(12, m)), max(1, min(28, d)))
    except Exception:
        return None


def _months_between(a, b):
    if not a or not b:
        return None
    return (b.year - a.year) * 12 + (b.month - a.month)


def honeypot_penalty(cand, today=date(2026, 6, 1)):
    """
    Returns (penalty, flags) where penalty in [0, 1].
    penalty >= ~0.6 means 'treat as impossible / exclude from top ranks'.
    """
    flags = []
    score = 0.0

    prof = cand.get("profile", {})
    hist = cand.get("career_history", []) or []
    skills = cand.get("skills", []) or []
    yoe = prof.get("years_of_experience", 0) or 0

    # 1) Date sanity within each role
    total_months = 0
    earliest_start = None
    for job in hist:
        sd = _parse_date(job.get("start_date"))
        ed = _parse_date(job.get("end_date")) or today
        dm_field = job.get("duration_months")
        if sd and earliest_start is None:
            earliest_start = sd
        if sd and ed:
            real = _months_between(sd, ed)
            if real is not None and real < -1:
                score += 0.45
                flags.append("end_before_start")
            # declared duration grossly inconsistent with the dates
            if dm_field is not None and real is not None and real >= 0:
                if abs(dm_field - real) > 18:
                    score += 0.20
                    flags.append("duration_mismatch")
            total_months += max(real or 0, 0)
        elif dm_field:
            total_months += dm_field
        # future-dated start
        if sd and sd > today:
            score += 0.30
            flags.append("future_start")

    # 2) Experience math: declared YOE wildly inconsistent with career history
    derived_years = total_months / 12.0
    if yoe and derived_years:
        if yoe - derived_years > 6:          # claims far more than history shows
            score += 0.30
            flags.append("yoe_exceeds_history")
        if derived_years - yoe > 8:          # history far exceeds claimed
            score += 0.15
            flags.append("history_exceeds_yoe")

    # 3) Tenure exceeds plausible career length (e.g. 8 yrs at a 3-yr-old company)
    #    We can't see company age, but a single role longer than the person's
    #    entire declared experience is impossible.
    for job in hist:
        dm = job.get("duration_months") or 0
        if yoe and dm > (yoe * 12) + 18:
            score += 0.35
            flags.append("single_role_exceeds_career")
            break

    # 4) Skill plausibility: "expert"/"advanced" with 0 months of use
    impossible_skill = 0
    for s in skills:
        prof_lvl = s.get("proficiency")
        dur = s.get("duration_months", None)
        if prof_lvl in ("advanced", "expert") and dur == 0:
            impossible_skill += 1
    if impossible_skill >= 5:
        score += 0.50
        flags.append("expert_skills_zero_months")
    elif impossible_skill >= 3:
        score += 0.25
        flags.append("some_expert_zero_months")

    # 5) Started working before a plausible age (career start vs first degree)
    edu = cand.get("education", []) or []
    grad_years = [e.get("end_year") for e in edu if e.get("end_year")]
    if grad_years and earliest_start:
        first_degree = min(grad_years)
        # starting professional work >3 yrs before finishing first degree is odd
        if earliest_start.year < first_degree - 3:
            score += 0.15
            flags.append("work_before_education")

    return min(score, 1.0), flags
