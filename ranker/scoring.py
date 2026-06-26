"""
scoring.py — transparent, component-based fit scoring.

Philosophy: read the profile the way a recruiter would. The decisive signals
are NOT keyword counts. They are:
  - What the person ACTUALLY did (titles + career-history descriptions)
  - Whether requirement evidence lives in real work vs a padded skills list
  - Disqualifiers the JD names explicitly (research-only, consulting-only,
    title-chasing, no recent code, CV/speech-only)
  - Behavioral availability (a perfect-on-paper but inactive candidate is,
    for hiring purposes, not actually available)

Every component returns a 0..1 score plus structured evidence used later to
write honest, candidate-specific reasoning.
"""
import re
from datetime import date

_WORD = re.compile(r"[a-z0-9\+\#\.\-/]+")


def _norm(s):
    return (s or "").lower()


def _count_terms(text, terms):
    """Number of distinct rubric terms that appear in text."""
    t = _norm(text)
    hits = []
    for term in terms:
        if term in t:
            hits.append(term)
    return hits


def candidate_text(cand):
    """Concatenated profile text, but we keep career descriptions separate
    because evidence in real work history is worth far more than the skills list."""
    prof = cand.get("profile", {})
    parts = [prof.get("headline", ""), prof.get("summary", ""),
             prof.get("current_title", ""), prof.get("current_industry", "")]
    desc_parts = []
    title_parts = []
    for job in cand.get("career_history", []) or []:
        title_parts.append(job.get("title", ""))
        desc_parts.append(job.get("description", ""))
        desc_parts.append(job.get("industry", ""))
    return {
        "profile": _norm(" ".join(parts)),
        "descriptions": _norm(" ".join(desc_parts)),
        "titles": _norm(" ".join(title_parts + [prof.get("current_title", "")])),
        "skills": _norm(" ".join(s.get("name", "") for s in cand.get("skills", []) or [])),
    }


# ---------------------------------------------------------------- components

def score_role_relevance(cand, txt, R):
    """The single most important anti-keyword-stuffer signal.
    A 'Marketing Manager' with every AI keyword is NOT a fit; an engineer
    whose history is building systems is."""
    rr = R["role_relevance"]
    titles = txt["titles"]
    cur = _norm(cand.get("profile", {}).get("current_title", ""))

    strong = any(p in titles for p in rr["strong_positive"])
    adj = any(p in titles for p in rr["adjacent_positive"])
    neg_cur = any(n in cur for n in rr["negative"])
    neg_any = any(n in titles for n in rr["negative"])

    if strong and not neg_cur:
        s = 1.0
    elif adj and not neg_cur:
        s = 0.72
    elif neg_cur:
        s = 0.08                      # current role is off-domain → hard down-weight
    elif neg_any:
        s = 0.30
    else:
        s = 0.45
    return s, {"strong_title": strong, "adjacent_title": adj, "offdomain_current": neg_cur}


def score_core_requirements(cand, txt, R):
    """Evidence for the must-haves, weighted so that proof in real career
    descriptions counts ~3x a bare skills-list keyword."""
    total_w = sum(c["weight"] for c in R["must_have_concepts"])
    earned = 0.0
    matched = []
    for c in R["must_have_concepts"]:
        in_desc = _count_terms(txt["descriptions"], c["terms"])
        in_prof = _count_terms(txt["profile"], c["terms"])
        in_skill = _count_terms(txt["skills"], c["terms"])
        # evidence strength: career description >> summary/headline >> skill list
        strength = 0.0
        if in_desc:
            strength = 1.0
        elif in_prof:
            strength = 0.6
        elif in_skill:
            strength = 0.25
        earned += c["weight"] * strength
        if strength > 0:
            matched.append((c["name"], round(strength, 2)))
    score = earned / total_w if total_w else 0.0
    return min(score, 1.0), {"concepts": matched}


def score_skill_trust(cand, R):
    """Catches lazy keyword stuffing: weight skills by proficiency x evidence
    (duration used, endorsements, on-platform assessment). 'Expert' with 0
    months and 0 endorsements is near-worthless."""
    skills = cand.get("skills", []) or []
    if not skills:
        return 0.4, {"trusted_ai_skills": 0}
    sig = cand.get("redrob_signals", {}) or {}
    assess = sig.get("skill_assessment_scores", {}) or {}
    prof_w = {"beginner": 0.3, "intermediate": 0.6, "advanced": 0.85, "expert": 1.0}
    trusted = 0.0
    total = 0.0
    for s in skills:
        base = prof_w.get(s.get("proficiency"), 0.5)
        dur = s.get("duration_months", 0) or 0
        end = s.get("endorsements", 0) or 0
        # trust multiplier: real use + social proof + verified assessment
        evidence = 0.0
        evidence += min(dur / 24.0, 1.0) * 0.6
        evidence += min(end / 20.0, 1.0) * 0.25
        a = assess.get(s.get("name", ""), None)
        if a is not None:
            evidence += min(a / 100.0, 1.0) * 0.15
        else:
            evidence += 0.05
        trusted += base * evidence
        total += base
    ratio = trusted / total if total else 0.0
    return min(ratio * 1.1, 1.0), {"skill_trust_ratio": round(ratio, 2)}


def _parse_year(s):
    try:
        return int(str(s).split("-")[0])
    except Exception:
        return None


def score_experience_fit(cand, R, today=date(2026, 6, 1)):
    band = R["experience_band"]
    yoe = cand.get("profile", {}).get("years_of_experience", 0) or 0
    lo, hi = band["min"], band["max"]
    slo, shi = band["soft_min"], band["soft_max"]
    if lo <= yoe <= hi:
        s = 1.0
    elif slo <= yoe < lo:
        s = 0.8
    elif hi < yoe <= shi:
        s = 0.75
    elif yoe < slo:
        s = max(0.2, 0.8 - (slo - yoe) * 0.18)
    else:
        s = max(0.25, 0.75 - (yoe - shi) * 0.06)   # very senior: role writes code
    return s, {"yoe": yoe}


def score_trajectory(cand, txt, R):
    """Product-company bias, anti-title-chasing, anti-consulting-only,
    anti-no-recent-code, plus the JD's hard disqualifiers."""
    hist = cand.get("career_history", []) or []
    flags = []
    s = 0.6

    companies = [_norm(j.get("company", "")) for j in hist]
    all_company_text = " ".join(companies)
    consulting_hits = sum(1 for f in R["consulting_firms"] if f in all_company_text)

    # consulting-only entire career
    if hist and consulting_hits >= max(1, len(hist)) and consulting_hits == len(hist):
        s -= 0.35
        flags.append("consulting_only")
    elif consulting_hits:
        flags.append("some_consulting")

    # title-chasing: many short stints
    short_stints = sum(1 for j in hist if (j.get("duration_months") or 0) < 18)
    if len(hist) >= 4 and short_stints >= 3:
        s -= 0.2
        flags.append("title_chaser")

    # research-only with no production signal
    is_research = any(m in txt["titles"] or m in txt["profile"] for m in R["research_only_markers"])
    has_prod = any(m in txt["descriptions"] or m in txt["profile"] for m in R["production_markers"])
    if is_research and not has_prod:
        s -= 0.3
        flags.append("research_only_no_prod")
    elif has_prod:
        s += 0.18
        flags.append("production_signal")

    # no recent code: long tenure in pure lead/architecture role, current title
    cur_title = _norm(cand.get("profile", {}).get("current_title", ""))
    if any(k in cur_title for k in ["architect", "tech lead", "engineering manager", "vp ", "director"]) \
       and not has_prod:
        s -= 0.15
        flags.append("possibly_no_recent_code")

    # CV/speech/robotics without NLP/IR
    cv = any(m in txt["descriptions"] or m in txt["profile"] for m in R["cv_speech_robotics"])
    ir = any(m in txt["descriptions"] or m in txt["profile"] for m in R["nlp_ir_markers"])
    if cv and not ir:
        s -= 0.2
        flags.append("cv_speech_only")

    # product company bonus (non-services industry in recent role)
    recent_inds = " ".join(_norm(j.get("industry", "")) for j in hist[:2])
    if recent_inds and "it services" not in recent_inds and "consulting" not in recent_inds:
        s += 0.1
        flags.append("product_company")

    return max(0.0, min(s, 1.0)), {"trajectory_flags": flags}


def score_semantic(cand, txt, R):
    """Lightweight semantic layer for plain-language Tier-5s who describe
    building ranking/recsys/search WITHOUT the buzzwords. We reward concept
    families appearing in real descriptions even if the exact skill keyword
    is absent. Pure-Python, single-pass, deterministic."""
    concept_families = [
        ["recommend", "recommendation", "recommender", "personaliz", "personalis"],
        ["search", "relevance", "ranking", "rank ", "ranked", "match", "matching"],
        ["retriev", "embedding", "similarity", "nearest neighbor", "semantic"],
        ["experiment", "a/b", "metric", "evaluat", "ndcg", "offline", "online test"],
        ["pipeline", "production", "deployed", "real users", "at scale", "serving"],
    ]
    desc = txt["descriptions"] + " " + txt["profile"]
    fired = 0
    for fam in concept_families:
        if any(term in desc for term in fam):
            fired += 1
    return fired / len(concept_families), {"semantic_families": fired}


def score_location(cand, R):
    prof = cand.get("profile", {})
    loc = _norm(prof.get("location", "") + " " + prof.get("country", ""))
    sig = cand.get("redrob_signals", {}) or {}
    relocate = sig.get("willing_to_relocate", False)
    if any(p in loc for p in R["location"]["preferred"]):
        return 1.0, {"location": "preferred_hub"}
    if any(t in loc for t in R["location"]["india_terms"]):
        return 0.8, {"location": "india_other"}
    if relocate:
        return 0.55, {"location": "relocatable"}
    return 0.2, {"location": "outside_no_relocate"}


# ---------------------------------------------------------------- behavioral

def behavioral_modifier(cand, R, today=date(2026, 6, 1)):
    """Multiplicative availability modifier. Down-weights perfect-on-paper but
    unreachable candidates; gently rewards engaged, verified, available ones."""
    sig = cand.get("redrob_signals", {}) or {}
    b = R["behavioral"]

    resp = sig.get("recruiter_response_rate", 0.0) or 0.0

    # recency
    la = sig.get("last_active_date")
    recency = 0.5
    try:
        y, m, d = [int(x) for x in str(la).split("-")[:3]]
        days = (today - date(y, m, d)).days
        if days <= 14:
            recency = 1.0
        elif days <= 45:
            recency = 0.8
        elif days <= 120:
            recency = 0.5
        else:
            recency = 0.15
    except Exception:
        pass

    otw = 1.0 if sig.get("open_to_work_flag") else 0.4
    icr = sig.get("interview_completion_rate", 0.5)
    icr = 0.5 if icr is None else icr
    saved = min((sig.get("saved_by_recruiters_30d", 0) or 0) / 10.0, 1.0)
    verified = (int(bool(sig.get("verified_email"))) + int(bool(sig.get("verified_phone")))
                + int(bool(sig.get("linkedin_connected")))) / 3.0
    notice = sig.get("notice_period_days", 90)
    notice = 90 if notice is None else notice
    notice_s = 1.0 if notice <= 30 else (0.7 if notice <= 60 else 0.45)

    raw = (b["response_rate_w"] * resp
           + b["recency_w"] * recency
           + b["open_to_work_w"] * otw
           + b["interview_completion_w"] * icr
           + b["saved_by_recruiters_w"] * saved
           + b["verified_w"] * verified
           + b["notice_period_w"] * notice_s)
    # map raw (0..1-ish) into [floor, ceil]
    mod = b["floor"] + (b["ceil"] - b["floor"]) * raw
    return max(b["floor"], min(b["ceil"], mod)), {
        "response_rate": round(resp, 2),
        "recency": recency,
        "open_to_work": bool(sig.get("open_to_work_flag")),
        "notice_days": notice,
    }


# ---------------------------------------------------------------- aggregate

COMPONENT_FNS = [
    ("role_relevance", lambda c, t, R: score_role_relevance(c, t, R)),
    ("core_requirements", lambda c, t, R: score_core_requirements(c, t, R)),
    ("semantic", lambda c, t, R: score_semantic(c, t, R)),
    ("skill_trust", lambda c, t, R: score_skill_trust(c, R)),
    ("experience_fit", lambda c, t, R: score_experience_fit(c, R)),
    ("trajectory", lambda c, t, R: score_trajectory(c, t, R)),
    ("location", lambda c, t, R: score_location(c, R)),
]


def ideal_profile_bonus(cand, comps, evidence):
    """The JD's 'how to read between the lines' paragraph describes one very
    specific ideal: ~6-8 yrs, 4-5 in applied ML at PRODUCT (not services)
    companies, has SHIPPED an end-to-end ranking/search/recsys system, located
    in / willing to relocate to an India hub, and ACTIVE on-platform so they can
    actually be reached. We give a small, capped additive bonus to candidates
    who satisfy this full conjunction so the very top of the list (NDCG@10) is
    the strongest possible. It only lifts candidates who already score well, so
    it cannot rescue a stuffer (fails role_relevance) or a honeypot (collapsed).
    """
    parts = []
    # strong core ML/eng role (not off-domain)
    parts.append(1.0 if comps.get("role_relevance", 0) >= 0.9 else 0.0)
    # shipped ranking / search / recsys with first-hand evidence
    concepts = dict(evidence.get("core_requirements", {}).get("concepts", []))
    shipped = max(concepts.get("ranking_recsys", 0),
                  concepts.get("embeddings_retrieval", 0),
                  concepts.get("vector_search_infra", 0))
    parts.append(1.0 if shipped >= 1.0 else (0.4 if shipped > 0 else 0.0))
    # product-company background
    tf = evidence.get("trajectory", {}).get("trajectory_flags", [])
    parts.append(1.0 if "product_company" in tf else 0.0)
    parts.append(1.0 if ("production_signal" in tf or "product_company" in tf) else 0.0)
    # in the tight 6-8 ideal band
    yoe = evidence.get("experience_fit", {}).get("yoe", 0) or 0
    parts.append(1.0 if 5.5 <= yoe <= 8.5 else (0.5 if 4 <= yoe <= 10 else 0.0))
    # India hub / relocatable
    loc = evidence.get("location", {}).get("location", "")
    parts.append(1.0 if loc in ("preferred_hub", "india_other") else (0.5 if loc == "relocatable" else 0.0))
    # actually reachable: recent + responsive
    bev = evidence.get("behavioral", {})
    reachable = (bev.get("recency", 0) >= 0.8) and (bev.get("response_rate", 0) >= 0.5)
    parts.append(1.0 if reachable else (0.4 if bev.get("recency", 0) >= 0.5 else 0.0))

    frac = sum(parts) / len(parts)
    # only a strong, near-complete match earns meaningful lift
    return 0.16 * (frac ** 2)


def normalize_display(sorted_desc_raw):
    """Map raw scores (already sorted descending) into a clean, non-increasing
    [0.05, 0.99] display column. Shared by rank.py and the sandbox so they agree
    and no score ever exceeds 1.0."""
    if not sorted_desc_raw:
        return []
    hi = max(sorted_desc_raw)
    lo = min(sorted_desc_raw)
    span = (hi - lo) or 1.0
    out = []
    prev = 1.0
    for s in sorted_desc_raw:
        norm = 0.05 + 0.94 * ((s - lo) / span)
        norm = min(norm, prev)
        prev = norm
        out.append(round(norm, 4))
    return out


def score_candidate(cand, R, honeypot_pen, honeypot_flags):
    txt = candidate_text(cand)
    W = R["weights"]
    comps = {}
    evidence = {}
    base = 0.0
    for name, fn in COMPONENT_FNS:
        s, ev = fn(cand, txt, R)
        comps[name] = s
        evidence[name] = ev
        base += W[name] * s

    mod, bev = behavioral_modifier(cand, R)
    evidence["behavioral"] = bev

    base = min(base + ideal_profile_bonus(cand, comps, evidence), 1.05)
    final = base * mod

    # honeypot / impossible-profile gate: collapse the score so it cannot
    # surface in the top ranks. Not a special-case lookup — a consistency tax.
    if honeypot_pen >= 0.55:
        final *= (1.0 - honeypot_pen) * 0.15
        evidence["honeypot"] = honeypot_flags
    elif honeypot_pen > 0:
        final *= (1.0 - 0.4 * honeypot_pen)
        evidence["honeypot"] = honeypot_flags

    return final, comps, evidence
