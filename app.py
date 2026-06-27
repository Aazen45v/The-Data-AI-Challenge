#!/usr/bin/env python3
"""
app.py - hosted sandbox demo (submission_spec Section 10.5).

Runs the SAME ranking pipeline rank.py uses, then surfaces the FULL picture for
each candidate: the seven component scores, the behavioral-availability modifier,
platform activity, career history, skills, requirement evidence found, and
honeypot/consistency flags - so a recruiter can see exactly why each person is
ranked where they are.

Deploy on HuggingFace Spaces / Streamlit Cloud. Run locally: streamlit run app.py
"""
import glob
import io
import json
import os
from datetime import date

import streamlit as st

from ranker.honeypot import honeypot_penalty
from ranker.scoring import (
    score_candidate, behavioral_modifier, normalize_display,
)
from ranker.reasoning import build_reasoning

st.set_page_config(page_title="Redrob Candidate Ranker", layout="wide", initial_sidebar_state="collapsed")

TODAY = date(2026, 6, 1)  # same fixed "today" the scoring/honeypot modules use

ACCENT = "#1A7A4E"
LOW = "#B0552F"
WARN_FG, WARN_BG = "#7A5E0E", "#F3ECD6"
GO_FG, GO_BG = "#115C3A", "#E4F0E7"
FLAG_FG, FLAG_BG = LOW, "#F4E6DD"

COMPONENT_LABELS = {
    "role_relevance": "Role relevance",
    "core_requirements": "Requirement evidence",
    "semantic": "Semantic match",
    "skill_trust": "Skill trust",
    "experience_fit": "Experience fit",
    "trajectory": "Trajectory",
    "location": "Location",
}
CONCEPT_LABELS = {
    "embeddings_retrieval": "embeddings/retrieval",
    "vector_search_infra": "vector search infra",
    "ranking_recsys": "ranking/recsys",
    "eval_frameworks": "ranking evaluation",
    "modern_ml_llm": "LLM/modern ML",
    "python_production": "production Python",
}
HONEYPOT_FLAG_LABELS = {
    "end_before_start": "a role's end date is before its start date",
    "duration_mismatch": "declared tenure doesn't match the start/end dates",
    "future_start": "a role starts in the future",
    "yoe_exceeds_history": "declared experience far exceeds career history",
    "history_exceeds_yoe": "career history far exceeds declared experience",
    "single_role_exceeds_career": "one role's tenure exceeds the entire declared career",
    "expert_skills_zero_months": "5+ ‘expert’/‘advanced’ skills with 0 months of use",
    "some_expert_zero_months": "several ‘expert’/‘advanced’ skills with 0 months of use",
    "work_before_education": "work history appears to start before finishing their first degree",
}
TRAJECTORY_FLAG_BADGES = {
    "consulting_only": ("Consulting-only career", "flag"),
    "some_consulting": ("Mostly consulting", "warn"),
    "title_chaser": ("Title-chaser pattern", "flag"),
    "research_only_no_prod": ("Research-only, no prod", "flag"),
    "cv_speech_only": ("CV/speech, not NLP/IR", "flag"),
    "product_company": ("Product company", "go"),
    "production_signal": ("Shipped to production", "go"),
}

# ---------------------------------------------------------------- helpers

def _days_since(date_str):
    try:
        y, m, d = [int(x) for x in str(date_str).split("-")[:3]]
        return (TODAY - date(y, m, d)).days
    except Exception:
        return None


def _val_color(v):
    return LOW if v < 0.4 else ACCENT


def _fit_color(v):
    if v >= 0.7:
        return ACCENT
    if v >= 0.4:
        return "#17150F"
    return LOW


def derive_flags(x):
    """Real, evidence-grounded badges for a candidate row (capped at 3)."""
    out = []
    pen, mod, comps, ev = x["pen"], x["mod"], x["comps"], x["ev"]
    traj_flags = ev.get("trajectory", {}).get("trajectory_flags", [])
    rr_ev = ev.get("role_relevance", {})
    sig = x["cand"].get("redrob_signals", {}) or {}
    resp = sig.get("recruiter_response_rate", 0) or 0
    otw = sig.get("open_to_work_flag", False)
    days = _days_since(sig.get("last_active_date"))

    if pen >= 0.55:
        out.append(("Honeypot · collapsed", "flag"))
    elif pen > 0:
        out.append(("Consistency flag", "warn"))

    for code in ("consulting_only", "title_chaser", "research_only_no_prod", "cv_speech_only"):
        if code in traj_flags:
            out.append(TRAJECTORY_FLAG_BADGES[code])

    if rr_ev.get("offdomain_current"):
        out.append(("Off-domain role", "flag"))

    if mod < 0.65:
        out.append(("Low availability", "flag"))
    elif resp and resp < 0.15:
        out.append((f"{resp:.0%} response rate", "warn"))
    elif not otw:
        out.append(("Not open-to-work", "warn"))

    if x["strong"]:
        out.append(("Strong fit", "go"))
    elif rr_ev.get("strong_title") and not rr_ev.get("offdomain_current"):
        out.append(("Core title match", "go"))
    elif rr_ev.get("adjacent_title") and not rr_ev.get("offdomain_current"):
        out.append(("Plain-language fit", "go"))

    concepts = ev.get("core_requirements", {}).get("concepts", [])
    if concepts and all(s < 1.0 for _, s in concepts):
        out.append(("Evidence is thin", "warn"))

    skt = ev.get("skill_trust", {}).get("skill_trust_ratio")
    if skt is not None and skt < 0.3 and len(x["cand"].get("skills") or []) >= 4:
        out.append(("Keyword-stuffer", "flag"))

    if days is not None and days <= 14 and not any(k == "flag" for _, k in out):
        out.append((f"Active · {days}d ago", "go"))

    seen, uniq = set(), []
    for lab, kind in out:
        if lab not in seen:
            uniq.append((lab, kind))
            seen.add(lab)
    priority = {"flag": 0, "warn": 1, "go": 2}
    uniq.sort(key=lambda t: priority[t[1]])
    return uniq[:3]


def fit_label(x):
    if x["pen"] >= 0.55:
        return "Honeypot"
    if x["mod"] < 0.6:
        return "Unreachable"
    fit = x["fit"]
    if fit >= 0.75 and x["strong"]:
        return "Strong fit"
    if fit >= 0.55:
        return "Moderate"
    if x["comps"].get("role_relevance", 0) < 0.5:
        return "Off-target"
    if fit < 0.15:
        return "Disqualified"
    return "Weak fit"


def banner_for(x):
    pen, mod, flags = x["pen"], x["mod"], x["flags"]
    if pen >= 0.55:
        labels = [HONEYPOT_FLAG_LABELS.get(f, f) for f in flags]
        return ("error", "Impossible / honeypot profile",
                "Internal consistency checks failed: " + "; ".join(labels) + ". Score collapsed.")
    if mod < 0.65:
        sig = x["cand"].get("redrob_signals", {}) or {}
        bits = []
        days = _days_since(sig.get("last_active_date"))
        if days is not None and days > 90:
            bits.append(f"inactive {days}d")
        resp = sig.get("recruiter_response_rate", 0) or 0
        if resp < 0.2:
            bits.append(f"{resp:.0%} recruiter-response rate")
        if not sig.get("open_to_work_flag"):
            bits.append("not open to work")
        notice = sig.get("notice_period_days")
        if notice and notice > 60:
            bits.append(f"{notice}-day notice")
        text = ", ".join(bits) or "low behavioral availability"
        return ("warn", "Availability collapsed",
                f"{text}. Strong on paper, but down-weighted ×{mod:.2f} for hiring purposes.")
    if pen > 0:
        labels = [HONEYPOT_FLAG_LABELS.get(f, f) for f in flags]
        return ("warn", "Minor consistency flags", "; ".join(labels))
    return None


def platform_signals(c):
    s = c.get("redrob_signals", {}) or {}
    sal = s.get("expected_salary_range_inr_lpa", {}) or {}
    gh = s.get("github_activity_score", -1)
    return {
        "Last active": s.get("last_active_date", "?"),
        "Open to work": "yes" if s.get("open_to_work_flag") else "no",
        "Recruiter response rate": f"{(s.get('recruiter_response_rate') or 0):.0%}",
        "Avg response time (h)": s.get("avg_response_time_hours", "?"),
        "Profile views 30d": s.get("profile_views_received_30d", 0),
        "Saved by recruiters 30d": s.get("saved_by_recruiters_30d", 0),
        "Search appearances 30d": s.get("search_appearance_30d", 0),
        "Interview completion": f"{(s.get('interview_completion_rate') or 0):.0%}",
        "Notice (days)": s.get("notice_period_days", "?"),
        "Work mode": s.get("preferred_work_mode", "?"),
        "Willing to relocate": "yes" if s.get("willing_to_relocate") else "no",
        "Expected salary (LPA)": f"{sal.get('min', '?')}–{sal.get('max', '?')}",
        "GitHub activity": ("none" if gh == -1 else gh),
        "Profile completeness": f"{s.get('profile_completeness_score', 0):.0f}%",
        "Verified": ", ".join(
            n for n, k in (("email", "verified_email"), ("phone", "verified_phone"),
                           ("linkedin", "linkedin_connected")) if s.get(k)) or "none",
    }


def badge_html(label, kind):
    fg, bg = {"go": (GO_FG, GO_BG), "warn": (WARN_FG, WARN_BG)}.get(kind, (FLAG_FG, FLAG_BG))
    return (f'<span style="display:inline-block;font-family:\'JetBrains Mono\',monospace;'
            f'font-size:11px;font-weight:600;letter-spacing:.01em;color:{fg};background:{bg};'
            f'padding:3px 9px;border-radius:999px;margin:0 5px 5px 0;white-space:nowrap;">{label}</span>')


def fingerprint_html(comps):
    bars = []
    for k in COMPONENT_LABELS:
        v = max(0.0, min(1.0, comps.get(k, 0)))
        h = round(6 + v * 30)
        bars.append(
            f'<div style="width:6px;height:{h}px;border-radius:2px;background:{_val_color(v)};'
            f'opacity:0.88;"></div>'
        )
    return ('<div style="display:flex;align-items:flex-end;gap:3px;height:36px;">'
            + "".join(bars) + "</div>")


# ---------------------------------------------------------------- styling

GLOBAL_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Hanken+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;700&display=swap');

html, body, [class*="css"] { font-family: 'Hanken Grotesk', -apple-system, BlinkMacSystemFont, sans-serif; }

.stApp {
    background: linear-gradient(180deg, #F4F7F5, #E9EFEF);
    color: #17150F;
}
/* Streamlit's own top toolbar (hamburger/Deploy) sits fixed above the page and
   was clipping the rounded top edge of our custom .rb-header pill. Hide it so
   our header is the only thing in that space. */
header[data-testid="stHeader"] {
    background: transparent;
    visibility: hidden;
    height: 0;
}
.block-container { padding-top: 2rem; max-width: 1180px; }

.rb-mono { font-family: 'JetBrains Mono', monospace; }

.rb-header {
    display: flex; align-items: center; justify-content: space-between;
    padding: 10px 18px; margin-bottom: 18px; border-radius: 14px;
    background: rgba(255,255,255,0.82); backdrop-filter: blur(16px) saturate(1.3);
    border: 1px solid rgba(30,42,35,0.08);
    box-shadow: 0 1px 2px rgba(30,45,35,0.04), 0 16px 36px -20px rgba(26,48,38,0.3);
}
.rb-logo { display: flex; align-items: center; gap: 10px; }
.rb-logo-mark {
    width: 28px; height: 28px; border-radius: 8px; background: #17150F;
    display: flex; align-items: center; justify-content: center;
}
.rb-logo-dot { width: 7px; height: 7px; border-radius: 50%; background: #5FCB94; }
.rb-logo-text { font-weight: 700; font-size: 15px; }
.rb-logo-text span { color: #8A8775; font-weight: 500; }
.rb-pill {
    display: flex; align-items: center; gap: 7px; font-family: 'JetBrains Mono', monospace;
    font-size: 12px; color: #3C3A30; background: rgba(26,122,78,0.08);
    padding: 5px 12px; border-radius: 999px;
}
.rb-pill-dot { width: 7px; height: 7px; border-radius: 50%; background: #1A7A4E; }

.rb-card {
    background: rgba(255,255,255,0.82); backdrop-filter: blur(16px) saturate(1.3);
    border: 1px solid rgba(30,42,35,0.08); border-radius: 16px; padding: 22px 24px;
    box-shadow: 0 1px 2px rgba(30,45,35,0.04), 0 16px 36px -20px rgba(26,48,38,0.3);
    margin-bottom: 16px;
}
.rb-card-dark {
    background: rgba(19,22,20,0.9); color: #EDEBE2; border-radius: 16px; padding: 22px 24px;
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.1), 0 18px 44px -18px rgba(8,14,10,0.55);
    margin-bottom: 16px;
}
.rb-card-dark .rb-muted { color: #9A988C; }

.rb-eyebrow {
    font-family: 'JetBrains Mono', monospace; font-size: 12px; letter-spacing: 0.06em;
    text-transform: uppercase; color: #1A7A4E; font-weight: 600; margin-bottom: 6px;
}
.rb-h1 { font-size: 34px; font-weight: 700; line-height: 1.15; margin: 0 0 6px 0; }
.rb-h1 .accent { color: #1A7A4E; }
.rb-sub { color: #6B6858; font-size: 14.5px; margin-bottom: 4px; }

.rb-statrow { display: flex; gap: 22px; margin-top: 14px; flex-wrap: wrap; }
.rb-stat-num { font-family: 'JetBrains Mono', monospace; font-weight: 700; font-size: 17px; color: #17150F; }
.rb-stat-lab { font-size: 12px; color: #8A8775; }

.rb-statcard {
    background: rgba(255,255,255,0.82); backdrop-filter: blur(16px) saturate(1.3);
    border: 1px solid rgba(30,42,35,0.08); border-radius: 14px; padding: 14px 18px;
    box-shadow: 0 1px 2px rgba(30,45,35,0.04), 0 16px 36px -20px rgba(26,48,38,0.3);
}
.rb-statcard .num { font-family: 'JetBrains Mono', monospace; font-weight: 700; font-size: 26px; }
.rb-statcard .lab { font-size: 12px; color: #8A8775; margin-top: 2px; }

.rb-row {
    background: rgba(255,255,255,0.82); backdrop-filter: blur(14px) saturate(1.3);
    border: 1px solid rgba(30,42,35,0.08); border-radius: 14px; padding: 16px 20px;
    margin-bottom: 10px; box-shadow: 0 1px 2px rgba(30,45,35,0.04), 0 10px 24px -18px rgba(26,48,38,0.25);
}
.rb-rank { font-family: 'JetBrains Mono', monospace; font-weight: 700; font-size: 20px; }
.rb-name { font-weight: 700; font-size: 15.5px; }
.rb-meta { font-family: 'JetBrains Mono', monospace; font-size: 12px; color: #8A8775; margin-top: 2px; }
.rb-reason { font-size: 13.5px; color: #3C3A30; margin-top: 8px; line-height: 1.5; }
.rb-fit-num { font-family: 'JetBrains Mono', monospace; font-weight: 700; font-size: 24px; text-align: right; }
.rb-fit-lab { font-size: 11.5px; color: #8A8775; text-align: right; margin-top: 2px; }

.rb-bar-track { height: 5px; border-radius: 3px; background: rgba(30,42,35,0.08); overflow: hidden; margin-top: 6px; }
.rb-bar-fill { height: 100%; border-radius: 3px; }

.rb-callout {
    border-left: 3px solid #1A7A4E; background: rgba(26,122,78,0.06);
    border-radius: 0 12px 12px 0; padding: 14px 18px; font-size: 13.5px; line-height: 1.55;
    margin-bottom: 16px;
}
.rb-banner-error { background: #F4E6DD; color: #7A2E10; border-radius: 12px; padding: 12px 16px; margin-bottom: 14px; font-size: 13.5px; }
.rb-banner-warn { background: #F3ECD6; color: #5C4608; border-radius: 12px; padding: 12px 16px; margin-bottom: 14px; font-size: 13.5px; }
.rb-banner-tag { font-weight: 700; margin-right: 6px; }

.rb-comp-row { margin-bottom: 12px; }
.rb-comp-lab { font-size: 13px; display: flex; justify-content: space-between; margin-bottom: 4px; }
.rb-comp-lab .w { color: #8A8775; font-family: 'JetBrains Mono', monospace; font-size: 11.5px; }

.rb-signal-row { display: flex; justify-content: space-between; padding: 7px 0; border-bottom: 1px solid rgba(30,42,35,0.06); font-size: 13px; }
.rb-signal-row .lab { color: #6B6858; }
.rb-signal-row .val { font-family: 'JetBrains Mono', monospace; font-weight: 500; }

.rb-timeline-item { display: flex; gap: 12px; padding-bottom: 16px; }
.rb-timeline-dot { width: 9px; height: 9px; border-radius: 50%; margin-top: 5px; flex-shrink: 0; }
.rb-timeline-current { font-size: 10px; font-weight: 700; color: #115C3A; background: #E4F0E7; padding: 1px 7px; border-radius: 999px; margin-left: 6px; }
.rb-timeline-title { font-weight: 600; font-size: 13.5px; }
.rb-timeline-meta { font-family: 'JetBrains Mono', monospace; font-size: 11.5px; color: #8A8775; margin: 1px 0 4px 0; }
.rb-timeline-desc { font-size: 12.5px; color: #4A4838; line-height: 1.45; }

.rb-skill-row { display: flex; align-items: center; gap: 10px; padding: 6px 0; font-size: 13px; }
.rb-skill-dot { width: 7px; height: 7px; border-radius: 50%; flex-shrink: 0; }
.rb-skill-name { flex: 1; font-weight: 500; }
.rb-skill-meta { font-family: 'JetBrains Mono', monospace; font-size: 11.5px; color: #8A8775; }

.rb-concept-pill { display: inline-block; font-size: 12px; padding: 4px 11px; border-radius: 999px; margin: 0 6px 6px 0; }

div[data-testid="stFileUploaderDropzone"] {
    background: rgba(244,247,245,0.6); border: 1.5px dashed rgba(26,122,78,0.35); border-radius: 12px;
}
div[data-testid="stVerticalBlockBorderWrapper"] {
    background: rgba(255,255,255,0.82) !important; backdrop-filter: blur(16px) saturate(1.3);
    border: 1px solid rgba(30,42,35,0.08) !important; border-radius: 16px !important;
    box-shadow: 0 1px 2px rgba(30,45,35,0.04), 0 16px 36px -20px rgba(26,48,38,0.3) !important;
    margin-bottom: 16px;
}
.stButton button[kind="primary"], .stDownloadButton button {
    background: #17150F; color: #F4F2EA; border-radius: 10px; border: none; font-weight: 600;
}
.stButton button[kind="primary"]:hover, .stDownloadButton button:hover { background: #2A271D; color: #F4F2EA; }
</style>
"""

st.markdown(GLOBAL_CSS, unsafe_allow_html=True)


@st.cache_data
def load_rubric(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def discover_rubrics():
    """Every rubric JSON in artifacts/ is a selectable search profile. Each is
    produced OFFLINE by build_rubric.py from its own job_description.txt — adding
    a new profile is just dropping another rubric file in artifacts/, no code
    change, and ranking itself still never touches the network."""
    out = []
    for path in sorted(glob.glob("artifacts/*.json")):
        try:
            d = load_rubric(path)
            label = d.get("_meta", {}).get("role_title") or os.path.basename(path)
        except Exception:
            label = os.path.basename(path)
        out.append((label, path))
    if not out:
        out = [("Senior AI Engineer — Founding Team", "artifacts/jd_rubric.json")]
    return out


def concept_label(key):
    return CONCEPT_LABELS.get(key, key.replace("_", " "))


RUBRIC_OPTIONS = discover_rubrics()
RUBRIC_PATHS = [p for _, p in RUBRIC_OPTIONS]

ss = st.session_state
ss.setdefault("view", "upload")
ss.setdefault("file_bytes", None)
ss.setdefault("file_name", None)
ss.setdefault("topn", 10)
ss.setdefault("selected_id", None)
ss.setdefault("rubric_path", RUBRIC_PATHS[0])

if ss.rubric_path not in RUBRIC_PATHS:
    ss.rubric_path = RUBRIC_PATHS[0]

if ss.view in ("list", "detail") and ss.file_bytes is None:
    ss.view = "upload"

R = load_rubric(ss.rubric_path)
ROLE_TITLE = R.get("_meta", {}).get("role_title", "Senior AI Engineer")
ROLE_SHORT = ROLE_TITLE.split("—")[0].strip()


def render_header():
    step = {"upload": "01", "list": "02", "detail": "03"}[ss.view]
    nav_parts = []
    for code, label, key in (("01", "Upload", "upload"), ("02", "Shortlist", "list"), ("03", "Candidate", "detail")):
        weight = "700;color:#17150F" if code == step else "500;color:#ACA994"
        nav_parts.append(f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:12px;font-weight:{weight};">{code} {label}</span>')
    nav_html = '<span style="color:#CBC7BA;margin:0 8px;">/</span>'.join(nav_parts)
    st.markdown(
        f"""
        <div class="rb-header">
            <div class="rb-logo">
                <div class="rb-logo-mark"><div class="rb-logo-dot"></div></div>
                <div class="rb-logo-text">Redrob<span> Ranker</span></div>
            </div>
            <div>{nav_html}</div>
            <div class="rb-pill"><div class="rb-pill-dot"></div>Active search · {ROLE_SHORT}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def run_ranking(file_bytes, rubric_path):
    rubric = load_rubric(rubric_path)
    cands = []
    for line in io.TextIOWrapper(io.BytesIO(file_bytes), encoding="utf-8"):
        line = line.strip()
        if line:
            try:
                cands.append(json.loads(line))
            except json.JSONDecodeError:
                pass

    scored = []
    honeypots = 0
    for c in cands:
        pen, flags = honeypot_penalty(c)
        if pen >= 0.55:
            honeypots += 1
        s, comps, ev = score_candidate(c, rubric, pen, flags)
        mod, bev = behavioral_modifier(c, rubric)
        strong = comps.get("role_relevance", 0) >= 0.9 and comps.get("core_requirements", 0) >= 0.5
        scored.append({"raw": s, "cand": c, "comps": comps, "ev": ev, "mod": mod, "bev": bev,
                       "pen": pen, "flags": flags, "strong": strong})

    scored.sort(key=lambda x: -x["raw"])
    norms = normalize_display([x["raw"] for x in scored])
    for x, n in zip(scored, norms):
        x["fit"] = n
        x["cid"] = x["cand"].get("candidate_id")
        x["reasoning"] = build_reasoning(x["cand"], x["comps"], x["ev"])

    return cands, scored, honeypots


# ---------------------------------------------------------------- screens

def screen_upload():
    left, right = st.columns([1.15, 1], gap="medium")

    with left:
        st.markdown('<div class="rb-eyebrow">Intelligent candidate ranking</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="rb-h1">Ten great matches.<br>'
            '<span class="accent">Not a thousand maybes.</span></div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="rb-sub">Structural reasoning over profiles — not keyword matching. '
            'CPU-only, no network calls at ranking time.</div>',
            unsafe_allow_html=True,
        )

        with st.container(border=True):
            up = st.file_uploader("Drop a candidates.jsonl file", type=["jsonl"], label_visibility="visible")
            if up is not None:
                ss.file_bytes = up.getvalue()
                ss.file_name = up.name

            if ss.file_bytes:
                n_lines = ss.file_bytes.count(b"\n") + (0 if ss.file_bytes.endswith(b"\n") else 1)
                st.caption(f"**{ss.file_name}** · ~{n_lines:,} records · {len(ss.file_bytes)/1e6:.1f} MB")

            ss.topn = st.slider("How many to shortlist", 5, 100, ss.topn, step=5)

            if st.button(f"Start ranking · top {ss.topn} →", type="primary", disabled=not ss.file_bytes,
                         use_container_width=True):
                ss.view = "list"
                st.rerun()

            st.markdown(
                f"""
                <div class="rb-statrow">
                    <div><div class="rb-stat-num">~45s</div><div class="rb-stat-lab">full 100K pool, CPU only</div></div>
                    <div><div class="rb-stat-num">7</div><div class="rb-stat-lab">transparent score components</div></div>
                    <div><div class="rb-stat-num">&lt;10%</div><div class="rb-stat-lab">honeypot tolerance in top 100</div></div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with right:
        # Mirrors the left column's eyebrow + subtitle rhythm so the dark card
        # below starts at roughly the same height as the upload card on the left.
        st.markdown(
            '<div class="rb-eyebrow">Search profile</div>'
            '<div class="rb-sub" style="margin-bottom:16px;">'
            'Switch roles to re-rank the same candidate pool against a different rubric.</div>',
            unsafe_allow_html=True,
        )
        labels = [lab for lab, _ in RUBRIC_OPTIONS]
        cur_idx = RUBRIC_PATHS.index(ss.rubric_path) if ss.rubric_path in RUBRIC_PATHS else 0
        chosen = st.selectbox("Search profile", labels, index=cur_idx,
                               disabled=len(labels) <= 1, key="_rubric_select",
                               label_visibility="collapsed")
        new_path = dict(RUBRIC_OPTIONS)[chosen]
        if new_path != ss.rubric_path:
            ss.rubric_path = new_path
            st.rerun()
        if len(labels) <= 1:
            st.caption("Only one profile available — drop more rubric files in artifacts/ "
                       "(built offline via build_rubric.py) to add more.")
        st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)

        concept_pills = "".join(
            f'<span class="rb-concept-pill" style="background:#E4F0E7;color:#115C3A;">{concept_label(c["name"])}</span>'
            for c in R.get("must_have_concepts", [])
        )
        meta = R.get("_meta", {})
        band = R.get("experience_band", {})
        st.markdown(
            f"""
            <div class="rb-card-dark">
                <div class="rb-eyebrow" style="color:#5FCB94;">Search definition</div>
                <div style="font-size:19px;font-weight:700;margin-bottom:4px;">{ROLE_TITLE}</div>
                <div class="rb-muted rb-mono" style="font-size:12.5px;margin-bottom:16px;">
                    {meta.get('company', 'Redrob AI')} · {meta.get('location', 'Pune/Noida, India (Hybrid)')} ·
                    {band.get('min', 5)}–{band.get('max', 9)} yrs
                </div>
                <div class="rb-muted" style="font-size:11.5px;letter-spacing:.05em;text-transform:uppercase;margin-bottom:8px;">
                    Must-have signals
                </div>
                <div style="margin-bottom:18px;">{concept_pills}</div>
                <div class="rb-muted" style="font-size:12.5px;line-height:1.55;border-top:1px solid rgba(255,255,255,0.1);padding-top:14px;">
                    The trap: the dataset hides keyword-stuffers and honeypots — profiles with every
                    AI buzzword but no hands-on evidence, and profiles with internally impossible
                    histories. We score consistency and real-work evidence, not keyword density.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def screen_list():
    cands, scored, honeypots = run_ranking(ss.file_bytes, ss.rubric_path)
    shown = scored[: ss.topn]
    strong_n = sum(1 for x in scored if x["strong"])

    title_col, btn_col = st.columns([3, 1.3])
    with title_col:
        st.markdown('<div class="rb-h1" style="font-size:26px;">Ranked shortlist</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="rb-sub">Top {len(shown)} of {len(cands):,} candidates, ranked by fit.</div>',
                     unsafe_allow_html=True)
    with btn_col:
        csv_lines = ["candidate_id,rank,score,reasoning"]
        for i, x in enumerate(shown):
            reason = '"' + x["reasoning"].replace('"', '""') + '"'
            csv_lines.append(f'{x["cid"]},{i + 1},{x["fit"]:.4f},{reason}')
        st.download_button("Download ranked CSV", "\n".join(csv_lines),
                            file_name="ranked_sample.csv", mime="text/csv", use_container_width=True)
        if st.button("↑ Upload a different file", use_container_width=True):
            ss.view = "upload"
            ss.file_bytes = None
            st.rerun()

    stat_cols = st.columns(4)
    for col, num, lab in zip(
        stat_cols,
        (f"{len(cands):,}", str(strong_n), str(honeypots), str(len(shown))),
        ("Candidates loaded", "Strong fits", "Honeypots flagged", "Requested"),
    ):
        col.markdown(f'<div class="rb-statcard"><div class="num">{num}</div><div class="lab">{lab}</div></div>',
                      unsafe_allow_html=True)

    st.markdown("<div style='height:18px;'></div>", unsafe_allow_html=True)

    for i, x in enumerate(shown):
        p = x["cand"].get("profile", {})
        badges_html = "".join(badge_html(lab, kind) for lab, kind in derive_flags(x))
        rank_color = ACCENT if i < 3 else "#CBC7BA"
        fit_color = _fit_color(x["fit"])

        row_l, row_m, row_r = st.columns([0.6, 3.4, 1.1])
        with row_l:
            st.markdown(f'<div class="rb-rank" style="color:{rank_color};padding-top:6px;">{i + 1:02d}</div>',
                         unsafe_allow_html=True)
        with row_m:
            st.markdown(
                f"""
                <div class="rb-name">{p.get('anonymized_name', x['cid'])}</div>
                <div class="rb-meta">{p.get('current_title', '?')} · {p.get('current_company', '?')}
                    &nbsp;·&nbsp; {p.get('years_of_experience', '?')} yrs · {p.get('location', '?')}</div>
                <div style="margin-top:8px;">{badges_html}</div>
                <div class="rb-reason">{x['reasoning']}</div>
                """,
                unsafe_allow_html=True,
            )
        with row_r:
            st.markdown(
                f"""
                <div style="text-align:right;">{fingerprint_html(x['comps'])}</div>
                <div class="rb-fit-num" style="color:{fit_color};">{x['fit']:.2f}</div>
                <div class="rb-bar-track"><div class="rb-bar-fill" style="width:{x['fit']*100:.0f}%;background:{fit_color};"></div></div>
                <div class="rb-fit-lab">{fit_label(x)}</div>
                """,
                unsafe_allow_html=True,
            )
            if st.button("View →", key=f"open_{x['cid']}", use_container_width=True):
                ss.selected_id = x["cid"]
                ss.view = "detail"
                st.rerun()
        st.markdown("<div style='border-bottom:1px solid rgba(30,42,35,0.07);margin:4px 0 14px 0;'></div>",
                     unsafe_allow_html=True)


def screen_detail():
    cands, scored, honeypots = run_ranking(ss.file_bytes, ss.rubric_path)
    idx = next((i for i, x in enumerate(scored) if x["cid"] == ss.selected_id), None)
    if idx is None:
        st.warning("Candidate not found in the current ranked set.")
        if st.button("← Back to shortlist"):
            ss.view = "list"
            st.rerun()
        return

    x = scored[idx]
    c = x["cand"]
    p = c.get("profile", {})

    if st.button("← Back to shortlist"):
        ss.view = "list"
        st.rerun()

    banner = banner_for(x)
    if banner:
        kind, tag, text = banner
        cls = "rb-banner-error" if kind == "error" else "rb-banner-warn"
        st.markdown(f'<div class="{cls}"><span class="rb-banner-tag">{tag}</span>{text}</div>',
                     unsafe_allow_html=True)

    badges_html = "".join(badge_html(lab, kind) for lab, kind in derive_flags(x))
    head_l, head_r = st.columns([2, 1])
    with head_l:
        st.markdown(
            f"""
            <div class="rb-mono" style="font-size:12px;color:#8A8775;margin-bottom:6px;">
                RANK {idx + 1} OF {len(scored)} · {x['cid']}
            </div>
            <div class="rb-h1" style="font-size:28px;">{p.get('anonymized_name', x['cid'])}</div>
            <div class="rb-sub" style="margin-bottom:4px;">{p.get('current_title', '?')} · {p.get('current_company', '?')}</div>
            <div class="rb-mono" style="font-size:12px;color:#8A8775;margin-bottom:10px;">
                {p.get('years_of_experience', '?')} yrs · {p.get('location', '?')}, {p.get('country', '')}
            </div>
            <div>{badges_html}</div>
            """,
            unsafe_allow_html=True,
        )
    with head_r:
        fit_color_dark = "#5FCB94" if x["fit"] >= 0.7 else ("#EDEBE2" if x["fit"] >= 0.4 else "#E89B7A")
        st.markdown(
            f"""
            <div class="rb-card-dark">
                <div class="rb-muted" style="font-size:11px;letter-spacing:.05em;text-transform:uppercase;">Final fit</div>
                <div class="rb-mono" style="font-size:40px;font-weight:700;color:{fit_color_dark};">{x['fit']:.2f}</div>
                <div style="font-size:13px;margin-bottom:8px;">{fit_label(x)}</div>
                <div style="height:4px;border-radius:3px;background:rgba(255,255,255,0.12);overflow:hidden;">
                    <div style="height:100%;width:{x['fit']*100:.0f}%;background:{fit_color_dark};"></div>
                </div>
                <div class="rb-muted rb-mono" style="font-size:11.5px;margin-top:10px;">
                    Availability ×{x['mod']:.2f}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(f'<div class="rb-callout">{x["reasoning"]}</div>', unsafe_allow_html=True)

    left, right = st.columns(2, gap="medium")

    with left:
        with st.container(border=True):
            st.markdown("**Component scores**")
            W = R["weights"]
            for k, lab in COMPONENT_LABELS.items():
                v = max(0.0, min(1.0, x["comps"].get(k, 0)))
                st.markdown(
                    f'<div class="rb-comp-row"><div class="rb-comp-lab"><span>{lab}</span>'
                    f'<span class="w">w={W.get(k, 0):.2f} · {v:.2f}</span></div>'
                    f'<div class="rb-bar-track"><div class="rb-bar-fill" style="width:{v*100:.0f}%;background:{_val_color(v)};"></div></div></div>',
                    unsafe_allow_html=True,
                )

        bev = x["bev"]
        mod = x["mod"]
        mod_color = ACCENT if mod >= 1.0 else ("#17150F" if mod >= 0.7 else LOW)
        if mod < 0.7:
            note = "Strong on paper, but behavioral signals say this person is hard to reach right now — steeply down-weighted."
        elif mod < 1.0:
            note = "Mildly down-weighted for availability — still reachable, just not ideal timing."
        else:
            note = "Engaged and available — no availability discount applied."
        with st.container(border=True):
            st.markdown(
                f'<div style="font-size:11px;letter-spacing:.05em;text-transform:uppercase;color:#8A8775;">Availability modifier</div>'
                f'<div class="rb-mono" style="font-size:30px;font-weight:700;color:{mod_color};">×{mod:.2f}</div>'
                f'<div style="font-size:12.5px;color:#4A4838;margin-bottom:12px;">{note}</div>',
                unsafe_allow_html=True,
            )
            sig = c.get("redrob_signals", {}) or {}
            days = _days_since(sig.get("last_active_date"))
            recency_lab = f"{days}d ago" if days is not None else "unknown"
            mini = st.columns(4)
            mini_vals = (
                ("Response rate", f"{bev.get('response_rate', 0):.0%}"),
                ("Last active", recency_lab),
                ("Open to work", "Yes" if bev.get("open_to_work") else "No"),
                ("Notice period", f"{bev.get('notice_days', '?')}d"),
            )
            for col, (lab, val) in zip(mini, mini_vals):
                col.markdown(f'<div style="font-size:11px;color:#8A8775;">{lab}</div>'
                              f'<div class="rb-mono" style="font-size:13.5px;font-weight:600;">{val}</div>',
                              unsafe_allow_html=True)

        concepts = x["ev"].get("core_requirements", {}).get("concepts", [])
        strong_c = [concept_label(k) for k, s in concepts if s >= 1.0]
        weak_c = [concept_label(k) for k, s in concepts if 0 < s < 1.0]
        with st.container(border=True):
            st.markdown('<div style="font-weight:600;margin-bottom:10px;">Requirement evidence</div>',
                         unsafe_allow_html=True)
            if strong_c:
                st.markdown('<div style="font-size:11.5px;color:#8A8775;margin-bottom:4px;">Hands-on in work history</div>'
                             + "".join(f'<span class="rb-concept-pill" style="background:#E4F0E7;color:#115C3A;">{l}</span>' for l in strong_c),
                             unsafe_allow_html=True)
            if weak_c:
                st.markdown('<div style="font-size:11.5px;color:#8A8775;margin:10px 0 4px 0;">Mentioned only (skills/summary)</div>'
                             + "".join(f'<span class="rb-concept-pill" style="background:rgba(30,42,35,0.07);color:#6B6858;">{l}</span>' for l in weak_c),
                             unsafe_allow_html=True)
            if not strong_c and not weak_c:
                st.markdown('<span style="font-size:13px;color:#8A8775;">No must-have concepts found in this profile.</span>',
                             unsafe_allow_html=True)

    with right:
        with st.container(border=True):
            st.markdown('<div style="font-weight:600;margin-bottom:6px;">Platform activity & availability</div>',
                         unsafe_allow_html=True)
            for lab, val in platform_signals(c).items():
                st.markdown(f'<div class="rb-signal-row"><span class="lab">{lab}</span><span class="val">{val}</span></div>',
                             unsafe_allow_html=True)

        hist = c.get("career_history", []) or []
        with st.container(border=True):
            st.markdown('<div style="font-weight:600;margin-bottom:8px;">Career history</div>', unsafe_allow_html=True)
            for job in hist:
                dot_color = ACCENT if job.get("is_current") else "#CFCBBE"
                current_badge = '<span class="rb-timeline-current">CURRENT</span>' if job.get("is_current") else ""
                st.markdown(
                    f"""
                    <div class="rb-timeline-item">
                        <div class="rb-timeline-dot" style="background:{dot_color};"></div>
                        <div>
                            <div class="rb-timeline-title">{job.get('title', '?')} · {job.get('company', '?')}{current_badge}</div>
                            <div class="rb-timeline-meta">{job.get('duration_months', '?')} months · {job.get('industry', '?')}</div>
                            <div class="rb-timeline-desc">{job.get('description', '')}</div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        skills = c.get("skills", []) or []
        top_sk = sorted(skills, key=lambda s: -(s.get("duration_months", 0) or 0))[:8]
        skt = x["ev"].get("skill_trust", {}).get("skill_trust_ratio", 1.0)
        with st.container(border=True):
            st.markdown(f'<div style="font-weight:600;margin-bottom:8px;">Top skills <span style="font-size:11.5px;color:#8A8775;font-weight:400;">'
                         f'(trust ratio {skt:.2f})</span></div>', unsafe_allow_html=True)
            for s in top_sk:
                dur = s.get("duration_months", 0) or 0
                dot_color = ACCENT if dur > 0 else LOW
                st.markdown(
                    f'<div class="rb-skill-row"><div class="rb-skill-dot" style="background:{dot_color};"></div>'
                    f'<div class="rb-skill-name">{s.get("name", "?")}</div>'
                    f'<div class="rb-skill-meta">{s.get("proficiency", "?")} · {dur}mo · {s.get("endorsements", 0)} endorsed</div></div>',
                    unsafe_allow_html=True,
                )

    summ = p.get("summary")
    if summ:
        st.markdown(f'<div class="rb-card"><div style="font-weight:600;margin-bottom:6px;">Profile summary</div>'
                     f'<div style="font-size:13.5px;line-height:1.55;color:#3C3A30;">{summ}</div></div>',
                     unsafe_allow_html=True)


# ---------------------------------------------------------------- main

render_header()

if ss.view == "upload":
    screen_upload()
elif ss.view == "list":
    screen_list()
elif ss.view == "detail":
    screen_detail()
