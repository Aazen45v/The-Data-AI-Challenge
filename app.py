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
import io
import json
from datetime import date

import streamlit as st

from ranker.honeypot import honeypot_penalty
from ranker.scoring import (
    score_candidate, behavioral_modifier, normalize_display,
)
from ranker.reasoning import build_reasoning

st.set_page_config(page_title="Redrob Candidate Ranker", layout="wide")

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

st.title("Redrob \u2014 Intelligent Candidate Ranker")
st.caption("Structural reasoning over profiles, not keyword matching. "
           "CPU-only, no network at ranking time.")


@st.cache_data
def load_rubric():
    with open("artifacts/jd_rubric.json") as f:
        return json.load(f)


R = load_rubric()

with st.expander("What this does"):
    st.markdown(
        "- Reads each profile and scores **role relevance, requirement evidence "
        "in real work history, skill trust, experience fit, trajectory, semantic "
        "concept match, and location**.\n"
        "- Applies a **behavioral availability** modifier (response rate, recency, "
        "open-to-work, notice).\n"
        "- Detects **impossible/honeypot** profiles via internal-consistency checks "
        "and collapses their score.\n"
        "- Emits **honest, candidate-specific reasoning** for every row."
    )

up = st.file_uploader("Upload a candidate sample (.jsonl, <=100 records)", type=["jsonl"])
topn = st.slider("Show top N", 5, 100, 10)


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
        "Expected salary (LPA)": f"{sal.get('min','?')}\u2013{sal.get('max','?')}",
        "GitHub activity": ("none" if gh == -1 else gh),
        "Profile completeness": f"{s.get('profile_completeness_score', 0):.0f}%",
        "Verified": ", ".join(
            x for x, k in (("email", "verified_email"), ("phone", "verified_phone"),
                           ("linkedin", "linkedin_connected")) if s.get(k)) or "none",
    }


if not up:
    st.info("Upload a .jsonl sample to see the ranker run. A 50-row sample "
            "(sample_candidates) works well. Note: a slice like the first 100 "
            "rows of the pool is an arbitrary sample, not the strongest "
            "candidates in the full 100k.")
    st.stop()

# ---- load ----
cands = []
for line in io.TextIOWrapper(up, encoding="utf-8"):
    line = line.strip()
    if line:
        try:
            cands.append(json.loads(line))
        except json.JSONDecodeError:
            pass

# ---- score everyone ----
scored = []
honeypots = 0
for c in cands:
    pen, flags = honeypot_penalty(c)
    if pen >= 0.55:
        honeypots += 1
    s, comps, ev = score_candidate(c, R, pen, flags)
    mod, _ = behavioral_modifier(c, R)
    strong = comps.get("role_relevance", 0) >= 0.9 and comps.get("core_requirements", 0) >= 0.5
    scored.append({"raw": s, "cand": c, "comps": comps, "ev": ev,
                   "mod": mod, "pen": pen, "flags": flags, "strong": strong})

scored.sort(key=lambda x: -x["raw"])
norms = normalize_display([x["raw"] for x in scored])
for x, n in zip(scored, norms):
    x["fit"] = n

# ---- summary ----
c1, c2, c3, c4 = st.columns(4)
c1.metric("Candidates loaded", len(cands))
c2.metric("Honeypots flagged", honeypots, help="Impossible/inconsistent profiles, score collapsed")
c3.metric("Strong fits", sum(1 for x in scored if x["strong"]),
          help="Core ML/eng role + first-hand requirement evidence")
c4.metric("Shown", min(topn, len(scored)))

shown = scored[:topn]

# ---- ranked table ----
table = []
for i, x in enumerate(shown):
    p = x["cand"].get("profile", {})
    table.append({
        "rank": i + 1,
        "candidate_id": x["cand"].get("candidate_id"),
        "title": p.get("current_title"),
        "yoe": p.get("years_of_experience"),
        "fit": x["fit"],
        "reasoning": build_reasoning(x["cand"], x["comps"], x["ev"]),
    })

st.subheader("Ranked shortlist")
st.dataframe(
    table, use_container_width=True, hide_index=True,
    column_config={
        "fit": st.column_config.ProgressColumn(
            "fit", min_value=0.0, max_value=1.0, format="%.2f"),
        "reasoning": st.column_config.TextColumn("reasoning", width="large"),
    },
)

# download (normalized, submission-style)
csv_lines = ["candidate_id,rank,score,reasoning"]
for r in table:
    reason = '"' + r["reasoning"].replace('"', '""') + '"'
    csv_lines.append(f'{r["candidate_id"]},{r["rank"]},{r["fit"]:.4f},{reason}')
st.download_button("Download ranked CSV", "\n".join(csv_lines),
                   file_name="ranked_sample.csv", mime="text/csv")

# ---- per-candidate full breakdown ----
st.subheader("Full breakdown")
st.caption("Why each candidate ranks where they do \u2014 every signal the ranker used.")

for i, x in enumerate(shown):
    c = x["cand"]
    p = c.get("profile", {})
    head = (f"#{i+1}  \u00b7  {p.get('current_title','?')}  \u00b7  "
            f"{p.get('years_of_experience','?')} yrs  \u00b7  "
            f"fit {x['fit']:.2f}  \u00b7  {c.get('candidate_id')}")
    with st.expander(head):
        if x["pen"] >= 0.55:
            st.error("Flagged as an impossible/honeypot profile \u2014 score collapsed: "
                     + ", ".join(x["flags"]))
        elif x["flags"]:
            st.warning("Minor consistency notes: " + ", ".join(x["flags"]))

        left, right = st.columns(2)

        with left:
            st.markdown("**Component scores**")
            for k, lab in COMPONENT_LABELS.items():
                v = x["comps"].get(k, 0)
                st.progress(min(max(v, 0.0), 1.0), text=f"{lab}: {v:.2f}")
            st.markdown(f"**Behavioral availability modifier:** \u00d7{x['mod']:.2f}")

            concepts = x["ev"].get("core_requirements", {}).get("concepts", [])
            if concepts:
                strong = [CONCEPT_LABELS.get(k, k) for k, s in concepts if s >= 1.0]
                weak = [CONCEPT_LABELS.get(k, k) for k, s in concepts if 0 < s < 1.0]
                st.markdown("**Requirement evidence**")
                if strong:
                    st.markdown("- hands-on in work history: " + ", ".join(strong))
                if weak:
                    st.markdown("- mentioned (skills/summary): " + ", ".join(weak))

        with right:
            st.markdown("**Platform activity & availability**")
            sig = platform_signals(c)
            st.table({"signal": list(sig.keys()), "value": list(sig.values())})

        st.markdown("**Career history**")
        hist = c.get("career_history", []) or []
        if hist:
            st.table({
                "company": [h.get("company", "?") for h in hist],
                "title": [h.get("title", "?") for h in hist],
                "months": [h.get("duration_months", "?") for h in hist],
                "industry": [h.get("industry", "?") for h in hist],
            })

        skills = c.get("skills", []) or []
        if skills:
            st.markdown("**Top skills** (proficiency \u00b7 months used \u00b7 endorsements)")
            top_sk = sorted(skills, key=lambda s: -(s.get("duration_months", 0) or 0))[:8]
            st.table({
                "skill": [s.get("name", "?") for s in top_sk],
                "proficiency": [s.get("proficiency", "?") for s in top_sk],
                "months": [s.get("duration_months", 0) for s in top_sk],
                "endorsements": [s.get("endorsements", 0) for s in top_sk],
            })

        summ = p.get("summary")
        if summ:
            st.markdown("**Profile summary**")
            st.write(summ)
