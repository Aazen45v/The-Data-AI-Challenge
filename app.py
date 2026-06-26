#!/usr/bin/env python3
"""
app.py — hosted sandbox demo (satisfies submission_spec Section 10.5).

Deploy on HuggingFace Spaces / Streamlit Cloud. It accepts a small candidate
sample (<=100 records, .jsonl) and runs the SAME ranking pipeline rank.py uses,
end-to-end, on CPU with no network, then shows the ranked table + download.

Run locally:  streamlit run app.py
"""
import io
import json

import streamlit as st

from ranker.honeypot import honeypot_penalty
from ranker.scoring import score_candidate
from ranker.reasoning import build_reasoning

st.set_page_config(page_title="Redrob Candidate Ranker", layout="wide")
st.title("Redrob — Intelligent Candidate Ranker")
st.caption("Structural reasoning over profiles, not keyword matching. "
           "CPU-only, no network at ranking time.")

with open("artifacts/jd_rubric.json") as f:
    R = json.load(f)

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
topn = st.slider("Show top N", 5, 100, 25)

if up:
    cands = []
    for line in io.TextIOWrapper(up, encoding="utf-8"):
        line = line.strip()
        if line:
            try:
                cands.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    st.write(f"Loaded **{len(cands)}** candidates.")

    scored = []
    for c in cands:
        pen, flags = honeypot_penalty(c)
        s, comps, ev = score_candidate(c, R, pen, flags)
        scored.append((s, c, comps, ev))
    scored.sort(key=lambda x: -x[0])

    rows = []
    for i, (s, c, comps, ev) in enumerate(scored[:topn]):
        rows.append({
            "rank": i + 1,
            "candidate_id": c.get("candidate_id"),
            "title": c.get("profile", {}).get("current_title"),
            "yoe": c.get("profile", {}).get("years_of_experience"),
            "score": round(s, 4),
            "reasoning": build_reasoning(c, comps, ev),
        })
    st.dataframe(rows, use_container_width=True, hide_index=True)

    csv_lines = ["candidate_id,rank,score,reasoning"]
    for r in rows:
        reason = '"' + r["reasoning"].replace('"', '""') + '"'
        csv_lines.append(f'{r["candidate_id"]},{r["rank"]},{r["score"]:.4f},{reason}')
    st.download_button("Download ranked CSV", "\n".join(csv_lines),
                       file_name="ranked_sample.csv", mime="text/csv")
else:
    st.info("Upload a .jsonl sample to see the ranker run. "
            "A 50-row sample (sample_candidates) works well.")
