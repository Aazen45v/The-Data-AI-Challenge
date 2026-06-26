"""
reasoning.py — write honest, candidate-specific reasoning strings.

The spec penalizes empty / identical / name-only-template / hallucinated /
rank-contradicting reasoning. We generate reasoning purely from facts already
extracted during scoring, so it is specific and CANNOT hallucinate skills the
candidate doesn't have. No LLM is used here (and none is allowed at rank time).
"""

_CONCEPT_LABEL = {
    "embeddings_retrieval": "embeddings/retrieval",
    "vector_search_infra": "vector search infra",
    "ranking_recsys": "ranking/recsys",
    "eval_frameworks": "ranking evaluation",
    "modern_ml_llm": "LLM/modern ML",
    "python_production": "production Python",
}


def build_reasoning(cand, comps, evidence):
    prof = cand.get("profile", {})
    title = prof.get("current_title", "professional")
    yoe = prof.get("years_of_experience", 0)
    bits = []

    # 1) who they are
    bits.append(f"{title}, {yoe:.1f} yrs")

    # 2) strongest requirement evidence actually found in their work
    concepts = evidence.get("core_requirements", {}).get("concepts", [])
    strong = [c for c, s in concepts if s >= 1.0]
    weak = [c for c, s in concepts if 0 < s < 1.0]
    if strong:
        labels = ", ".join(_CONCEPT_LABEL.get(c, c) for c in strong[:3])
        bits.append(f"hands-on evidence in {labels}")
    elif weak:
        labels = ", ".join(_CONCEPT_LABEL.get(c, c) for c in weak[:2])
        bits.append(f"some exposure to {labels} (skills/summary only)")
    else:
        bits.append("no direct retrieval/ranking evidence in work history")

    # 3) role-fit verdict (the anti-stuffer signal)
    rr = evidence.get("role_relevance", {})
    if rr.get("offdomain_current"):
        bits.append("current role is off-domain despite AI keywords — likely keyword-stuffer")
    elif rr.get("strong_title"):
        bits.append("core ML/eng title match")
    elif rr.get("adjacent_title"):
        bits.append("adjacent engineering background")

    # 4) trajectory / disqualifier notes (only if present)
    tf = evidence.get("trajectory", {}).get("trajectory_flags", [])
    notes = []
    if "consulting_only" in tf:
        notes.append("entire career at services firms (JD disqualifier)")
    if "title_chaser" in tf:
        notes.append("frequent short stints")
    if "research_only_no_prod" in tf:
        notes.append("research-only, no production signal")
    if "cv_speech_only" in tf:
        notes.append("CV/speech focus without NLP/IR")
    if "product_company" in tf:
        notes.append("product-company experience")
    if "production_signal" in tf and "production" not in " ".join(notes):
        notes.append("shipped to real users")
    if notes:
        bits.append("; ".join(notes[:2]))

    # 5) availability
    bev = evidence.get("behavioral", {})
    avail = []
    rr_rate = bev.get("response_rate")
    if rr_rate is not None:
        avail.append(f"response rate {rr_rate:.2f}")
    if bev.get("recency", 1) <= 0.15:
        avail.append("inactive 4+ months")
    if not bev.get("open_to_work", True):
        avail.append("not open-to-work")
    nd = bev.get("notice_days")
    if nd is not None and nd > 60:
        avail.append(f"{nd}d notice")
    if avail:
        bits.append(", ".join(avail))

    # 6) honeypot flag (rare; explains a low rank)
    hp = evidence.get("honeypot")
    if hp:
        bits.append(f"profile-consistency issues ({', '.join(hp[:2])})")

    text = "; ".join(bits)
    text = text[0].upper() + text[1:]
    if not text.endswith("."):
        text += "."
    # keep it to ~1-2 sentences / reasonable length
    return text[:300]
