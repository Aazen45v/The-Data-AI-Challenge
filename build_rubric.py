#!/usr/bin/env python3
"""
build_rubric.py — OFFLINE pre-computation step (the ONLY LLM/network use).

This is NOT part of the ranking step. It runs once, on your machine, to turn
the free-text job description into the structured rubric that rank.py consumes
(artifacts/jd_rubric.json). The spec explicitly allows pre-computation to use
the network and exceed the 5-minute budget; the *ranking* step (rank.py) then
runs with zero network.

Why an LLM here at all? Because "understand the JD, don't keyword-match it" is
exactly what an LLM is good at: reading between the lines of a messy founder-
written JD to extract must-haves, the named disqualifiers (research-only,
consulting-only, title-chasing, CV/speech-only), the soft experience band, and
the location intent. We do that ONCE and freeze the result.

The committed artifacts/jd_rubric.json was produced/verified this way and is
hand-checked. Re-running this script is optional and reproducible; if the call
fails or no key is set, the existing committed rubric is left untouched so the
pipeline never depends on the network.

Usage:
  export OPENROUTER_API_KEY=sk-or-...
  python build_rubric.py --jd job_description.txt --out artifacts/jd_rubric.json \
      --model anthropic/claude-3.5-sonnet
"""
import argparse
import json
import os
import sys
import urllib.request

ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"

SYSTEM = """You are a senior technical recruiter + ML engineer. You convert a
messy, founder-written job description into a STRICT JSON ranking rubric for an
automated candidate ranker. You read between the lines: extract not just the
words, but what the role actually needs, the disqualifiers the JD names, and the
implicit signals. Output ONLY valid JSON, no prose, no markdown fences."""

# The schema we want back. We pass the current committed rubric as a worked
# example so the model returns the same shape (which rank.py already consumes).
INSTRUCTIONS = """Return a JSON object with EXACTLY these keys, same shapes as the
example below: experience_band, must_have_concepts, nice_to_have_concepts,
role_relevance, consulting_firms, research_only_markers, production_markers,
cv_speech_robotics, nlp_ir_markers, location, weights, behavioral.

Rules:
- must_have_concepts: each {name, weight (0-1), terms:[lowercase strings]}.
  Terms must be substrings you'd literally find in a profile. Cover synonyms
  and tools, not just the JD's exact words (e.g. retrieval -> also "bge","e5").
- role_relevance.negative: titles that, if they are the candidate's CURRENT role,
  mean they are a keyword-stuffer no matter how good their skills list looks.
- weights across the 7 scoring components must sum to ~1.0.
- Encode the JD's explicit disqualifiers (pure research w/o production;
  consulting-only career; title-chasing; no recent code; CV/speech/robotics
  without NLP/IR) via the marker lists.
Here is the current rubric to match the SHAPE of (improve its content, keep keys):
"""


def call_openrouter(jd_text, example_rubric, model, api_key):
    body = {
        "model": model,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content":
                INSTRUCTIONS
                + json.dumps(example_rubric)
                + "\n\nJOB DESCRIPTION:\n" + jd_text
                + "\n\nReturn ONLY the JSON rubric."},
        ],
    }
    req = urllib.request.Request(
        ENDPOINT,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/redrob-ranker",
            "X-Title": "Redrob Ranker rubric build",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    content = data["choices"][0]["message"]["content"].strip()
    # strip accidental ```json fences
    if content.startswith("```"):
        content = content.split("```")[1]
        if content.startswith("json"):
            content = content[4:]
    return json.loads(content)


REQUIRED_KEYS = {"experience_band", "must_have_concepts", "role_relevance",
                 "weights", "behavioral", "location"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jd", default="job_description.txt")
    ap.add_argument("--out", default="artifacts/jd_rubric.json")
    ap.add_argument("--model", default="anthropic/claude-3.5-sonnet")
    args = ap.parse_args()

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("OPENROUTER_API_KEY not set. Leaving committed rubric untouched. "
              "(rank.py does not need this step.)", file=sys.stderr)
        sys.exit(0)

    jd_text = open(args.jd, "r", encoding="utf-8").read()
    example = json.load(open(args.out, "r", encoding="utf-8"))  # current committed

    try:
        rubric = call_openrouter(jd_text, example, args.model, api_key)
    except Exception as e:
        print(f"LLM call failed ({e}). Keeping committed rubric.", file=sys.stderr)
        sys.exit(1)

    missing = REQUIRED_KEYS - set(rubric)
    if missing:
        print(f"Model output missing keys {missing}; keeping committed rubric.",
              file=sys.stderr)
        sys.exit(1)

    rubric.setdefault("_meta", {})
    rubric["_meta"]["source"] = f"Generated by build_rubric.py via OpenRouter model {args.model}"
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(rubric, f, indent=2, ensure_ascii=False)
    print(f"Wrote refreshed rubric to {args.out}")


if __name__ == "__main__":
    main()
