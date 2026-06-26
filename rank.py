#!/usr/bin/env python3
"""
rank.py — produce the top-100 candidate ranking for the released JD.

COMPUTE CONTRACT (enforced at Stage 3):
  * CPU only, no GPU
  * NO network — no hosted LLM/API calls of any kind
  * < 5 minutes, < 16 GB RAM on the full 100k pool

How it stays inside the budget:
  * candidates.jsonl is STREAMED line-by-line (never fully loaded)
  * only a small top-K heap is kept in memory
  * scoring is pure-Python, single pass, no model downloads
  * the only "LLM understanding" of the JD is precomputed offline into
    artifacts/jd_rubric.json (committed) and just read here.

Usage:
  python rank.py --candidates ./candidates.jsonl --out ./submission.csv
"""
import argparse
import csv
import heapq
import json
import sys
import time
from pathlib import Path

from ranker.honeypot import honeypot_penalty
from ranker.scoring import score_candidate
from ranker.reasoning import build_reasoning

TOPN = 100
KEEP = 400  # keep a buffer > 100 so tie-breaking & honeypot collapse are clean


def load_rubric(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def iter_candidates(path):
    """Stream a .jsonl (or .jsonl.gz) file one record at a time."""
    p = Path(path)
    if p.suffix == ".gz":
        import gzip
        opener = lambda: gzip.open(p, "rt", encoding="utf-8")
    else:
        opener = lambda: open(p, "r", encoding="utf-8")
    with opener() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rubric", default=str(Path(__file__).parent / "artifacts" / "jd_rubric.json"))
    ap.add_argument("--progress", action="store_true")
    args = ap.parse_args()

    R = load_rubric(args.rubric)
    t0 = time.time()

    heap = []  # min-heap of (score, tiebreak, candidate_id, reasoning)
    n = 0
    for cand in iter_candidates(args.candidates):
        n += 1
        cid = cand.get("candidate_id", "")
        if not cid:
            continue

        pen, flags = honeypot_penalty(cand)
        score, comps, evidence = score_candidate(cand, R, pen, flags)

        # deterministic secondary signal for tie-breaks: role+core then id
        tie = (comps.get("role_relevance", 0) * 0.6
               + comps.get("core_requirements", 0) * 0.4)

        item = (score, tie, cid, cand, comps, evidence)
        if len(heap) < KEEP:
            heapq.heappush(heap, item)
        elif (score, tie) > (heap[0][0], heap[0][1]):
            heapq.heapreplace(heap, item)

        if args.progress and n % 10000 == 0:
            print(f"  ...{n} scored ({time.time()-t0:.1f}s)", file=sys.stderr)

    # sort best-first by RAW score: score desc, tie desc, candidate_id asc
    ranked = sorted(heap, key=lambda x: (-x[0], -x[1], x[2]))[:TOPN]

    # normalize raw scores into a clean [0.05, 0.99] display column (shared with
    # the sandbox via ranker.scoring.normalize_display).
    from ranker.scoring import normalize_display
    norms = normalize_display([r[0] for r in ranked])
    rows = []
    for (score, tie, cid, cand, comps, evidence), norm in zip(ranked, norms):
        reasoning = build_reasoning(cand, comps, evidence)
        rows.append([cid, norm, reasoning])

    # Validator rule: among rows with the SAME (rounded) score, candidate_id must
    # be ascending. Re-sort by (score desc, candidate_id asc) and assign ranks.
    # Distinct scores keep their order; only true display-ties are reordered.
    rows.sort(key=lambda r: (-r[1], r[0]))

    out = Path(args.out)
    with open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["candidate_id", "rank", "score", "reasoning"])
        for i, (cid, score, reasoning) in enumerate(rows):
            w.writerow([cid, i + 1, f"{score:.4f}", reasoning])

    print(f"Scored {n} candidates → wrote {len(rows)} rows to {out} "
          f"in {time.time()-t0:.1f}s", file=sys.stderr)


if __name__ == "__main__":
    main()
