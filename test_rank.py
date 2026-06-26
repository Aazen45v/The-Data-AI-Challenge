#!/usr/bin/env python3
"""
test_rank.py — fast, no-network tests that protect the Stage-3 reproduction.

Run:  python test_rank.py
Exits non-zero on any failure. Uses only sample_candidates.json (no full pool,
no network), so it runs in seconds inside the sandbox.
"""
import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).parent
SAMPLE = ROOT / "sample_candidates.json"


def _load_sample_as_jsonl(dst):
    data = json.load(open(SAMPLE))
    recs = data if isinstance(data, list) else data.get("candidates", [])
    with open(dst, "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    return len(recs)


def run():
    fails = []

    with tempfile.TemporaryDirectory() as td:
        jl = Path(td) / "sample.jsonl"
        out = Path(td) / "sub.csv"
        n = _load_sample_as_jsonl(jl)

        r = subprocess.run(
            [sys.executable, str(ROOT / "rank.py"),
             "--candidates", str(jl), "--out", str(out)],
            capture_output=True, text=True)
        if r.returncode != 0:
            fails.append(f"rank.py exited {r.returncode}: {r.stderr[-400:]}")
            _report(fails); return

        rows = list(csv.DictReader(open(out)))

        # 1) header + row count (<=100, or n if fewer candidates)
        expected = min(n, 100)
        if len(rows) != expected:
            fails.append(f"expected {expected} rows, got {len(rows)}")

        # 2) ranks 1..k unique & contiguous
        ranks = [int(r["rank"]) for r in rows]
        if sorted(ranks) != list(range(1, len(rows) + 1)):
            fails.append("ranks are not the contiguous set 1..k")

        # 3) candidate_ids unique and well-formed
        ids = [r["candidate_id"] for r in rows]
        if len(set(ids)) != len(ids):
            fails.append("duplicate candidate_id in output")
        import re
        if any(not re.match(r"^CAND_[0-9]{7}$", i) for i in ids):
            fails.append("malformed candidate_id")

        # 4) score strictly non-increasing with rank
        scores = [float(r["score"]) for r in rows]
        if any(scores[i] < scores[i + 1] - 1e-9 for i in range(len(scores) - 1)):
            fails.append("score increases as rank increases (must be non-increasing)")

        # 5) scores differentiate (not all identical)
        if len(set(round(s, 4) for s in scores)) < max(2, len(rows) // 5):
            fails.append("scores barely differentiate candidates")

        # 6) reasoning present, non-identical, plausibly specific
        reasons = [r["reasoning"] for r in rows]
        if any(not x.strip() for x in reasons):
            fails.append("empty reasoning present")
        if len(set(reasons)) < len(reasons):
            fails.append("duplicate reasoning strings present")

        # 7) determinism: same input -> identical output
        out2 = Path(td) / "sub2.csv"
        subprocess.run([sys.executable, str(ROOT / "rank.py"),
                        "--candidates", str(jl), "--out", str(out2)],
                       capture_output=True, text=True)
        if open(out).read() != open(out2).read():
            fails.append("non-deterministic output across runs")

    _report(fails)


def _report(fails):
    if fails:
        print("FAILED:")
        for f in fails:
            print("  -", f)
        sys.exit(1)
    print("All tests passed.")


if __name__ == "__main__":
    run()
