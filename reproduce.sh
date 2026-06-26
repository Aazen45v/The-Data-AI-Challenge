#!/usr/bin/env bash
# reproduce.sh — one command to validate the toolchain and produce submission.csv.
# No network, CPU only. Pass the path to candidates.jsonl as $1
# (defaults to ./candidates.jsonl).
set -euo pipefail

CANDIDATES="${1:-./candidates.jsonl}"
OUT="${2:-./submission.csv}"

echo "[1/3] Running test suite (no network, seconds)..."
python3 test_rank.py

echo "[2/3] Ranking $CANDIDATES ..."
python3 rank.py --candidates "$CANDIDATES" --out "$OUT" --progress

echo "[3/3] Validating $OUT ..."
python3 validate_submission.py "$OUT"

echo "Done. Wrote $OUT"
