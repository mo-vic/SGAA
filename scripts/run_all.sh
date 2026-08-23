#!/usr/bin/env bash
# run_all.sh — run the full SGAA pipeline (verify -> experiments -> figures -> analyze).
# Usage: ./scripts/run_all.sh
set -euo pipefail
cd "$(dirname "$0")/.."                     # repository root
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}src"
python -m sgaa run-all
