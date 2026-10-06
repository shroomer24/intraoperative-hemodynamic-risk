#!/usr/bin/env bash
# Architectural approval must exist before credential sourcing or any hosted command.
set +x
set -euo pipefail
if [ "$#" -ne 1 ] || { [ "$1" != calibration ] && [ "$1" != test ]; }; then
    printf '%s\n' 'Usage: run_locked_evaluation.sh calibration|test (no overrides)' >&2
    exit 2
fi
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
phase="$1"
if [ ! -f "$repo_dir/artifacts/modeling-v01/locked_execution/approvals/$phase.json" ]; then
    printf '%s\n' 'Architectural review approval required; no execution started.' >&2
    exit 2
fi
set -a
if ! source "$HOME/.tabpfn.env" >/dev/null 2>&1; then
    set +a
    set +x
    printf '%s\n' 'Secure credential initialization failed; no execution started.' >&2
    exit 2
fi
set +a
set +x
cd "$repo_dir"
INTRAOP_LOCKED_EXTERNAL_EXECUTION=normal_terminal \
    /Library/Frameworks/Python.framework/Versions/3.14/bin/python3 -u \
    scripts/run_locked_evaluation.py --phase "$phase"
