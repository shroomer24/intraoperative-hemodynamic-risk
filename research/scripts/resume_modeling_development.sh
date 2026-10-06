#!/usr/bin/env bash
# Hosted Phase 1 continuation: normal local terminal only.
set +x
set -euo pipefail
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
python_executable='/Library/Frameworks/Python.framework/Versions/3.14/bin/python3'
set -a
source "$HOME/.tabpfn.env"
set +a
set +x
cd "$repo_dir"
PYTHONPATH="$repo_dir/../../work" "$python_executable" -u scripts/modeling_development_resume.py "$@"
