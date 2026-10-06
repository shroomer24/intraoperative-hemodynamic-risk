#!/bin/bash
set +x
set -euo pipefail
if [ "$#" -ne 0 ]; then
  echo "No TEST configuration overrides are permitted." >&2
  exit 2
fi
CAP_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
if [ -e "$CAP_DIR/../locked_execution/test" ] || [ -L "$CAP_DIR/../locked_execution/test" ]; then
  echo "Prior TEST state retained. No rerun or automatic retry." >&2
  exit 2
fi
set -a
source "$HOME/.tabpfn.env" >/dev/null 2>&1
set +a
cd "$CAP_DIR"
export INTRAOP_FINAL_TEST_EXECUTION=normal_terminal
exec /Library/Frameworks/Python.framework/Versions/3.14/bin/python3 -I -u "$CAP_DIR/run_final_test.py"
