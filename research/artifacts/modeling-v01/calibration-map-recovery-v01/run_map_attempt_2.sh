#!/bin/bash
set +x
set -euo pipefail
if [ "$#" -ne 0 ]; then
    echo "MAP Attempt 2 accepts no arguments." >&2
    exit 2
fi
RECOVERY_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ATTEMPT_DIR="${RECOVERY_DIR}/../locked_execution/calibration/attempts/tabpfn_map/attempt_2"
if [ -e "$ATTEMPT_DIR" ] || [ -L "$ATTEMPT_DIR" ]; then
    echo "Prior Attempt 2 retained. No duplicate or retry." >&2
    exit 2
fi
set -a
source "$HOME/.tabpfn.env" >/dev/null 2>&1
set +a
set +x
cd "$RECOVERY_DIR"
export INTRAOP_MAP_RECOVERY_EXECUTION=normal_terminal
exec /Library/Frameworks/Python.framework/Versions/3.14/bin/python3 -I -u "$RECOVERY_DIR/recover_map.py"
