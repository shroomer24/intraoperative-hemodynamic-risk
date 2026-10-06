#!/bin/bash
set +x
set -euo pipefail
if [ "$#" -ne 0 ]; then
    echo "Calibration aggregation accepts no arguments." >&2
    exit 2
fi
AGGREGATION_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CALIBRATION_DIR="${AGGREGATION_DIR}/../locked_execution/calibration"
if [ -e "$CALIBRATION_DIR/aggregation/attempt_1" ] || [ -L "$CALIBRATION_DIR/aggregation/attempt_1" ] || [ -e "$CALIBRATION_DIR/completion_receipt.json" ]; then
    echo "Prior aggregate execution retained. No duplicate or retry." >&2
    exit 2
fi
cd "$AGGREGATION_DIR"
export INTRAOP_AGGREGATION_EXECUTION=normal_terminal
exec /Library/Frameworks/Python.framework/Versions/3.14/bin/python3 -I -u "$AGGREGATION_DIR/run_aggregation.py"
