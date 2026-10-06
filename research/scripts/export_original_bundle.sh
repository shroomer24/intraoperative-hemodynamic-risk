#!/usr/bin/env bash
# Path-only adapter. Requires the retained sealed scientific bundle, never refits.
set -euo pipefail
if [[ $# -ne 2 ]]; then
  echo "Usage: export_original_bundle.sh SCIENTIFIC_BUNDLE NEW_EXPORT_APP" >&2
  exit 2
fi
here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
scientific_bundle="$1"
export_app="$2"
[[ ! -e "$export_app" ]] || { echo "Export destination exists; stopped" >&2; exit 2; }
mkdir -p "$export_app/tools"
cp "$here/export/export_replay.py" "$here/export/validate_replay_export.py" "$here/export/replay_schema.json" "$export_app/tools/"
INTRAOP_SCIENTIFIC_REPO="$scientific_bundle" INTRAOP_EXPORT_APP="$export_app"   python -B "$export_app/tools/export_replay.py"
