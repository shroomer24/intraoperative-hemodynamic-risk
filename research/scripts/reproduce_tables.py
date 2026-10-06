"""Post-publication path adapter calling the original frozen table pipeline.

An explicit private roster is mandatory. This never infers or replaces the
historical pilot selection, and refuses to overwrite any destination.
"""

from locked_bootstrap import REPO  # isort: skip
import argparse
import json
from pathlib import Path

from intraop.data.acquisition import acquire_subset, sha256_file
from intraop.data.pipeline import run_tables


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--selection", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--acquire", action="store_true")
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError("Destination exists; original or partial evidence must remain intact")
    selection = json.loads(a.selection.read_text())
    if set(selection) != {"completed_case_ids", "development_subjects_pinned_to_training"}:
        raise ValueError("Exact historical private selection schema required")
    cases = selection["completed_case_ids"]
    subjects = selection["development_subjects_pinned_to_training"]
    if len(cases) != 150 or len(set(cases)) != 150 or len(subjects) != 10:
        raise ValueError("Exact 150-case roster and ten inspected pilot subjects required")
    if a.acquire:
        for start in range(0, len(cases), 20):
            acquire_subset(a.source, cases[start : start + 20], max_total_bytes=8_000_000_000)
    run_tables(a.source, a.output, cases, seed=42, realized_development_subjects=subjects)
    expected = json.loads((REPO / "reference/table_manifest.json").read_text())["tables"]
    matches = {name: sha256_file(a.output / name) == digest for name, digest in expected.items()}
    (a.output / "public_replication_comparison.json").write_text(
        json.dumps(
            {
                "scope": "NEW_REPLICATION_OUTPUT_NOT_ORIGINAL_EVIDENCE",
                "table_hash_matches": matches,
                "scientific_functions": "original frozen run_tables; byte-identical implementation",
                "original_sources_modified": False,
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps({"table_hash_matches": matches, "original_sources_modified": False}))
    if not all(matches.values()):
        raise ValueError("Replication differs; retain evidence, do not relabel it as original")


if __name__ == "__main__":
    main()
