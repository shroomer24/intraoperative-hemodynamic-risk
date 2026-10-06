"""Process acquired prefix, freeze eligible-subject split, and generate human audits."""

import argparse
from pathlib import Path

from intraop.data.cohort import run_cohort, verify_artifacts
from intraop.logging import configure_logging


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/raw/vitaldb-1.0.0"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/vitaldb-cohort-v01"))
    parser.add_argument(
        "--pilot-manifest", type=Path, default=Path("artifacts/vitaldb-pilot/split_manifest.json")
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument(
        "--reuse-waveform-audit-from",
        type=Path,
        help="Verify and reuse the parent pilot waveform audit without new adjudication",
    )
    args = parser.parse_args()
    configure_logging()
    if args.verify_only:
        print(verify_artifacts(args.output))
    else:
        report = run_cohort(
            args.source,
            args.output,
            args.pilot_manifest,
            seed=args.seed,
            waveform_parent=args.reuse_waveform_audit_from,
        )
        print(
            f"Realized {report['unique_eligible_subjects']} eligible subjects, "
            f"{report['eligible_windows']} windows. No model work performed."
        )


if __name__ == "__main__":
    main()
