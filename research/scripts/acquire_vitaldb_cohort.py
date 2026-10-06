"""Freeze and acquire an outcome-blind deterministic prefix (at most 150 cases)."""

import argparse
from pathlib import Path

from intraop.data.cohort_acquisition import HARD_CAP_BYTES, acquire_cohort


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/raw/vitaldb-1.0.0"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/vitaldb-cohort-v01"))
    parser.add_argument("--pilot-cases", nargs="+", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target", type=int, default=150)
    parser.add_argument("--max-new-bytes", type=int, default=HARD_CAP_BYTES)
    parser.add_argument("--max-seconds", type=int, default=1800)
    parser.add_argument(
        "--resume-from",
        type=Path,
        help="Finalized parent acquisition manifest; initialize a separate checkpoint",
    )
    args = parser.parse_args()
    result = acquire_cohort(
        args.source,
        args.output,
        args.pilot_cases,
        seed=args.seed,
        target=args.target,
        max_new_bytes=args.max_new_bytes,
        max_seconds=args.max_seconds,
        resume_from=args.resume_from,
    )
    print(f"Stop: {result['stop_reason']}; completed {len(result['completed_case_ids'])}")


if __name__ == "__main__":
    main()
