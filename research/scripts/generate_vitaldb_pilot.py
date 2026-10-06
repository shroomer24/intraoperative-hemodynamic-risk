"""Run local ingestion, labels, features, reports, and development-safe subject splits."""

import argparse
from pathlib import Path

from intraop.data.pipeline import run_pilot
from intraop.logging import configure_logging


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-ids", nargs="+", type=int, required=True)
    parser.add_argument("--source", type=Path, default=Path("data/raw/vitaldb-1.0.0"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/vitaldb-pilot"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    logger = configure_logging()
    result = run_pilot(
        args.source,
        args.output,
        args.case_ids,
        seed=args.seed,
    )
    logger.info(
        "Pilot: %d/%d cases parsed, %d eligible windows; see %s",
        result["cases_parsed"],
        result["cases_attempted"],
        result["eligible_windows"],
        args.output,
    )
    if not result["cases_parsed"]:
        parser.exit(1, "No cases parsed; inspect case_report.csv and the report for errors.\n")


if __name__ == "__main__":
    main()
