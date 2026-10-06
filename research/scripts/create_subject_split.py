"""Freeze an eligible cohort split once enough nonpilot subjects are processed."""

import argparse
import json
from pathlib import Path

import pandas as pd

from intraop.evaluation.subject_splits import make_subject_split, persist_split_manifest
from intraop.logging import configure_logging


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument(
        "--pilot-manifest",
        type=Path,
        required=True,
        help="Preserve development subjects from the inspected pilot",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    configure_logging()
    metadata = pd.read_csv(args.metadata, dtype={"subject_id": str, "case_id": str})
    episodes = pd.read_csv(args.episodes, dtype={"subject_id": str, "evaluable": "boolean"})
    subjects = sorted(metadata.subject_id.unique())
    positive = set(episodes.loc[episodes.evaluable.fillna(False), "subject_id"])
    status = {s: s in positive for s in subjects}
    prior = json.loads(args.pilot_manifest.read_text())
    development = [s for s in prior["development_subjects_pinned_to_training"] if s in subjects]
    result = make_subject_split(
        subjects, seed=args.seed, episode_status=status, development_subjects=development
    )
    result["population_policy"] = "actual_eligible_window_cohort"
    persist_split_manifest(args.output, result)
    print(
        json.dumps(
            {
                "subject_counts": result["counts"],
                "stratified": result["stratified"],
                "fallback_reason": result["stratification_fallback_reason"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
