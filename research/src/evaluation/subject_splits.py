"""Deterministic four-way subject partitions with explicit stratification fallback."""

import hashlib
import json
import logging
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

PARTITIONS = ("training", "tuning", "calibration", "test")
FRACTIONS = np.array([0.60, 0.15, 0.10, 0.15])


def _counts(number: int) -> np.ndarray:
    target = number * FRACTIONS
    result = np.floor(target).astype(int)
    remainder = number - result.sum()
    for index in np.argsort(-(target - result), kind="stable")[:remainder]:
        result[index] += 1
    return result


def make_subject_split(
    subjects: list[str],
    *,
    seed: int,
    episode_status: dict[str, bool] | None = None,
    development_subjects: list[str] | None = None,
) -> dict:
    """Largest-remainder subject counts; pilot inspection subjects forced to train.

    Label status must be complete for the supplied population. Stratification
    uses confirmed AND evaluable episode status, if each stratum has sufficient
    subjects for all four partitions. Missing/low counts yield a logged fallback.
    Fixed pilot subjects are reserved first and never occupy final test.
    """
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    if any(not isinstance(s, str) or not s for s in subjects):
        raise ValueError("Subject identifiers must be nonempty strings")
    unique = sorted(set(subjects))
    counts = _counts(len(unique))
    if not len(unique) or (counts == 0).any():
        raise ValueError("Too few subjects to form four nonempty partitions")
    development = sorted(set(development_subjects or []))
    if set(development) - set(unique) or len(development) > counts[0]:
        raise ValueError("Development subjects must fit within the population's training quota")
    rng = np.random.default_rng(seed)
    assignments = {subject: "training" for subject in development}
    remaining = [subject for subject in unique if subject not in assignments]
    remaining_counts = counts.copy()
    remaining_counts[0] -= len(development)
    stratified = False
    reason = None
    if episode_status is None or set(unique) - set(episode_status):
        reason = "Episode status unavailable for the complete split population"
    elif any(type(episode_status[s]) is not bool for s in unique):
        raise ValueError("Subject-level episode status must be Boolean")
    elif development:
        reason = (
            "Development subjects are pinned; preserve exact quotas with unstratified allocation"
        )
    else:
        positives = [s for s in remaining if episode_status[s]]
        negatives = [s for s in remaining if not episode_status[s]]
        positive_counts = _counts(len(positives))
        negative_counts = counts - positive_counts
        if (positive_counts == 0).any() or (negative_counts == 0).any():
            reason = "Episode strata too small to populate every partition"
        else:
            for stratum, quotas in ((positives, positive_counts), (negatives, negative_counts)):
                shuffled = rng.permutation(stratum)
                offset = 0
                for partition, quota in zip(PARTITIONS, quotas, strict=True):
                    for subject in shuffled[offset : offset + quota]:
                        assignments[str(subject)] = partition
                    offset += quota
            stratified = True
    if not stratified:
        logging.getLogger("intraop.evaluation").warning(
            "Subject stratification fallback: %s", reason
        )
        shuffled = rng.permutation(remaining)
        offset = 0
        for partition, quota in zip(PARTITIONS, remaining_counts, strict=True):
            for subject in shuffled[offset : offset + quota]:
                assignments[str(subject)] = partition
            offset += quota
    ordered = dict(sorted(assignments.items()))
    return {
        "seed": seed,
        "grouping_unit": "subject_id",
        "requested_fractions": dict(zip(PARTITIONS, FRACTIONS.tolist(), strict=True)),
        "counts": dict(Counter(ordered.values())),
        "stratified": stratified,
        "stratification_fallback_reason": reason,
        "development_subjects_pinned_to_training": development,
        "population_sha256": hashlib.sha256("\n".join(unique).encode()).hexdigest(),
        "subject_to_partition": ordered,
    }


def persist_split_manifest(path: Path, manifest: dict) -> None:
    if path.exists():
        existing = json.loads(path.read_text())
        if existing != manifest:
            raise ValueError("Refusing to overwrite a different frozen split manifest")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2) + "\n")


def partition_positions(metadata: pd.DataFrame, manifest: dict) -> dict[str, np.ndarray]:
    partitions = metadata.subject_id.map(manifest["subject_to_partition"])
    if partitions.isna().any():
        raise ValueError("A dataset subject is absent from the frozen split manifest")
    return {name: np.flatnonzero((partitions == name).to_numpy()) for name in PARTITIONS}
