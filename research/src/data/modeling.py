"""Read frozen partition rows without parsing held-out predictor/label values."""

import csv
import hashlib
import io
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from intraop.data.datasets import ClassificationDataset


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def write_json(path: Path, value: dict, *, exclusive: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x" if exclusive else "w") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


class FrozenPartitions:
    """Metadata routes rows; excluded feature/label lines are never parsed.

    Hashing frozen files checks bytes, not scientific values. The CSV format is
    the authoritative numeric one-row-per-line export. No cohort regeneration.
    """

    def __init__(self, cohort: Path, modeling: Path):
        self.cohort = cohort
        self.modeling = modeling
        self.split = read_json(cohort / "realized_split_manifest.json")
        self.features = read_json(cohort / "feature_schema.json")["feature_names"]
        self.feature_sets = read_json(modeling / "feature_sets.json")
        if len(self.features) != 74 or self.feature_sets["full"] != self.features:
            raise ValueError("Frozen 74-feature contract differs")
        expected_map = [name for name in self.features if name.startswith("map_")]
        if len(expected_map) != 18 or self.feature_sets["map_only"] != expected_map:
            raise ValueError("Frozen ordered 18-feature MAP-only contract differs")
        if self.feature_sets["current_map"] != ["map_latest"]:
            raise ValueError("Current-MAP contract differs")
        self.access_log = []

    def verify_integrity(self) -> dict:
        manifest = read_json(self.cohort / "table_manifest.json")
        for name, expected in manifest["tables"].items():
            if sha256(self.cohort / name) != expected:
                raise ValueError(f"Frozen artifact hash differs: {name}")
        integrity = read_json(self.cohort / "integrity_report.json")
        for flag in ("table_hashes_verified", "eligible_anchor_alignment_verified"):
            if integrity[flag] is not True:
                raise ValueError(f"Frozen integrity flag failed: {flag}")
        if integrity["subject_overlap"] or not integrity["forbidden_predictors_absent"]:
            raise ValueError("Frozen integrity overlap/predictor failure")
        return {"all_table_manifest_hashes_verified": True, "test_values_parsed": False}

    def load(self, partition: str) -> ClassificationDataset:
        if partition not in {"training", "tuning", "calibration", "test"}:
            raise ValueError("Unknown partition")
        if partition in {"calibration", "test"}:
            raise PermissionError(
                "Held-out values require the approved locked execution capability"
            )
        data = _load_partition(self.cohort, self.split, self.features, partition)
        self.access_log.append(partition)
        return data

    def validate_lock(self) -> dict:
        from intraop.evaluation.locked_execution import validate_lock

        return validate_lock(self.modeling)[0]


def predictor_array(data: ClassificationDataset, names: list[str]) -> np.ndarray:
    if len(set(names)) != len(names) or set(names) - set(data.X.columns):
        raise ValueError("Unknown/duplicate predictor")
    if set(names) & set(data.metadata.columns):
        raise ValueError("Metadata cannot enter X")
    return data.X.loc[:, names].to_numpy(dtype=float, copy=True)


def _load_partition(cohort: Path, split: dict, features: list[str], partition: str):
    """Private row reader: public entrypoints must authorize partition before I/O."""
    chosen, metadata_lines = [], []
    with (cohort / "metadata.csv").open() as stream:
        metadata_header = next(stream)
        routing_columns = next(csv.reader([metadata_header]))
        subject_column = routing_columns.index("subject_id")
        allowed_subjects = [
            subject
            for subject, assignment in split["subject_to_partition"].items()
            if assignment == partition
        ]
        alternatives = "|".join(re.escape(subject) for subject in allowed_subjects)
        route = re.compile(
            r"^(?:[^,]*,){" + str(subject_column) + r"}(?:" + alternatives + r")(?=,)"
        )
        for index, line in enumerate(stream):
            # Match the permitted roster without extracting excluded identifiers
            # or parsing any excluded metadata fields. Full-file hashes protect
            # the one-record-per-line frozen CSV and complete subject roster.
            if route.match(line):
                chosen.append(index)
                metadata_lines.append(line)
        metadata_count = index + 1
    metadata = list(csv.DictReader(io.StringIO(metadata_header + "".join(metadata_lines))))
    positions = set(chosen)
    frames = []
    for name in ("features.csv", "labels.csv"):
        with (cohort / name).open() as stream:
            header = next(stream)
            lines = []
            for index, line in enumerate(stream):
                if index in positions:
                    lines.append(line)
            if index + 1 != metadata_count:
                raise ValueError("Frozen table row counts are not aligned")
        frame = pd.read_csv(io.StringIO(header + "".join(lines)))
        frame.index = pd.Index(chosen, name="source_row")
        frames.append(frame)
    X, labels = frames
    if X.columns.tolist() != features or labels.columns.tolist() != ["label"]:
        raise ValueError("Table schema differs")
    m = pd.DataFrame(metadata, index=X.index)
    for name in m.columns:
        if name not in {"subject_id", "case_id", "window_id"}:
            m[name] = pd.to_numeric(m[name], errors="coerce")
    data = ClassificationDataset(X, labels.label, m)
    if data.groups.nunique() != split["counts"][partition]:
        raise ValueError("Frozen subject count differs")
    stats = read_json(cohort / "integrity_report.json")["split_statistics"]
    expected = stats["partitions"][partition]
    if len(X) != expected["eligible_windows"] or int(data.y.sum()) != expected["positive_windows"]:
        raise ValueError("Frozen partition counts differ")
    return data


class DevelopmentPartitions:
    """Development capability: only TRAINING/TUNING, with no lock override.

    Contains no TEST/CALIBRATION-loading method and never inherits the broader
    FrozenPartitions interface. A rejected partition fails before table I/O.
    """

    def __init__(self, cohort: Path, modeling: Path, expected_manifest_hashes: dict):
        for name, digest in expected_manifest_hashes.items():
            if sha256(cohort / name) != digest:
                raise ValueError(f"Frozen checkpoint manifest differs: {name}")
        contract = FrozenPartitions(cohort, modeling)
        self.cohort = cohort
        self.features = contract.features
        self.feature_sets = contract.feature_sets
        self.split = contract.split
        self.access_log = []
        # These hashes check bytes, not held-out values. Only three model tables
        # are relevant; audit/episode files are never opened by development.
        table_manifest = read_json(cohort / "table_manifest.json")
        for name in ("features.csv", "labels.csv", "metadata.csv"):
            if sha256(cohort / name) != table_manifest["tables"][name]:
                raise ValueError(f"Frozen model table hash differs: {name}")
        assignment = self.split["subject_to_partition"]
        if len(assignment) != 150 or self.split["counts"] != {
            "training": 90,
            "tuning": 23,
            "calibration": 15,
            "test": 22,
        }:
            raise ValueError("Frozen split population/counts differ")

    def load(self, partition: str) -> ClassificationDataset:
        if partition not in ("training", "tuning"):
            raise PermissionError("Development permits only TRAINING and TUNING")
        data = _load_partition(self.cohort, self.split, self.features, partition)
        self.access_log.append(partition)
        return data
