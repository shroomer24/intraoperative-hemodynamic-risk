import json
import tempfile
import unittest
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from intraop.evaluation.subject_splits import (
    make_subject_split,
    partition_positions,
    persist_split_manifest,
)


class SubjectSplitTests(unittest.TestCase):
    def test_deterministic_exact_counts_and_no_subject_overlap(self):
        subjects = [f"s{i:03d}" for i in range(100)]
        first = make_subject_split(subjects, seed=42)
        second = make_subject_split(subjects[::-1] + subjects[:5], seed=42)
        self.assertEqual(first, second)
        self.assertEqual(
            first["counts"], {"training": 60, "tuning": 15, "calibration": 10, "test": 15}
        )
        groups = {
            p: {s for s, partition in first["subject_to_partition"].items() if p == partition}
            for p in first["counts"]
        }
        for left, right in combinations(groups.values(), 2):
            self.assertFalse(left & right)

    def test_repeated_operations_and_windows_remain_in_one_partition(self):
        manifest = make_subject_split([f"s{i}" for i in range(20)], seed=7)
        frame = pd.DataFrame(
            {"subject_id": ["s1", "s1", "s1", "s2"], "case_id": ["c1", "c1", "c2", "c3"]},
            index=[10, 20, 30, 40],
        )
        partitions = partition_positions(frame, manifest)
        location = [p for p, indices in partitions.items() if 0 in indices][0]
        self.assertTrue(set([0, 1, 2]) <= set(partitions[location]))
        np.testing.assert_array_equal(
            np.sort(np.concatenate(list(partitions.values()))), [0, 1, 2, 3]
        )

    def test_practical_episode_stratification(self):
        subjects = [f"s{i}" for i in range(100)]
        status = {s: i < 20 for i, s in enumerate(subjects)}
        manifest = make_subject_split(subjects, seed=42, episode_status=status)
        self.assertTrue(manifest["stratified"])
        for p in manifest["counts"]:
            assigned = [
                s for s, partition in manifest["subject_to_partition"].items() if partition == p
            ]
            self.assertTrue(any(status[s] for s in assigned))
            self.assertTrue(any(not status[s] for s in assigned))

    def test_low_count_stratification_falls_back_with_logged_reason(self):
        subjects = [f"s{i}" for i in range(20)]
        status = {s: s == "s0" for s in subjects}
        with self.assertLogs("intraop.evaluation", level="WARNING") as logs:
            first = make_subject_split(subjects, seed=42, episode_status=status)
        self.assertFalse(first["stratified"])
        self.assertIn("too small", first["stratification_fallback_reason"])
        self.assertTrue(logs.output)
        self.assertEqual(first, make_subject_split(subjects, seed=42, episode_status=status))

    def test_pilot_inspection_subjects_are_reserved_for_training(self):
        manifest = make_subject_split(
            [f"s{i}" for i in range(20)], seed=42, development_subjects=["s1", "s2"]
        )
        self.assertEqual(manifest["subject_to_partition"]["s1"], "training")
        self.assertEqual(manifest["subject_to_partition"]["s2"], "training")

    def test_frozen_manifest_cannot_be_reassigned_and_unknown_subject_rejected(self):
        manifest = make_subject_split([f"s{i}" for i in range(20)], seed=42)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "split.json"
            persist_split_manifest(path, manifest)
            persist_split_manifest(path, manifest)
            self.assertEqual(json.loads(path.read_text()), manifest)
            with self.assertRaises(ValueError):
                persist_split_manifest(
                    path, make_subject_split([f"s{i}" for i in range(20)], seed=1)
                )
        with self.assertRaises(ValueError):
            partition_positions(pd.DataFrame({"subject_id": ["unknown"]}), manifest)

    def test_small_population_or_excess_pinned_subjects_fail_clearly(self):
        with self.assertRaises(ValueError):
            make_subject_split(["s1", "s2"], seed=42)
        with self.assertRaises(ValueError):
            make_subject_split(
                [f"s{i}" for i in range(10)],
                seed=42,
                development_subjects=[f"s{i}" for i in range(10)],
            )
