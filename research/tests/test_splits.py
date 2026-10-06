import unittest

import numpy as np
import pandas as pd

from intraop.evaluation.splits import grouped_temporal_split, patient_group_split


class SplitTests(unittest.TestCase):
    def test_random_holdout_is_disjoint_complete_and_reproducible(self):
        groups = pd.Series(["g1", "g1", "g2", "g3", "g4", "g4"], index=[10, 20, 30, 40, 50, 60])
        first = patient_group_split(groups, test_fraction=0.25, seed=42)
        second = patient_group_split(groups, test_fraction=0.25, seed=42)
        np.testing.assert_array_equal(first.train, second.train)
        np.testing.assert_array_equal(first.test, second.test)
        self.assertFalse(set(groups.iloc[first.train]) & set(groups.iloc[first.test]))
        self.assertEqual(sorted([*first.train, *first.test]), list(range(len(groups))))
        self.assertEqual(groups.iloc[first.test].nunique(), 1)

    def test_group_assignment_is_independent_of_row_order(self):
        groups = pd.Series(["g3", "g1", "g2", "g4", "g1"])
        split = patient_group_split(groups, test_fraction=0.5, seed=7)
        reordered = groups.iloc[::-1]
        other = patient_group_split(reordered, test_fraction=0.5, seed=7)
        self.assertEqual(set(groups.iloc[split.test]), set(reordered.iloc[other.test]))

    def test_invalid_random_split_inputs(self):
        cases = [
            (pd.Series([], dtype=str), 0.5, 1),
            (pd.Series(["g1", "g1"]), 0.5, 1),
            (pd.Series(["g1", None]), 0.5, 1),
            (pd.Series(["g1", "g2"]), 0, 1),
            (pd.Series(["g1", "g2"]), 0.5, True),
        ]
        for groups, fraction, seed in cases:
            with self.subTest(), self.assertRaises(ValueError):
                patient_group_split(groups, test_fraction=fraction, seed=seed)

    def temporal_fixture(self):
        groups = pd.Series(["early", "early", "later", "later"], index=[10, 20, 30, 40])
        start = pd.Series(
            pd.to_datetime(["2000-01-01", "2000-01-02", "2000-01-04", "2000-01-05"]),
            index=groups.index,
        )
        end = start + pd.Timedelta(hours=1)
        return groups, start, end

    def test_temporal_split_has_patient_and_interval_separation(self):
        groups, start, end = self.temporal_fixture()
        cutoff = pd.Timestamp("2000-01-03")
        split = grouped_temporal_split(groups, start, end, cutoff=cutoff)
        np.testing.assert_array_equal(split.train, [0, 1])
        np.testing.assert_array_equal(split.test, [2, 3])
        self.assertFalse(set(groups.iloc[split.train]) & set(groups.iloc[split.test]))
        self.assertLess(end.iloc[split.train].max(), start.iloc[split.test].min())

    def test_crossing_patient_or_label_dependency_interval_raises(self):
        groups, start, end = self.temporal_fixture()
        groups.iloc[2] = "early"
        with self.assertRaises(ValueError):
            grouped_temporal_split(groups, start, end, cutoff=pd.Timestamp("2000-01-03"))
        groups, start, end = self.temporal_fixture()
        end.iloc[1] = pd.Timestamp("2000-01-03")
        with self.assertRaises(ValueError):
            grouped_temporal_split(groups, start, end, cutoff=pd.Timestamp("2000-01-03"))

    def test_test_interval_can_start_at_cutoff(self):
        groups, start, end = self.temporal_fixture()
        split = grouped_temporal_split(groups, start, end, cutoff=start.iloc[2])
        np.testing.assert_array_equal(split.test, [2, 3])

    def test_misaligned_and_empty_temporal_cohorts_rejected(self):
        groups, start, end = self.temporal_fixture()
        with self.assertRaises(ValueError):
            grouped_temporal_split(
                groups, start.reset_index(drop=True), end, cutoff=pd.Timestamp("2000-01-03")
            )
        with self.assertRaises(ValueError):
            grouped_temporal_split(groups, start, end, cutoff=pd.Timestamp("1999-01-01"))
