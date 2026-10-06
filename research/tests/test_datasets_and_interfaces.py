import unittest

import numpy as np
import pandas as pd

from intraop.data.datasets import ClassificationDataset, InMemoryCaseDataset, TimeSeriesSchema
from intraop.features.base import PlaceholderFeatureExtractor, Window
from intraop.models.base import PlaceholderClassifier


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.schema = TimeSeriesSchema("case", "subject", "time_seconds", ("value",))
        self.frame = pd.DataFrame(
            {
                "case": ["c1", "c1", "c2"],
                "subject": ["s1", "s1", "s1"],
                "time_seconds": [0.0, 1.0, 0.0],
                "value": [1.0, np.nan, 3.0],
            }
        )

    def test_patient_reader_preserves_missingness_and_defensive_copies(self):
        reader = InMemoryCaseDataset(self.frame, self.schema)
        self.frame.loc[0, "value"] = 99
        first = reader.get_case("c1")
        self.assertEqual(first.iloc[0]["value"], 1)
        self.assertTrue(pd.isna(first.iloc[1]["value"]))
        first.loc[0, "value"] = 88
        self.assertEqual(reader.get_case("c1").iloc[0]["value"], 1)
        self.assertEqual(reader.case_ids, ("c1", "c2"))
        with self.assertRaises(KeyError):
            reader.get_case("unknown")

    def test_reader_rejects_missing_columns_ids_unparsed_time_and_unsorted_records(self):
        invalid = [
            self.frame.drop(columns="value"),
            self.frame.assign(case=["c1", None, "c2"]),
            self.frame.assign(time_seconds=["unparsed"] * 3),
            self.frame.iloc[[1, 0, 2]],
            self.frame.assign(value=["unconverted"] * 3),
            self.frame.assign(time_seconds=pd.to_datetime(["2000-01-01"] * 3)),
            self.frame.assign(subject=["s1", "s2", "s1"]),
        ]
        for frame in invalid:
            with self.subTest(), self.assertRaises(ValueError):
                InMemoryCaseDataset(frame, self.schema)

    def test_empty_and_duplicate_schema_rejected(self):
        with self.assertRaises(ValueError):
            InMemoryCaseDataset(self.frame.iloc[:0], self.schema)
        with self.assertRaises(ValueError):
            InMemoryCaseDataset(
                self.frame, TimeSeriesSchema("case", "subject", "time_seconds", ("case",))
            )

    def tabular(self, **overrides):
        index = pd.Index([10, 20, 30])
        metadata = pd.DataFrame(
            {
                "case_id": ["c1", "c1", "c2"],
                "subject_id": ["s1", "s1", "s1"],
                "anchor_time_seconds": [300.0, 360.0, 300.0],
                "history_start_seconds": [0.0, 60.0, 0.0],
                "history_end_seconds": [300.0, 360.0, 300.0],
                "future_observation_end_seconds": [660.0, 720.0, 660.0],
            },
            index=index,
        )
        fields = {
            "X": pd.DataFrame({"f1": [1.0, np.nan, 3.0]}, index=index),
            "y": pd.Series([0, 1, 0], index=index),
            "metadata": metadata,
        }
        return ClassificationDataset(**{**fields, **overrides})

    def test_tabular_subset_uses_positions_and_preserves_alignment(self):
        dataset = self.tabular().subset(np.array([2, 0]))
        self.assertEqual(dataset.X.index.tolist(), [30, 10])
        self.assertEqual(dataset.metadata.case_id.tolist(), ["c2", "c1"])
        self.assertEqual(dataset.groups.tolist(), ["s1", "s1"])

    def test_tabular_rejects_misaligned_missing_and_infinite_values(self):
        invalid = [
            {"y": pd.Series([0, 1, 0])},
            {"y": pd.Series([0, None, 0], index=[10, 20, 30])},
            {"X": pd.DataFrame({"f1": [1, np.inf, 3]}, index=[10, 20, 30])},
        ]
        for overrides in invalid:
            with self.subTest(), self.assertRaises(ValueError):
                self.tabular(**overrides)

    def test_metadata_never_in_features_and_no_fabricated_datetimes(self):
        dataset = self.tabular()
        with self.assertRaises(ValueError):
            self.tabular(X=dataset.X.assign(subject_id=1))
        invalid = dataset.metadata.copy()
        invalid["history_end_seconds"] = pd.to_datetime(["2000-01-01"] * 3)
        with self.assertRaises(ValueError):
            self.tabular(metadata=invalid)

    def test_multiple_cases_for_one_subject_remain_separate_trajectories(self):
        reader = InMemoryCaseDataset(self.frame, self.schema)
        self.assertEqual(len(reader.get_case("c1")), 2)
        self.assertEqual(len(reader.get_case("c2")), 1)
        self.assertEqual(reader.get_case("c2").time_seconds.iloc[0], 0)

    def test_placeholders_fail_explicitly(self):
        extractor = PlaceholderFeatureExtractor()
        window = Window("c1", "s1", 300.0, 0.0, 300.0, pd.DataFrame({"time_seconds": [1.0]}))
        with self.assertRaises(NotImplementedError):
            extractor.extract(window)
        with self.assertRaises(NotImplementedError):
            _ = extractor.feature_names
        model = PlaceholderClassifier()
        dataset = self.tabular()
        with self.assertRaises(NotImplementedError):
            model.fit(dataset.X, dataset.y, seed=42)
        with self.assertRaises(NotImplementedError):
            model.predict_proba(dataset.X)
