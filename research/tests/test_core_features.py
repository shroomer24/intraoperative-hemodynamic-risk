import unittest

import numpy as np
import pandas as pd

from intraop.features.base import Window
from intraop.features.core import FEATURE_NAMES, CoreFeatureExtractor, least_squares_slope
from tests.helpers import processed, window


class CoreFeatureTests(unittest.TestCase):
    def test_stable_schema_names_order_and_no_identifiers_or_future_metadata(self):
        values = CoreFeatureExtractor().extract(window(processed(np.full(900, 80))))
        self.assertEqual(len(FEATURE_NAMES), 74)
        self.assertEqual(tuple(values), FEATURE_NAMES)
        self.assertEqual(len(set(FEATURE_NAMES)), 74)
        for forbidden in (
            "subject_id",
            "case_id",
            "label",
            "future_observation_end_seconds",
            "matched_episode_onset_seconds",
            "opend",
        ):
            self.assertNotIn(forbidden, values)

    def test_absent_optional_signals_have_missing_summaries_and_full_missing_fraction(self):
        values = CoreFeatureExtractor().extract(window(processed(np.full(900, 80))))
        for signal in ("sbp", "dbp", "hr", "spo2", "etco2"):
            self.assertTrue(np.isnan(values[f"{signal}_latest"]))
            self.assertEqual(values[f"{signal}_missing_fraction"], 1)
            self.assertTrue(np.isnan(values[f"{signal}_measurement_age_seconds"]))

    def test_future_changes_cannot_enter_features(self):
        first = processed(np.full(900, 80))
        values = np.full(900, 80)
        values[301:] = 10
        other = processed(values)
        pd.testing.assert_series_equal(
            pd.Series(CoreFeatureExtractor().extract(window(first))),
            pd.Series(CoreFeatureExtractor().extract(window(other))),
        )

    def test_window_refuses_future_rows_and_datetime_inputs(self):
        for history in (
            pd.DataFrame({"time_seconds": [301.0]}),
            pd.DataFrame({"time_seconds": pd.to_datetime(["2000-01-01"])}),
        ):
            with self.assertRaises(ValueError):
                Window("c1", "s1", 300.0, 0.0, 300.0, history)

    def test_slopes_use_actual_elapsed_seconds_and_two_actual_observations(self):
        self.assertEqual(least_squares_slope([0, 10, 30], [0, 20, 60]), 2)
        self.assertEqual(least_squares_slope([0, 10, 30], [0, np.nan, 60]), 2)
        self.assertTrue(np.isnan(least_squares_slope([1, 1], [2, 3])))
        case = processed(np.full(900, 80))
        case.raw_valid_samples["map"] = pd.DataFrame({"time_seconds": [300.0], "value": [80.0]})
        values = CoreFeatureExtractor().extract(window(case))
        self.assertTrue(np.isnan(values["map_slope_300s"]))

    def test_ordered_minute_bins_and_minute_change(self):
        values = np.full(900, 80.0)
        for minute in range(1, 6):
            values[1 + (minute - 1) * 60 : 1 + minute * 60] = 65 + minute
        features = CoreFeatureExtractor().extract(window(processed(values)))
        self.assertEqual(
            [features[f"map_minute_{i}_median"] for i in range(1, 6)], [66, 67, 68, 69, 70]
        )
        self.assertEqual(features["map_median_latest_minus_previous_minute"], 1)

    def test_map_threshold_fractions_are_strictly_below_70_and_75(self):
        values = np.full(900, 75.0)
        values[1:101] = 65
        values[101:201] = 70
        features = CoreFeatureExtractor().extract(window(processed(values)))
        self.assertAlmostEqual(features["map_fraction_below_70"], 1 / 3)
        self.assertAlmostEqual(features["map_fraction_below_75"], 2 / 3)

    def test_pulse_pressure_and_ratio_use_aligned_past_values(self):
        case = processed(np.full(900, 80))
        for signal, value in (("sbp", 100), ("dbp", 60), ("hr", 80)):
            case.grid[signal] = float(value)
            case.grid[f"{signal}_age_seconds"] = 0.0
            case.raw_valid_samples[signal] = pd.DataFrame(
                {"time_seconds": [1.0, 300.0], "value": [value, value]}
            )
        features = CoreFeatureExtractor().extract(window(case))
        self.assertEqual(features["pulse_pressure_latest"], 40)
        self.assertEqual(features["pulse_pressure_slope_300s"], 0)
        self.assertEqual(features["hr_sbp_ratio_latest"], 0.8)

    def test_invalid_ratio_denominator_is_missing(self):
        case = processed(np.full(900, 80))
        case.grid["sbp"] = 0.0
        case.grid["hr"] = 80.0
        self.assertTrue(
            np.isnan(CoreFeatureExtractor().extract(window(case))["hr_sbp_ratio_latest"])
        )
