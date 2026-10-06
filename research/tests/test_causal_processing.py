import gzip
import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from intraop.data.preprocessing import causal_hold, preprocess_case, valid_samples
from intraop.data.vitaldb_reader import CaseSignals, recording_clock_origin


class CausalProcessingTests(unittest.TestCase):
    def test_fractional_future_samples_never_move_backwards(self):
        samples = pd.DataFrame({"time_seconds": [0.5, 2.1], "value": [80.0, 55.0]})
        held, _ = causal_hold(samples, np.array([0, 1, 2, 3]))
        self.assertTrue(np.isnan(held[0]))
        np.testing.assert_array_equal(held[1:], [80, 80, 55])

    def test_freshness_ten_seconds_is_inclusive(self):
        samples = pd.DataFrame({"time_seconds": [0.0], "value": [80.0]})
        held, age = causal_hold(samples, np.array([0, 10, 11]))
        np.testing.assert_array_equal(held[:2], [80, 80])
        self.assertTrue(np.isnan(held[2]))
        np.testing.assert_array_equal(age, [0, 10, 11])

    def test_future_mutation_does_not_change_prefix(self):
        samples = pd.DataFrame({"time_seconds": [0.0, 5.0, 100.0], "value": [80.0, 70.0, 20.0]})
        first = causal_hold(samples, np.arange(11))[0]
        samples.loc[2, "value"] = 200
        np.testing.assert_array_equal(first, causal_hold(samples, np.arange(11))[0])

    def test_map_checks_preserve_positive_low_values_and_boundary(self):
        data = pd.DataFrame(
            {
                "time_seconds": np.arange(8, dtype=float),
                "value": [np.nan, np.inf, -1, 0, 1, 65, 250, 251],
            }
        )
        self.assertEqual(valid_samples(data, signal="map").value.tolist(), [1, 65, 250])

    def test_invalid_observation_does_not_reset_last_valid_age(self):
        data = pd.DataFrame({"time_seconds": [0.0, 5.0], "value": [80.0, -1.0]})
        held, _ = causal_hold(valid_samples(data, signal="map"), np.array([5, 10, 11]))
        self.assertEqual(held[:2].tolist(), [80, 80])
        self.assertTrue(np.isnan(held[2]))

    def test_duplicate_timestamp_uses_last_valid_file_record(self):
        samples = pd.DataFrame({"time_seconds": [0.0, 0.0], "value": [80.0, 70.0]})
        self.assertEqual(causal_hold(samples, np.array([0]))[0][0], 70)

    def test_optional_tracks_remain_missing_and_ordering_is_flag_only(self):
        samples = {
            signal: pd.DataFrame({"time_seconds": [0.0, 1.0], "value": values})
            for signal, values in {"map": [80, 80], "sbp": [70, 70], "dbp": [90, 90]}.items()
        }
        case = preprocess_case(CaseSignals("c1", "s1", 0, 20, samples, {}))
        self.assertTrue(case.grid.loc[0, "map_ordering_inconsistent"])
        self.assertEqual(case.grid.loc[0, "map"], 80)
        self.assertTrue(case.grid.hr.isna().all())

    def test_fresh_preoperative_hold_and_surgical_grid_boundaries(self):
        samples = {"map": pd.DataFrame({"time_seconds": [9.0, 15.0, 21.0], "value": [80, 70, 30]})}
        case = preprocess_case(CaseSignals("c1", "s1", 10, 20, samples, {}))
        self.assertEqual(case.grid.loc[0, "map"], 80)
        self.assertEqual(case.grid.time_seconds.tolist(), list(range(10, 21)))
        self.assertEqual(case.grid.loc[10, "map"], 70)

    def test_recording_origin_comes_from_all_tracks_not_selected_sample(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tiny.vital"
            with gzip.open(path, "wb") as handle:
                handle.write(b"VITA" + struct.pack("<IH", 3, 10) + b"\0" * 10)
                for time, track in ((1002.5, 1), (1000.0, 2)):
                    payload = struct.pack("<HdHf", 10, time, track, 80)
                    handle.write(struct.pack("<BI", 1, len(payload)) + payload)
            self.assertEqual(recording_clock_origin(path), 1000)

    def test_recording_end_cannot_be_extended_by_sample_hold(self):
        samples = {"map": pd.DataFrame({"time_seconds": [0.0, 10.0], "value": [80, 80]})}
        case = preprocess_case(
            CaseSignals("c1", "s1", 0, 30, samples, {}, recording_end_seconds=12)
        )
        self.assertEqual(case.grid.time_seconds.max(), 12)

    def test_new_header_explicit_zero_is_not_shifted_to_first_signal(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tiny.vital"
            header = b"\0" * 10 + struct.pack("<dd", 0.0, 10.0)
            payload = struct.pack("<HdHf", 10, 2.5, 1, 80)
            with gzip.open(path, "wb") as handle:
                handle.write(b"VITA" + struct.pack("<IH", 3, len(header)) + header)
                handle.write(struct.pack("<BI", 1, len(payload)) + payload)
            self.assertEqual(recording_clock_origin(path), 0)

    def test_truncated_vital_file_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "broken.vital"
            with gzip.open(path, "wb") as handle:
                handle.write(b"VITA")
            with self.assertRaises(ValueError):
                recording_clock_origin(path)
