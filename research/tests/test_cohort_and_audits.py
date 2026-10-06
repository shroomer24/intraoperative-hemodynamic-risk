import json
import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from intraop.data.acquisition import SOURCE_URL, download_file, sha256_file
from intraop.data.audits import boundary_observation, select_waveform_examples, waveform_segment
from intraop.data.cohort import load_tables, split_statistics
from intraop.data.cohort_acquisition import HARD_CAP_BYTES, acquire_cohort, candidate_plan
from intraop.data.pipeline import run_tables
from intraop.data.vitaldb_reader import CaseSignals
from intraop.features.core import FEATURE_NAMES, CoreFeatureExtractor
from tests import test_acquisition_and_pipeline as pipeline_tests
from tests.helpers import processed, window


def clinical_fixture(number=30):
    return pd.DataFrame(
        {
            "caseid": range(1, number + 1),
            "subjectid": range(1, number + 1),
            "age": [40] * number,
            "ane_type": ["General"] * number,
            "opstart": [0] * number,
            "opend": [900] * number,
            "caseend": [1000] * number,
            "casestart": [0] * number,
            "aline1": ["Left radial"] * number,
        }
    )


class CohortAcquisitionTests(unittest.TestCase):
    def test_order_is_reproducible_row_order_invariant_and_outcome_blind(self):
        frame = clinical_fixture()
        first = candidate_plan(frame, [1, 2], seed=42, target=20)
        frame["future_hypotension"] = np.arange(len(frame))
        frame["label"] = 1
        frame["episode_count"] = 100
        other = candidate_plan(frame.iloc[::-1], [2, 1], seed=42, target=20)
        self.assertEqual(first, other)
        self.assertEqual(first["candidate_order"][:2], [1, 2])
        self.assertNotEqual(
            first["candidate_order"],
            candidate_plan(frame, [1, 2], seed=7, target=20)["candidate_order"],
        )

    def test_existing_population_filter_and_line_hint_only(self):
        frame = clinical_fixture()
        frame.loc[2, "age"] = 17
        frame.loc[3, "ane_type"] = "Spinal"
        frame.loc[4, "aline1"] = None
        order = candidate_plan(frame, [1, 2], seed=42, target=20)["candidate_order"]
        self.assertFalse({3, 4, 5} & set(order))

    def test_stream_cap_cleans_partial_and_counts_failed_transfer(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "case.vital"
            transferred = []
            with patch("urllib.request.urlopen", return_value=BytesIO(b"abcdef")):
                with self.assertRaisesRegex(RuntimeError, "byte budget"):
                    download_file(
                        path,
                        SOURCE_URL + "vital_files/0001.vital",
                        remaining_bytes=4,
                        on_bytes=transferred.append,
                    )
            self.assertEqual(sum(transferred), 4)
            self.assertFalse(path.exists())
            self.assertFalse(path.with_suffix(".vital.part").exists())

    def test_resumable_ledger_and_manifest_freeze(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "out"
            clinical_fixture().to_csv(root / "clinical_data.csv", index=False)
            (root / "vital_files").mkdir()
            local = root / "vital_files/0001.vital"
            local.write_bytes(b"pilot")
            (root / "SHA256SUMS.txt").write_text(f"{sha256_file(local)}  vital_files/0001.vital\n")
            with patch("urllib.request.urlopen", return_value=BytesIO(b"too-large")):
                first = acquire_cohort(root, output, [1], target=2, max_new_bytes=4)
            self.assertEqual(first["completed_case_ids"], [1])
            self.assertEqual(first["transferred_new_vital_bytes"], 4)
            with patch("urllib.request.urlopen") as urlopen:
                second = acquire_cohort(root, output, [1], target=2, max_new_bytes=4)
            urlopen.assert_not_called()
            self.assertEqual(first, second)
            second["frozen_for_processing"] = True
            (output / "acquisition_manifest.json").write_text(json.dumps(second))
            with patch("urllib.request.urlopen") as frozen_urlopen:
                frozen = acquire_cohort(root, output, [1], target=2, max_new_bytes=4)
            frozen_urlopen.assert_not_called()
            self.assertEqual(frozen, second)
            with self.assertRaisesRegex(ValueError, "frozen acquisition"):
                acquire_cohort(root, output, [1], seed=7, target=2, max_new_bytes=4)
            with self.assertRaises(ValueError):
                acquire_cohort(root, output, [1], max_new_bytes=HARD_CAP_BYTES + 1)


class RealizedCohortTests(unittest.TestCase):
    setUp = pipeline_tests.PipelineTests.setUp
    read = pipeline_tests.PipelineTests.read

    def test_realized_population_excludes_zero_eligible_subject_and_pins_pilot(self):
        clinical = clinical_fixture(31)
        clinical.to_csv(self.root / "clinical_data.csv", index=False)

        def read(root, row):
            case = self.read(root, row)
            if int(row.caseid) == 31:
                case.samples["map"]["value"] = 64.0
            return case

        output = self.root / "realized"
        with patch("intraop.data.pipeline.read_local_case", side_effect=read):
            report = run_tables(
                self.root,
                output,
                list(range(1, 32)),
                realized_development_subjects=["1", "2"],
                seed=42,
            )
        dataset, episodes = load_tables(output)
        manifest_path = output / "realized_split_manifest.json"
        before = manifest_path.read_bytes()
        manifest = json.loads(before)
        self.assertEqual(set(manifest["subject_to_partition"]), set(dataset.groups))
        self.assertNotIn("31", manifest["subject_to_partition"])
        self.assertEqual(manifest["subject_to_partition"]["1"], "training")
        self.assertEqual(
            manifest["counts"], {"training": 18, "tuning": 5, "calibration": 3, "test": 4}
        )
        stats = split_statistics(dataset, episodes, manifest, report["cases"])
        self.assertTrue(all(not v for v in stats["subject_intersections"].values()))
        table_before = (output / "table_manifest.json").read_bytes()
        with patch("intraop.data.pipeline.read_local_case", side_effect=read):
            run_tables(
                self.root,
                output,
                list(range(31, 0, -1)),
                realized_development_subjects=["2", "1"],
                seed=42,
            )
        self.assertEqual(before, manifest_path.read_bytes())
        self.assertEqual(table_before, (output / "table_manifest.json").read_bytes())


class AuditTests(unittest.TestCase):
    def test_boundary_observation_never_modifies_timestamps(self):
        samples = {
            "map": pd.DataFrame({"time_seconds": [0.5, 10.2, 900.3], "value": [80.0, 70.0, 60.0]})
        }
        case = CaseSignals(
            "1",
            "1",
            -200,
            950,
            samples,
            {"source_clock_origin": 10000, "origin_method": "all_record_minimum"},
        )
        original = samples["map"].copy(deep=True)
        row = pd.Series({"opstart": -200, "opend": 950, "caseend": 1000})
        observation = boundary_observation(case, row)
        pd.testing.assert_frame_equal(original, samples["map"])
        self.assertEqual(case.opstart_seconds, -200)
        self.assertIn("opstart_materially_before", observation["warnings"])
        self.assertEqual(observation["first_monitor_seconds"], 0.5)

    def test_original_waveform_times_gain_offset_and_gap_separator(self):
        track = SimpleNamespace(
            type=1,
            srate=2.0,
            fmt=3,
            gain=2.0,
            offset=1.0,
            recs=[
                {"dt": 100.25, "val": np.array([10, 20, 30])},
                {"dt": 105.0, "val": np.array([40])},
            ],
        )
        times, values = waveform_segment(track, 100.0, 0.0, 8.0)
        np.testing.assert_array_equal(times, [0.25, 0.75, 1.25, np.nan, 5.0, np.nan])
        np.testing.assert_array_equal(values, [21, 41, 61, np.nan, 81, np.nan])
        self.assertFalse(len(waveform_segment(None, 0, 0, 10)[0]))

    def test_waveform_and_audit_metadata_never_enter_features(self):
        case = processed(np.full(900, 80))
        clean = CoreFeatureExtractor().extract(window(case))
        case.grid["SNUADC/ART"] = 1234.0
        case.grid["waveform_available"] = True
        case.grid["opend"] = 900
        case.raw_valid_samples["SNUADC/ART"] = pd.DataFrame(
            {"time_seconds": [300.0], "value": [99999.0]}
        )
        observed = CoreFeatureExtractor().extract(window(case))
        pd.testing.assert_series_equal(pd.Series(clean), pd.Series(observed))
        self.assertEqual(tuple(observed), FEATURE_NAMES)

    def test_audit_selection_is_deterministic_training_only_and_unique(self):
        anchors = pd.DataFrame(
            [
                {
                    "case_id": str(i),
                    "subject_id": str(i),
                    "anchor_time_seconds": float(t),
                    "status": "eligible",
                    "label": t % 120 == 0,
                    "matched_episode_onset_seconds": t + 50 if t % 120 == 0 else np.nan,
                    "map_ordering_inconsistent_history_seconds": 0,
                    "history_map_coverage": 1.0,
                }
                for i in range(1, 5)
                for t in range(300, 900, 60)
            ]
        )
        minima = {(r.case_id, r.anchor_time_seconds): 70.0 for r in anchors.itertuples()}
        first = select_waveform_examples(anchors, {"1", "2", "3"}, minima)
        second = select_waveform_examples(anchors.iloc[::-1], {"3", "2", "1"}, minima)
        pd.testing.assert_frame_equal(pd.DataFrame(first), pd.DataFrame(second))
        self.assertNotIn("4", {r["subject_id"] for r in first})
        self.assertEqual(len(first), len({(r["case_id"], r["anchor_time_seconds"]) for r in first}))
