import json
import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from intraop.data.acquisition import acquire_subset
from intraop.data.pipeline import run_pilot
from intraop.data.protocol import TRACKS
from intraop.data.vitaldb_reader import CaseSignals, population_eligibility


class AcquisitionTests(unittest.TestCase):
    def test_no_implicit_full_download_empty_duplicate_or_excess_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            for cases in ([], [1, 1], list(range(1, 22)), [0], [6389], [True]):
                with self.subTest(), self.assertRaises(ValueError):
                    acquire_subset(Path(directory), cases)

    def test_network_failure_documents_exact_canonical_manual_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch("urllib.request.urlopen", side_effect=OSError("network unavailable")):
                with self.assertRaisesRegex(RuntimeError, "curl --fail --location") as error:
                    acquire_subset(Path(directory), [1])
            self.assertIn(
                "https://physionet.org/files/vitaldb/1.0.0/LICENSE.txt", str(error.exception)
            )
            self.assertFalse(list(Path(directory).glob("*.part")))

    def test_population_boundaries_and_top_coded_age(self):
        base = {
            "age": "18",
            "ane_type": "General",
            "opstart": 0,
            "opend": 900,
            "caseend": 1000,
            "casestart": 0,
        }
        self.assertIsNone(population_eligibility(pd.Series(base)))
        self.assertIsNone(population_eligibility(pd.Series({**base, "age": ">89"})))
        for overrides in ({"age": "17"}, {"ane_type": "Spinal"}, {"opend": -1}):
            self.assertIsNotNone(population_eligibility(pd.Series({**base, **overrides})))

    def test_download_byte_budget_enforced_before_file_is_committed(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch("urllib.request.urlopen", return_value=BytesIO(b"too many bytes")):
                with self.assertRaisesRegex(RuntimeError, "byte budget"):
                    acquire_subset(Path(directory), [1], max_total_bytes=1)
            self.assertFalse((Path(directory) / "LICENSE.txt").exists())


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        clinical = pd.DataFrame(
            {
                "caseid": range(1, 22),
                "subjectid": [1, 1, *range(2, 21)],
                "age": [40] * 21,
                "ane_type": ["General"] * 21,
                "opstart": [0] * 21,
                "opend": [900] * 21,
                "caseend": [1000] * 21,
                "casestart": [0] * 21,
            }
        )
        clinical.to_csv(self.root / "clinical_data.csv", index=False)
        (self.root / "track_names.csv").write_text("tname\nSolar8000/ART_MBP\n")
        (self.root / "LICENSE.txt").write_text("Synthetic test fixture, not a dataset license")

    def read(self, root, row):
        samples, track_audit = {}, {}
        for signal in TRACKS:
            present = signal == "map"
            samples[signal] = (
                pd.DataFrame(
                    {"time_seconds": np.arange(901, dtype=float), "value": np.full(901, 80.0)}
                )
                if present
                else pd.DataFrame(columns=["time_seconds", "value"], dtype=float)
            )
            track_audit[signal] = {"track_present": present}
        return CaseSignals(
            str(int(row.caseid)), str(int(row.subjectid)), 0, 900, samples, {"tracks": track_audit}
        )

    def test_complete_tables_reports_repeated_cases_and_development_splits(self):
        output = self.root / "output"
        with patch("intraop.data.pipeline.read_local_case", side_effect=self.read):
            report = run_pilot(self.root, output, [1, 2])
        self.assertEqual(report["cases_parsed"], 2)
        self.assertEqual(report["unique_subjects"], 1)
        self.assertEqual(report["unique_cases"], 2)
        self.assertEqual(report["eligible_windows"], 10)
        features = pd.read_csv(output / "features.csv")
        labels = pd.read_csv(output / "labels.csv")
        metadata = pd.read_csv(output / "metadata.csv", index_col="window_id")
        self.assertEqual(features.shape, (10, 74))
        self.assertTrue(features.index.equals(labels.index))
        self.assertTrue(features.index.equals(metadata.index))
        self.assertNotIn("subject_id", features.columns)
        self.assertEqual(report["pilot_windows_by_partition"]["test"], 0)
        self.assertFalse(
            json.loads((output / "provenance.json").read_text())["release_checksums_verified"]
        )
        self.assertTrue((output / "cohort_report.md").is_file())

    def test_absent_required_map_excludes_case_without_becoming_negative(self):
        def absent(root, row):
            case = self.read(root, row)
            case.audit["tracks"]["map"]["track_present"] = False
            return case

        with patch("intraop.data.pipeline.read_local_case", side_effect=absent):
            report = run_pilot(self.root, self.root / "empty", [1, 2])
        self.assertEqual(report["eligible_windows"], 0)
        self.assertEqual(report["negative_windows"], 0)
        self.assertEqual(report["pilot_case_exclusions"]["required_art_mbp_unavailable"], 2)
