"""Continuation-specific invariants absent from the original acquisition tests."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from intraop.data.acquisition import sha256_file
from intraop.data.cohort_acquisition import acquire_cohort, candidate_plan
from tests.test_cohort_and_audits import clinical_fixture


class ContinuationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        frame = clinical_fixture()
        frame.to_csv(self.root / "clinical_data.csv", index=False)
        self.order = candidate_plan(frame, [1], seed=42, target=4)["candidate_order"]
        (self.root / "vital_files").mkdir()
        lines = []
        for case in self.order[:4]:
            path = self.root / f"vital_files/{case:04d}.vital"
            path.write_bytes(f"source-{case}".encode())
            lines.append(f"{sha256_file(path)}  vital_files/{case:04d}.vital")
        (self.root / "SHA256SUMS.txt").write_text("\n".join(lines))
        self.parent = self.root / "v01/acquisition_manifest.json"
        acquire_cohort(self.root, self.parent.parent, [1], target=2)
        parent = json.loads(self.parent.read_text())
        parent["frozen_for_processing"] = True
        self.parent.write_text(json.dumps(parent))
        self.parent_bytes = self.parent.read_bytes()

    def test_separate_checkpoint_preserves_prefix_and_reuses_verified_sources(self):
        with patch("urllib.request.urlopen") as request:
            result = acquire_cohort(
                self.root, self.root / "v02", [1], target=4, resume_from=self.parent
            )
        request.assert_not_called()
        self.assertEqual(result["completed_case_ids"], self.order[:4])
        self.assertEqual(result["continuation"]["parent_completed_case_ids"], self.order[:2])
        self.assertEqual(result["transferred_new_vital_bytes"], 0)
        self.assertEqual([r["case_id"] for r in result["attempts"]], self.order[2:4])
        self.assertTrue(all(r["success"] for r in result["attempts"]))
        self.assertGreaterEqual(result["elapsed_acquisition_seconds"], 0)
        self.assertEqual(self.parent.read_bytes(), self.parent_bytes)

    def test_corrupt_parent_file_rejected_before_checkpoint_or_download(self):
        (self.root / "vital_files/0001.vital").write_bytes(b"corrupt")
        with patch("urllib.request.urlopen") as request:
            with self.assertRaisesRegex(ValueError, "integrity failure"):
                acquire_cohort(self.root, self.root / "v02", [1], target=4, resume_from=self.parent)
        request.assert_not_called()
        self.assertFalse((self.root / "v02/acquisition_manifest.json").exists())

    def test_changed_order_rejected_before_download(self):
        with patch("urllib.request.urlopen") as request:
            with self.assertRaisesRegex(ValueError, "plan differs"):
                acquire_cohort(
                    self.root, self.root / "v02", [1], seed=7, target=4, resume_from=self.parent
                )
        request.assert_not_called()

    def test_new_transfer_ledger_and_first_failure_preserve_exact_prefix(self):
        for case in self.order[2:4]:
            (self.root / f"vital_files/{case:04d}.vital").unlink()

        def transfer(path, url, *, remaining_bytes, on_bytes):
            on_bytes(3)
            raise RuntimeError("interrupted")

        with patch("intraop.data.cohort_acquisition.download_file", side_effect=transfer):
            result = acquire_cohort(
                self.root, self.root / "v02", [1], target=4, resume_from=self.parent
            )
        self.assertEqual(result["completed_case_ids"], self.order[:2])
        self.assertEqual(result["transferred_new_vital_bytes"], 3)
        self.assertEqual(len(result["attempts"]), 1)
        self.assertFalse(result["attempts"][0]["success"])
        self.assertEqual(result["attempts"][0]["case_id"], self.order[2])
        self.assertEqual(result["attempts"][0]["transferred_bytes"], 3)
        self.assertEqual(self.parent.read_bytes(), self.parent_bytes)
