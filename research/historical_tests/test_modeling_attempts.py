"""Attempt recovery tests use synthetic state only; never launch hosted inference."""

import copy
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from intraop.data.modeling import read_json, sha256, write_json

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import modeling_attempts as attempts  # noqa: E402


class AttemptRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.output = Path(self.temporary.name) / "development"
        self.output.mkdir()
        (self.output / "process_candidates").mkdir()
        self.logs = self.output / "process_candidates" / f"{attempts.MAP_KEY}_logs"
        self.logs.mkdir()
        self.bundle = self.output / "process_candidates" / attempts.MAP_KEY
        self.prediction = self.output / "candidates" / f"{attempts.MAP_KEY}.csv"
        self.plan = read_json(REPO / "artifacts/modeling-v01/development_plan.json")
        self.plan_hash = sha256(REPO / "artifacts/modeling-v01/development_plan.json")
        self.process_evidence = {"status": "NO_RELEVANT_ACTIVE_PROCESS", "method": "synthetic test"}

    def preserve(self):
        return attempts.preserve_empty_attempt(
            self.output,
            self.plan,
            self.plan_hash,
            process_evidence=self.process_evidence,
        )

    def retry(self, *, plan=None, active=None, authorized=True):
        return attempts.retry_map_attempt(
            self.output,
            plan or self.plan,
            self.plan_hash,
            explicitly_authorized=authorized,
            scanner=lambda: active or [],
        )

    def test_empty_directory_is_indeterminate_not_failed(self):
        self.assertEqual(
            attempts.attempt_state(self.bundle, self.prediction, self.logs), "INDETERMINATE"
        )
        self.assertFalse((self.logs / "failure_marker.json").exists())

    def test_first_attempt_preserved_and_no_success_failure_server_inference(self):
        before = self.logs.stat()
        first = self.preserve()
        self.assertEqual(first["status"], "indeterminate")
        self.assertEqual(first["fit_outcome"], "unknown")
        self.assertEqual(first["hosted_request_reached_server"], "unknown")
        self.assertEqual(first["attempt_number"], 1)
        self.assertEqual(list(self.logs.iterdir()), [])
        after = self.logs.stat()
        self.assertEqual(
            (before.st_ino, before.st_mtime_ns, before.st_ctime_ns),
            (after.st_ino, after.st_mtime_ns, after.st_ctime_ns),
        )
        with self.assertRaises(FileExistsError):
            self.preserve()

    def test_active_worker_blocks_retry_without_second_attempt(self):
        self.preserve()
        with self.assertRaisesRegex(RuntimeError, "ACTIVE"):
            self.retry(active=[{"pid": 12345, "state": "R"}])
        self.assertFalse(
            (attempts.ledger_dir(self.output, attempts.MAP_KEY) / "attempt_2.json").exists()
        )

    def test_completed_bundle_or_prediction_blocks_retry(self):
        self.preserve()
        self.bundle.mkdir()
        with self.assertRaisesRegex(PermissionError, "Completed artifact"):
            self.retry()
        self.bundle.rmdir()
        self.prediction.parent.mkdir()
        self.prediction.write_bytes(b"synthetic-completed")
        digest = sha256(self.prediction)
        with self.assertRaisesRegex(PermissionError, "Completed artifact"):
            self.retry()
        self.assertEqual(sha256(self.prediction), digest)

    def test_successful_exit_blocks_duplicate_retry(self):
        self.preserve()
        write_json(self.logs / "exit.json", {"returncode": 0})
        with self.assertRaisesRegex(PermissionError, "Exit evidence"):
            self.retry()
        self.assertEqual(read_json(self.logs / "exit.json")["returncode"], 0)

    def test_failed_attempt_still_stops(self):
        self.preserve()
        write_json(self.logs / "failure_marker.json", {"returncode": 2})
        with self.assertRaisesRegex(PermissionError, "FAILED"):
            self.retry()
        self.assertEqual(attempts.attempt_state(self.bundle, self.prediction, self.logs), "FAILED")
        # The second attempt is itself protected; its nonzero exit cannot be retried.

    def test_new_failure_in_attempt_two_still_stops(self):
        self.preserve()
        new_logs = self.retry()
        write_json(new_logs / "exit.json", {"returncode": 7})
        with patch.object(attempts, "inspect_processes", return_value=[]):
            with self.assertRaisesRegex(RuntimeError, "FAILED"):
                attempts.paths_for_candidate(
                    self.output,
                    self.plan,
                    self.plan_hash,
                    "tabpfn_map",
                    "v3p5_default_seed42",
                    retry_map=True,
                )
        self.assertEqual(read_json(new_logs / "exit.json")["returncode"], 7)

    def test_explicit_retry_creates_attempt_two_only_in_new_path(self):
        self.preserve()
        first = attempts.ledger_dir(self.output, attempts.MAP_KEY) / "attempt_1.json"
        digest = sha256(first)
        new_logs = self.retry()
        self.assertNotEqual(new_logs, self.logs)
        self.assertTrue(new_logs.name.endswith("_attempt_2_logs"))
        self.assertFalse(new_logs.exists())  # supervisor creates it only at the new launch
        self.assertEqual(list(self.logs.iterdir()), [])
        self.assertEqual(sha256(first), digest)
        second = read_json(first.parent / "attempt_2.json")
        self.assertEqual(second["attempt_number"], 2)
        self.assertFalse(second["retry_is_additional_model_selection"])
        with self.assertRaisesRegex(PermissionError, "already reserved"):
            self.retry()

    def test_indeterminate_retry_requires_explicit_flag(self):
        self.preserve()
        with self.assertRaisesRegex(PermissionError, "explicit retry flag"):
            self.retry(authorized=False)
        with patch.object(attempts, "inspect_processes", return_value=[]):
            with self.assertRaisesRegex(PermissionError, "explicit retry flag"):
                attempts.paths_for_candidate(
                    self.output, self.plan, self.plan_hash, "tabpfn_map", "v3p5_default_seed42"
                )

    def test_retry_cannot_change_scientific_config_or_interpreter_or_split(self):
        self.preserve()
        for field in ("model", "features", "interpreter", "client", "split"):
            changed = copy.deepcopy(self.plan)
            if field == "model":
                changed["models"]["tabpfn_map"]["candidates"][0]["hyperparameters"][
                    "model_path"
                ] = "forbidden"
            elif field == "features":
                changed["models"]["tabpfn_map"]["features"].reverse()
            elif field == "interpreter":
                changed["python_executable"] = "/forbidden/python"
            elif field == "client":
                changed["package_versions"]["tabpfn-client"] = "forbidden"
            else:
                changed["dataset_manifest_hashes"]["realized_split_manifest.json"] = "forbidden"
            with (
                self.subTest(field=field),
                self.assertRaisesRegex(ValueError, "configuration changed"),
            ):
                self.retry(plan=changed)

    def test_original_evidence_mutation_stops_retry(self):
        self.preserve()
        (self.logs / "new_evidence.txt").write_text("synthetic evidence")
        with self.assertRaisesRegex(ValueError, "evidence changed"):
            self.retry()
        self.assertTrue((self.logs / "new_evidence.txt").exists())

    def test_indeterminate_second_attempt_never_gets_automatic_third_retry(self):
        self.preserve()
        new_logs = self.retry()
        new_logs.mkdir()
        with patch.object(attempts, "inspect_processes", return_value=[]):
            with self.assertRaisesRegex(RuntimeError, "further retry"):
                attempts.paths_for_candidate(
                    self.output,
                    self.plan,
                    self.plan_hash,
                    "tabpfn_map",
                    "v3p5_default_seed42",
                    retry_map=True,
                )

    def test_full_tabpfn_new_candidate_gets_normal_attempt_and_complete_skips(self):
        with patch.object(attempts, "inspect_processes", return_value=[]):
            bundle, logs, state = attempts.paths_for_candidate(
                self.output, self.plan, self.plan_hash, "tabpfn_full", "v3p5_default_seed42"
            )
        self.assertEqual(state, "NEW")
        first = read_json(
            attempts.ledger_dir(self.output, "tabpfn_full__v3p5_default_seed42") / "attempt_1.json"
        )
        self.assertEqual(first["attempt_number"], 1)
        bundle.mkdir()
        write_json(logs / "exit.json", {"returncode": 0})
        result = attempts.paths_for_candidate(
            self.output, self.plan, self.plan_hash, "tabpfn_full", "v3p5_default_seed42"
        )
        self.assertEqual(result[2], "COMPLETE")

    def test_completed_retry_uses_attempt_two_success_logs_and_skips(self):
        self.preserve()
        second_logs = self.retry()
        self.bundle.mkdir()
        write_json(second_logs / "exit.json", {"returncode": 0})
        bundle, logs, state = attempts.paths_for_candidate(
            self.output,
            self.plan,
            self.plan_hash,
            "tabpfn_map",
            "v3p5_default_seed42",
            retry_map=True,
        )
        self.assertEqual(state, "COMPLETE")
        self.assertEqual(logs, second_logs)
        self.assertEqual(bundle, self.bundle)
        self.assertEqual(list(self.logs.iterdir()), [])

    def test_process_inspection_failure_blocks_launch(self):
        with patch.object(subprocess, "run", side_effect=PermissionError("sandbox")):
            with self.assertRaisesRegex(RuntimeError, "inspection unavailable"):
                attempts.require_inactive()

    def test_process_inspection_matches_workers_parents_and_sanitizes(self):
        rows = (
            f"{os.getpid()} 1 R /usr/bin/python3 /usr/bin/python3 "
            "/repo/modeling_development_resume.py\n"
            "8001 1 R /usr/bin/python3 /usr/bin/python3 /repo/modeling_tabpfn_worker.py "
            "--model tabpfn_map --candidate v3p5_default_seed42 Bearer synthetic_secret\n"
            "8002 1 S /usr/bin/python3 /usr/bin/python3 /repo/modeling_development_resume.py\n"
            "8003 1 R /bin/sh /bin/sh -c 'modeling_tabpfn_worker.py'\n"
        )
        with patch.dict(os.environ, {"TABPFN_TOKEN": "synthetic_secret"}):
            with patch.object(
                subprocess, "run", return_value=subprocess.CompletedProcess([], 0, stdout=rows)
            ):
                found = attempts.inspect_processes()
        self.assertEqual([r["pid"] for r in found], [8001, 8002])
        self.assertNotIn("synthetic_secret", json.dumps(found))

    def test_parent_retry_then_full_completion_preserves_all_previous_candidates(self):
        import tests.test_modeling_process_isolation as process_tests

        fixture = process_tests.ProcessIsolationTests()
        fixture.setUp()
        fixture.test_checkpoint_continuation_completes_only_missing_work_and_can_resume_again(
            indeterminate_map=True,
        )


if __name__ == "__main__":
    unittest.main()
