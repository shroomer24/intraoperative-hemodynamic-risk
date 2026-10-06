"""B-only recovery: temporary evidence, synthetic matrices, mocked SDK, no network."""

import importlib
import io
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import ExitStack, contextmanager, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import test_tabpfn_representation_study as previous

from intraop.data.modeling import DevelopmentPartitions
from intraop.evaluation.development import export_predictions

ROOT = previous.ROOT
RECOVERY = previous.STUDY / "recovery-b-v01"
sys.path.insert(0, str(RECOVERY))
runner = importlib.import_module("recovery_run")
recovery = importlib.import_module("recovery_support")
worker = importlib.import_module("recovery_worker")

original = recovery.original
PLAN = previous.PLAN


class RecoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        positions = np.array_split(np.arange(13975), 90)
        cls.training = previous.fixture(
            13975,
            90,
            prefix="fake_train",
            positives=[v[:12] for v in positions[:41]] + [positions[41][:6]],
        )
        queries = np.array_split(np.arange(3193), 23)
        cls.tuning = previous.fixture(
            3193,
            23,
            prefix="fake_query",
            positives=[v[:10] for v in queries[:9]] + [queries[0][10:14]],
        )

    @contextmanager
    def evidence(self):
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            study = Path(directory)
            rec = study / "recovery-b-v01"
            rec.mkdir()
            stack.enter_context(patch.object(recovery, "STUDY", study))
            stack.enter_context(patch.object(recovery, "RECOVERY", rec))
            stack.enter_context(patch.object(original, "STUDY", study))
            stack.enter_context(patch.object(worker, "STUDY", study))
            stack.enter_context(patch.object(runner, "STUDY", study))
            prep = {"candidates": {}}
            for variant in original.IDS[1:]:
                X, y, query, positions = original.matrices(
                    self.training, self.tuning, PLAN, variant
                )
                prep["candidates"][variant] = {
                    "variant": variant,
                    "feature_names": next(v for v in PLAN["variants"] if v["id"] == variant)[
                        "feature_names"
                    ],
                    "training_X_sha256": original.array_hash(X),
                    "training_y_sha256": original.array_hash(y),
                    "tuning_X_sha256": original.array_hash(query),
                    "selected_position_sha256": original.positions_hash(positions),
                    "query_source_order_sha256": original.positions_hash(
                        self.tuning.X.index.to_numpy()
                    ),
                }
            original.atomic_json(study / "preparation_manifest.json", prep, exclusive=True)
            original.atomic_bytes(
                study / "preparation_manifest.sha256", b"retained-seal\n", exclusive=True
            )
            original.atomic_json(
                study / "D_context_selection_manifest.json",
                original.context_manifest(self.training, positions),
                exclusive=True,
            )
            for p in [rec / "validation_report.json", study / "validation_report_v02.json"]:
                original.atomic_json(p, {"status": "PASS", "mode": "MOCK_ONLY"}, exclusive=True)
            original.atomic_bytes(
                rec / "approved_recovery_request.md",
                (RECOVERY / "approved_recovery_request.md").read_bytes(),
                exclusive=True,
            )
            baseline = study / "retained_baseline.csv"
            export_predictions(
                self.tuning,
                np.full(3193, 0.03),
                model="tabpfn_full",
                candidate_id="v3p5_default_seed42",
                probability_output=True,
            ).to_csv(baseline, index=False)
            stack.enter_context(patch.object(runner, "BASELINE", baseline))
            old = study / "execution/candidates" / recovery.B / "attempt_1"
            identity = original.digest_json(prep["candidates"][recovery.B])
            original.atomic_json(
                old / "reservation.json", {"scientific_identity_sha256": identity}, exclusive=True
            )
            original.atomic_json(
                old / "worker_started.json", {"status": "STARTED", "pid": 1234}, exclusive=True
            )
            protocol = {
                "status": "WORKER_FAILED",
                "variant": recovery.B,
                "error_type": "ValueError",
            }
            original.atomic_json(
                old / "logs/exit.json",
                {"returncode": 2, "signal": None, "worker_protocol": protocol},
                exclusive=True,
            )
            original.atomic_json(
                old / "logs/failure_marker.json",
                {"status": "FAILED", "returncode": 2},
                exclusive=True,
            )
            original.atomic_json(old / "logs/stdout.json", protocol, exclusive=True)
            original.atomic_bytes(
                old / "logs/stderr.txt", b"Raw stderr not retained.\n", exclusive=True
            )
            original.atomic_json(
                study / "execution/review_required.json",
                {"status": "STOPPED_FOR_REVIEW", "error_type": "RuntimeError"},
                exclusive=True,
            )
            original.atomic_json(
                study / "execution/started_manifest.json",
                {
                    "identity": {
                        "study_plan_sha256": original.SPEC_HASH,
                        "preparation_manifest_sha256": original.sha256(
                            study / "preparation_manifest.json"
                        ),
                        "baseline_A_prediction_sha256": original.sha256(baseline),
                    }
                },
                exclusive=True,
            )
            original.atomic_bytes(
                rec / "software_tests.txt", b"Mock validation only\n", exclusive=True
            )
            auth = {
                "status": "APPROVED",
                "variant": recovery.B,
                "attempt_2_authorization": "EXECUTION_RECOVERY_ONLY",
                "attempt_1_operational_state": "FAILED",
                "attempt_1_hosted_outcome": "INDETERMINATE",
                "maximum_B_attempt": 2,
                "C_D_retry_authorized": False,
                "C_D_progression_requires_verified_B_complete": True,
                "automatic_retry_permitted": False,
                "attempt_3_permitted": False,
                "independent_candidate_evaluation_number": 1,
                "approval_request_sha256": recovery.APPROVAL_SHA,
                "study_plan_sha256": original.SPEC_HASH,
                "preparation_manifest_sha256": original.sha256(study / "preparation_manifest.json"),
                "preparation_seal_sha256": original.sha256(study / "preparation_manifest.sha256"),
                "attempt_1_evidence_sha256": recovery.tree_hashes(old),
                "attempt_1_directory_inventory": ["logs"],
                "historical_execution_records_sha256": {
                    name: original.sha256(study / "execution" / name)
                    for name in ["review_required.json", "started_manifest.json"]
                },
                "scientific_contract": recovery.scientific_contract(
                    PLAN, prep, self.training, self.tuning
                ),
                "recovery_code_hashes": {
                    name: original.sha256(ROOT / name) for name in recovery.CODE_PATHS
                },
                "validation_evidence_sha256": {
                    name: original.sha256(rec / name) for name in recovery.VALIDATION_FILES
                },
            }
            self.write_auth(rec, auth)
            for mod in [original, runner]:
                stack.enter_context(patch.object(mod, "verified_spec", return_value=PLAN))
                stack.enter_context(patch.object(mod, "verify_preparation", return_value=prep))
                stack.enter_context(
                    patch.object(
                        mod,
                        "load_development",
                        side_effect=lambda _p: (
                            SimpleNamespace(access_log=["training", "tuning"]),
                            self.training,
                            self.tuning,
                        ),
                    )
                )
            stack.enter_context(
                patch.object(original, "environment", return_value={"mocked": True})
            )
            yield SimpleNamespace(
                study=study, rec=rec, old=old, auth=auth, prep=prep, baseline=baseline, stack=stack
            )

    def write_auth(self, rec, auth):
        path = rec / "B_attempt_2_authorization.json"
        original.atomic_json(path, auth)
        original.atomic_bytes(path.with_suffix(".sha256"), (original.sha256(path) + "\n").encode())

    def reserve(self, evidence):
        return recovery.reserve_B_attempt_2(
            PLAN, evidence.prep, self.training, self.tuning, scanner=lambda: []
        )

    def test_attempt_1_is_immutable_and_new_path_is_used(self):
        with self.evidence() as e:
            before = recovery.tree_hashes(e.old)
            historical = original.sha256(e.study / "execution/review_required.json")
            logs = self.reserve(e)
            self.assertEqual(logs, e.old.parent / "attempt_2/logs")
            self.assertEqual(recovery.state(recovery.B), "INDETERMINATE")
            self.assertEqual(before, recovery.tree_hashes(e.old))
            self.assertEqual(
                historical, original.sha256(e.study / "execution/review_required.json")
            )
            self.assertEqual(
                original.read_json(e.old / "logs/failure_marker.json")["status"], "FAILED"
            )

    def test_exact_authorization_and_seal_are_required(self):
        for field, value in [
            ("variant", "C_causal_contrasts78"),
            ("maximum_B_attempt", 3),
            ("attempt_2_authorization", "GENERIC_RETRY"),
            ("C_D_retry_authorized", True),
        ]:
            with self.evidence() as e:
                e.auth[field] = value
                self.write_auth(e.rec, e.auth)
                with self.assertRaises(PermissionError):
                    self.reserve(e)
                self.assertFalse(recovery.attempt_path(recovery.B).exists())
        with self.evidence() as e:
            (e.rec / "B_attempt_2_authorization.sha256").write_text("wrong\n")
            with self.assertRaises(ValueError):
                self.reserve(e)

    def test_scientific_fingerprint_mismatch_blocks(self):
        with self.evidence() as e:
            e.auth["scientific_contract"]["training_X_sha256"] = "changed"
            self.write_auth(e.rec, e.auth)
            with self.assertRaisesRegex(ValueError, "fingerprint"):
                self.reserve(e)
            self.assertFalse(recovery.attempt_path(recovery.B).exists())

    def test_preparation_seal_and_model_configuration_changes_block(self):
        with self.evidence() as e:
            (e.study / "preparation_manifest.sha256").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "preparation"):
                self.reserve(e)
        with self.evidence() as e:
            auth = original.read_json(e.rec / "B_attempt_2_authorization.json")
            auth["scientific_contract"]["constructor"]["random_state"] = 7
            self.write_auth(e.rec, auth)
            with self.assertRaisesRegex(ValueError, "fingerprint"):
                self.reserve(e)

    def test_reviewed_source_hash_mismatch_blocks(self):
        with self.evidence() as e:
            e.auth["recovery_code_hashes"][recovery.CODE_PATHS[0]] = "changed"
            self.write_auth(e.rec, e.auth)
            with self.assertRaisesRegex(ValueError, "implementation"):
                self.reserve(e)

    def test_attempt_1_and_historical_stop_hash_changes_block(self):
        for name in ["attempt", "stop"]:
            with self.evidence() as e:
                target = (
                    e.old / "logs/stdout.json"
                    if name == "attempt"
                    else e.study / "execution/review_required.json"
                )
                target.write_bytes(target.read_bytes() + b" ")
                with self.assertRaises(ValueError):
                    self.reserve(e)
                self.assertFalse(recovery.attempt_path(recovery.B).exists())

    def test_active_worker_blocks_before_allocation(self):
        with self.evidence() as e:
            with self.assertRaisesRegex(RuntimeError, "ACTIVE"):
                recovery.reserve_B_attempt_2(
                    PLAN, e.prep, self.training, self.tuning, scanner=lambda: [123]
                )
            self.assertFalse(recovery.attempt_path(recovery.B).exists())

    def test_unrecognized_downstream_attempt_blocks_B_allocation(self):
        with self.evidence() as e:
            (e.study / "execution/candidates" / original.IDS[2]).mkdir()
            with self.assertRaisesRegex(RuntimeError, "Unexpected C/D"):
                self.reserve(e)
            self.assertFalse(recovery.attempt_path(recovery.B).exists())

    def test_process_inspection_unavailable_fails_closed(self):
        with patch.object(recovery.subprocess, "run", side_effect=OSError("private raw error")):
            with self.assertRaisesRegex(RuntimeError, "unavailable"):
                recovery.require_inactive()

    def test_process_scan_recognizes_new_workers_without_exposing_argv(self):
        response = SimpleNamespace(
            stdout=f"{os.getpid()} Python python recovery_run.py\n"
            "8123 Python python recovery_worker.py --private token-value\n"
            "8124 bash bash harmless\n"
        )
        with patch.object(recovery.subprocess, "run", return_value=response):
            self.assertEqual(recovery.inspect_processes(), [8123])

    def test_existing_bundle_prediction_or_success_blocks_duplicate(self):
        for name in ["bundle", "prediction.csv", "completion.json", "exit.json"]:
            with self.evidence() as e:
                directory = e.old.parent
                if name == "bundle":
                    (directory / name).mkdir()
                elif name == "exit.json":
                    original.atomic_json(
                        directory / "orphan/exit.json", {"returncode": 0}, exclusive=True
                    )
                else:
                    original.atomic_bytes(directory / name, b"retained result", exclusive=True)
                with self.assertRaises(RuntimeError):
                    self.reserve(e)
                self.assertFalse(recovery.attempt_path(recovery.B).exists())

    def test_attempt_3_and_C_D_retry_are_impossible(self):
        with self.evidence() as e:
            self.reserve(e)
            retained = recovery.tree_hashes(recovery.attempt_path(recovery.B))
            with self.assertRaises(RuntimeError):
                self.reserve(e)
            self.assertEqual(retained, recovery.tree_hashes(recovery.attempt_path(recovery.B)))
            for variant, number in [(recovery.B, 3), (original.IDS[2], 2), (original.IDS[3], 2)]:
                with self.assertRaises(PermissionError):
                    worker.execute(variant, number)
            self.assertFalse((e.old.parent / "attempt_3").exists())

    def test_stage_records_have_only_safe_names_timestamps_status(self):
        with tempfile.TemporaryDirectory() as d:
            attempt = Path(d)
            for stage in recovery.STAGES:
                recovery.mark_stage(attempt, stage)
            for p in (attempt / "stages").glob("*.json"):
                value = original.read_json(p)
                self.assertEqual(set(value), {"stage", "timestamp_utc", "status"})
            with self.assertRaises(ValueError):
                recovery.mark_stage(attempt, "token=private-payload")
            self.assertEqual(recovery.stage_summary(attempt)["last_completed_stage"], "COMPLETE")

    def test_extra_raw_payload_fields_in_diagnostics_are_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)
            original.atomic_json(
                path / "stages/01_WORKER_STARTED.json",
                {
                    "stage": "WORKER_STARTED",
                    "status": "completed",
                    "timestamp_utc": original.timestamp(),
                    "authorization": "fake secret",
                },
                exclusive=True,
            )
            with self.assertRaisesRegex(ValueError, "Unsafe"):
                recovery.stage_summary(path)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)
            original.atomic_json(
                path / "http_events/000001.json",
                {
                    "status": "RESPONSE_RECEIVED",
                    "timestamp_utc": original.timestamp(),
                    "http_status": 200,
                    "http_category": "HTTP_2XX",
                    "patient_values": [3, 4],
                },
                exclusive=True,
            )
            with self.assertRaisesRegex(ValueError, "Unsafe"):
                recovery.stage_summary(path)

    def test_HTTP_status_is_safe_and_no_transport_retry_occurs(self):
        with tempfile.TemporaryDirectory() as d:
            calls = []

            class Client:
                def send(self, *a, **k):
                    calls.append(1)
                    return SimpleNamespace(
                        status_code=401, body="patient-value", headers={"authorization": "secret"}
                    )

            original_send = Client.send
            with worker.no_transport_retries(SimpleNamespace(Client=Client), Path(d)):
                with self.assertRaises(worker.OneShotTransportFailure) as failure:
                    Client().send()
            self.assertEqual(failure.exception.http_status, 401)
            self.assertEqual(len(calls), 1)
            self.assertIs(Client.send, original_send)
            evidence = "".join(p.read_text() for p in Path(d).rglob("*.json"))
            for sensitive in ["patient-value", "authorization", "secret", "headers", "body"]:
                self.assertNotIn(sensitive, evidence)
            summary = recovery.stage_summary(Path(d))
            self.assertTrue(summary["http_response_received"])
            self.assertEqual(summary["last_observed_http_status"], 401)

    def test_transport_raw_exception_string_is_discarded(self):
        with tempfile.TemporaryDirectory() as d:

            class Client:
                def send(self, *a, **k):
                    raise ValueError("token=secret signed_url patient-values")

            with worker.no_transport_retries(SimpleNamespace(Client=Client), Path(d)):
                with self.assertRaises(worker.OneShotTransportFailure) as failure:
                    Client().send()
            self.assertIsNone(failure.exception.http_status)
            self.assertNotIn("secret", str(failure.exception))
            self.assertFalse(recovery.stage_summary(Path(d))["http_response_received"])

    def test_holdout_partitions_rejected_before_value_reader(self):
        data = DevelopmentPartitions.__new__(DevelopmentPartitions)
        with patch("intraop.data.modeling._load_partition") as reader:
            for partition in ["calibration", "test"]:
                with self.assertRaises(PermissionError):
                    data.load(partition)
            reader.assert_not_called()

    def mocked_sdk_and_workers(self, e, *, fail=False, bad_metadata=False):
        fits = []

        class MockClassifier:
            def __init__(self, **kwargs):
                if kwargs != original.CONSTRUCTOR:
                    raise AssertionError("Scientific constructor changed")
                self.classes_ = np.array([0, 1])

            def fit(self, X, y):
                fits.append(X.shape[1])
                if fail is True or fail == X.shape[1]:
                    raise ValueError("token=secret signed_url patient-values")

            def predict_proba(self, X):
                scores = np.where(self_tuning.y.to_numpy() == 1, 0.9, 0.01)
                self._last_meta = {
                    "package_version": "9.0.0",
                    "billing_model_version": "v3.5",
                    "n_estimators": 8,
                    "classes": [0, 1],
                    "test_set_num_rows": len(X),
                    "test_set_num_cols": X.shape[1],
                    "tabpfn_config": original.CONSTRUCTOR,
                }
                if bad_metadata:
                    self._last_meta["billing_model_version"] = "v3.5-fast"
                return np.column_stack([1 - scores, scores])

        self_tuning = self.tuning
        e.stack.enter_context(
            patch.dict(
                sys.modules,
                {
                    "tabpfn_client": SimpleNamespace(TabPFNClassifier=MockClassifier),
                    "httpx": SimpleNamespace(
                        Client=type("Client", (), {"send": lambda *a, **k: None})
                    ),
                },
            )
        )
        e.stack.enter_context(patch.object(recovery, "require_inactive"))
        e.stack.enter_context(patch.object(recovery, "inspect_processes", return_value=[]))

        def child_run(command, **kwargs):
            variant = command[command.index("--variant") + 1]
            number = command[command.index("--attempt") + 1]
            stream = io.StringIO()
            with (
                patch.object(
                    sys, "argv", ["recovery_worker.py", "--variant", variant, "--attempt", number]
                ),
                redirect_stdout(stream),
            ):
                code = worker.main()
            return subprocess.CompletedProcess(
                command, code, stream.getvalue(), "RAW child secret discarded"
            )

        e.stack.enter_context(patch.object(runner.subprocess, "run", side_effect=child_run))
        return fits

    def test_attempt_2_failure_blocks_C_D_and_all_future_launches(self):
        with self.evidence() as e:
            before = recovery.tree_hashes(e.old)
            fits = self.mocked_sdk_and_workers(e, fail=True)
            with redirect_stdout(io.StringIO()), self.assertRaises(RuntimeError):
                runner.run()
            self.assertEqual(fits, [71])
            self.assertEqual(recovery.state(recovery.B), "FAILED")
            self.assertEqual(before, recovery.tree_hashes(e.old))
            failure = original.read_json(recovery.attempt_path(recovery.B) / "failure.json")
            self.assertEqual(failure["last_completed_stage"], "HOSTED_FIT_STARTED")
            self.assertEqual(failure["error_type"], "ValueError")
            self.assertTrue(failure["hosted_call_begun"])
            self.assertFalse(failure["hosted_call_returned"])
            record = original.read_json(recovery.attempt_path(recovery.B) / "logs/exit.json")
            self.assertEqual(record["returncode"], 2)
            self.assertIsNone(record["signal"])
            for variant in original.IDS[2:]:
                self.assertFalse((e.study / "execution/candidates" / variant).exists())
            with self.assertRaises(RuntimeError):
                runner.run()
            self.assertEqual(fits, [71])
            for path in recovery.attempt_path(recovery.B).rglob("*.json"):
                self.assertNotIn("token=secret", path.read_text())

    def test_response_processing_failure_is_distinguished_after_predict_returns(self):
        with self.evidence() as e:
            fits = self.mocked_sdk_and_workers(e, bad_metadata=True)
            with redirect_stdout(io.StringIO()), self.assertRaises(RuntimeError):
                runner.run()
            failure = original.read_json(recovery.attempt_path(recovery.B) / "failure.json")
            self.assertEqual(failure["last_completed_stage"], "RESPONSE_PROCESSING_STARTED")
            self.assertTrue(failure["hosted_fit_returned"])
            self.assertTrue(failure["predict_proba_returned"])
            self.assertTrue(failure["hosted_call_returned"])
            self.assertEqual(fits, [71])
            self.assertFalse((e.study / "execution/candidates" / original.IDS[2]).exists())

    def test_native_worker_signal_retains_stage_and_stops_C_D(self):
        with self.evidence() as e:
            e.stack.enter_context(patch.object(recovery, "require_inactive"))
            e.stack.enter_context(patch.object(recovery, "inspect_processes", return_value=[]))

            def crash(command, **kwargs):
                attempt = recovery.attempt_path(recovery.B)
                for stage in [
                    "WORKER_STARTED",
                    "INPUT_VALIDATED",
                    "HOSTED_CALL_STARTED",
                    "HOSTED_FIT_STARTED",
                ]:
                    recovery.mark_stage(attempt, stage)
                return subprocess.CompletedProcess(command, -11, "", "private raw crash stream")

            e.stack.enter_context(patch.object(runner.subprocess, "run", side_effect=crash))
            with redirect_stdout(io.StringIO()), self.assertRaises(RuntimeError):
                runner.run()
            evidence = original.read_json(recovery.attempt_path(recovery.B) / "logs/exit.json")
            self.assertEqual(evidence["returncode"], -11)
            self.assertEqual(evidence["signal"], 11)
            self.assertEqual(evidence["last_completed_stage"], "HOSTED_FIT_STARTED")
            self.assertEqual(evidence["hosted_outcome"], "INDETERMINATE")
            self.assertEqual(recovery.state(recovery.B), "FAILED")
            self.assertFalse((e.study / "execution/candidates" / original.IDS[2]).exists())

    def test_C_failure_receives_no_retry_and_blocks_D(self):
        with self.evidence() as e:
            fits = self.mocked_sdk_and_workers(e, fail=78)
            with redirect_stdout(io.StringIO()), self.assertRaises(RuntimeError):
                runner.run()
            self.assertEqual(fits, [71, 78])
            self.assertEqual(recovery.state(recovery.B), "COMPLETE")
            self.assertEqual(recovery.state(original.IDS[2]), "FAILED")
            self.assertFalse((e.study / "execution/candidates" / original.IDS[3]).exists())
            with self.assertRaises(RuntimeError):
                runner.run()
            self.assertEqual(fits, [71, 78])

    def test_attempt_2_success_permits_C_D_once_and_completed_outputs_are_immutable(self):
        with self.evidence() as e:
            before = recovery.tree_hashes(e.old)
            baseline = e.baseline.read_bytes()
            fits = self.mocked_sdk_and_workers(e)
            with redirect_stdout(io.StringIO()):
                result = runner.run()
            self.assertEqual(result["status"], "COMPLETE")
            self.assertEqual(fits, [71, 78, 74])
            self.assertEqual(recovery.state(recovery.B), "COMPLETE")
            self.assertEqual(before, recovery.tree_hashes(e.old))
            self.assertEqual(e.baseline.read_bytes(), baseline)
            self.assertEqual(
                original.read_json(recovery.attempt_path(recovery.B) / "completion.json")[
                    "attempt_number"
                ],
                2,
            )
            self.assertEqual(result["B_attempt_1_hosted_outcome"], "INDETERMINATE")
            self.assertFalse((e.study / "model_lock_manifest.json").exists())
            for variant in original.IDS[2:]:
                self.assertTrue(
                    (
                        e.study / "execution/candidates" / variant / "attempt_1/completion.json"
                    ).exists()
                )
                self.assertFalse(
                    (e.study / "execution/candidates" / variant / "attempt_2").exists()
                )
            completed = recovery.tree_hashes(e.study / "execution/candidates")
            with redirect_stdout(io.StringIO()):
                runner.run()
            self.assertEqual(fits, [71, 78, 74])
            self.assertEqual(completed, recovery.tree_hashes(e.study / "execution/candidates"))
            with self.assertRaises(FileExistsError):
                worker.execute(recovery.B, 2)
            self.assertEqual(completed, recovery.tree_hashes(e.study / "execution/candidates"))

    def test_allocated_indeterminate_attempt_stops_without_retry(self):
        with self.evidence() as e:
            self.reserve(e)
            fits = self.mocked_sdk_and_workers(e)
            with self.assertRaises(RuntimeError):
                runner.run()
            self.assertEqual(fits, [])
            self.assertEqual(recovery.state(recovery.B), "INDETERMINATE")
            self.assertFalse((e.old.parent / "attempt_3").exists())

    def test_C_D_worker_cannot_bypass_failed_B(self):
        with self.evidence() as e:
            fits = self.mocked_sdk_and_workers(e)
            for variant in original.IDS[2:]:
                with self.assertRaisesRegex(RuntimeError, "B recovery"):
                    worker.execute(variant, 1)
            self.assertEqual(fits, [])


if __name__ == "__main__":
    unittest.main()
