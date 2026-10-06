"""Process isolation and checkpoint tests. No hosted calls or held-out access."""

import ast
import io
import json
import os
import signal
import sys
import tempfile
import unittest
import zipfile
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from intraop.data.modeling import read_json, sha256, write_json
from intraop.evaluation.development import development_metrics, export_predictions
from tests.test_modeling_development import FakeEstimator, synthetic_data

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import modeling_attempts as attempts  # noqa: E402
import modeling_candidate_worker as worker  # noqa: E402
import modeling_development_resume as resume  # noqa: E402
import modeling_process_support as support  # noqa: E402


class ProcessIsolationTests(unittest.TestCase):
    def setUp(self):
        self.plan = read_json(REPO / "artifacts/modeling-v01/development_plan.json")

    def test_exact_eight_candidate_grid_is_unchanged(self):
        expected = []
        for n in (200, 400):
            for depth in (2, 3):
                for lr in (0.05, 0.1):
                    expected.append(
                        {
                            "n_estimators": n,
                            "max_depth": depth,
                            "learning_rate": lr,
                            "min_child_weight": 5,
                            "subsample": 0.8,
                            "colsample_bytree": 0.8,
                            "reg_lambda": 1,
                            "reg_alpha": 0,
                            "objective": "binary:logistic",
                            "eval_metric": "logloss",
                            "tree_method": "hist",
                            "random_state": 42,
                            "n_jobs": 1,
                            "verbosity": 0,
                        }
                    )
        candidates = self.plan["models"]["xgboost"]["candidates"]
        self.assertEqual([c["hyperparameters"] for c in candidates], expected)
        self.assertEqual([c["simplicity_rank"] for c in candidates], list(range(8)))
        self.assertEqual(candidates[0]["candidate_id"], "n200_d2_lr0p05")

    def test_contamination_fails_before_model_construction(self):
        for name in ("torch", "torch.nn", "tabpfn", "tabpfn_client"):
            with patch.dict(sys.modules, {name: SimpleNamespace()}):
                with self.assertRaisesRegex(RuntimeError, "isolation failure"):
                    support.assert_isolated()
                with patch.object(worker, "build_estimator") as factory:
                    with self.assertRaises(RuntimeError):
                        worker.fit_scores("xgboost", None, None, None, None)
                    factory.assert_not_called()

    def test_worker_import_blocker_rejects_even_transitive_imports(self):
        source = ast.parse((REPO / "scripts/modeling_xgboost_worker.py").read_text())
        blocker = next(n for n in source.body if isinstance(n, ast.ClassDef))
        namespace = {"importlib": __import__("importlib.abc")}
        exec(compile(ast.Module(body=[blocker], type_ignores=[]), "blocker", "exec"), namespace)
        guard = namespace["RuntimeBlocker"]()
        for name in ("torch", "torch.nn", "tabpfn", "tabpfn_client"):
            with self.assertRaisesRegex(RuntimeError, "forbidden import"):
                guard.find_spec(name)
        self.assertIsNone(guard.find_spec("numpy"))

    def test_supervisor_retains_signal_and_nonzero_exit(self):
        for script, expected in (
            ("import os,signal;os.kill(os.getpid(),signal.SIGSEGV)", -signal.SIGSEGV),
            ("raise SystemExit(7)", 7),
        ):
            with tempfile.TemporaryDirectory() as folder:
                logs = Path(folder) / "logs"
                with self.assertRaises(support.ChildFailure) as caught:
                    support.supervise([sys.executable, "-c", script], logs)
                self.assertEqual(caught.exception.returncode, expected)
                evidence = read_json(logs / "failure_marker.json")
                self.assertEqual(evidence["returncode"], expected)
                self.assertEqual(evidence["signal"], 11 if expected < 0 else None)
                self.assertTrue((logs / "stderr.txt").exists())
                self.assertTrue((logs / "stdout.txt").exists())

    def test_sanitized_child_output_removes_auth_and_payload(self):
        with patch.dict(os.environ, {"TABPFN_TOKEN": "synthetic_secret"}):
            text = support.sanitize_output(
                "Bearer synthetic_secret subject_id=synthetic123 X=[70,71,72]"
            )
        self.assertNotIn("synthetic_secret", text)
        self.assertNotIn("synthetic123", text)
        self.assertNotIn("70,71,72", text)

    def test_atomic_publish_failure_keeps_existing_and_no_temp_files(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / "result"
            target.write_bytes(b"completed")
            with self.assertRaises(FileExistsError):
                support.atomic_bytes(target, b"replacement", exclusive=True)
            self.assertEqual(target.read_bytes(), b"completed")
            self.assertEqual(list(Path(folder).iterdir()), [target])
            with patch.object(os, "replace", side_effect=OSError("simulated interruption")):
                with self.assertRaises(OSError):
                    support.atomic_bytes(target, b"partial")
            self.assertEqual(target.read_bytes(), b"completed")
            self.assertEqual(list(Path(folder).iterdir()), [target])

    def test_worker_failure_never_publishes_bundle(self):
        args = SimpleNamespace(model="xgboost", candidate="n200_d2_lr0p05", validation="synthetic")
        with tempfile.TemporaryDirectory() as folder:
            args.output = str(Path(folder) / "candidate")
            with patch.object(worker, "verified_plan", return_value=(self.plan, {})):
                with patch.object(worker, "build_estimator", side_effect=RuntimeError("failed")):
                    with self.assertRaises(RuntimeError):
                        worker.execute("xgboost", args)
            self.assertFalse(Path(args.output).exists())
            self.assertEqual(list(Path(folder).iterdir()), [])

    def test_worker_pass_uses_exact_config_and_commits_hashed_bundle(self):
        args = SimpleNamespace(model="xgboost", candidate="n200_d2_lr0p05", validation="synthetic")
        estimator = FakeEstimator()
        with tempfile.TemporaryDirectory() as folder:
            args.output = str(Path(folder) / "candidate")
            with patch.object(worker, "verified_plan", return_value=(self.plan, {})):
                with patch.object(worker, "build_estimator", return_value=estimator) as factory:
                    worker.execute("xgboost", args)
            factory.assert_called_once_with(
                "xgboost", self.plan["models"]["xgboost"]["candidates"][0]["hyperparameters"]
            )
            bundle = read_json(Path(args.output) / "bundle_manifest.json")
            for name, digest in bundle["files_sha256"].items():
                self.assertEqual(sha256(Path(args.output) / name), digest)
            self.assertEqual(estimator.fit_X.shape, (256, 74))
            self.assertEqual(
                read_json(Path(args.output) / "provenance.json")["partition_access_log"], []
            )

    def test_partial_resume_skips_all_ten_completed_candidates(self):
        completed = {
            (name, c["candidate_id"])
            for name in ("prevalence", "current_map", "logistic_map", "logistic_full")
            for c in self.plan["models"][name]["candidates"]
        }
        self.assertEqual(len(completed), 10)
        missing = resume.remaining_candidates(self.plan, completed)
        self.assertEqual(len(missing), 10)
        self.assertEqual(missing[0], ("xgboost", "n200_d2_lr0p05"))
        self.assertTrue(completed.isdisjoint(missing))
        self.assertEqual([name for name, _ in missing].count("xgboost"), 8)
        with self.assertRaises(ValueError):
            resume.remaining_candidates(self.plan, {("unexpected", "fixed")})

    def test_preservation_detects_mutation_and_preserves_aggregate_prefix(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            baseline, output = root / "preserved", root / "development"
            baseline.mkdir()
            output.mkdir()
            values = {
                "external_training_smoke.json": {
                    "status": "PASS",
                    "classes": [0, 1],
                    "positive_probability_column": 1,
                },
                "started_manifest.json": {"development_plan_sha256": "planhash"},
                "selected_configs.json": {"prevalence": {"fixed": True}},
            }
            for name, value in values.items():
                write_json(baseline / name, value)
            (baseline / "candidate_metrics.csv").write_text(
                "model,candidate_id\nprevalence,fixed\n"
            )
            for path in baseline.iterdir():
                (output / path.name).write_bytes(path.read_bytes())
            manifest = {
                "preservation_directory": "preserved",
                "original_plan_sha256": "planhash",
                "partial_files_sha256": {p.name: sha256(p) for p in baseline.iterdir()},
            }
            with patch.object(resume, "REPO", root):
                resume.preserve_check(output, manifest)
                resume.update_aggregates(
                    output, baseline, [{"model": "xgboost", "candidate_id": "missing"}]
                )
                resume.preserve_check(output, manifest)
                self.assertTrue(
                    (output / "candidate_metrics.csv")
                    .read_bytes()
                    .startswith((baseline / "candidate_metrics.csv").read_bytes())
                )
                (output / "external_training_smoke.json").write_text("{}")
                with self.assertRaisesRegex(ValueError, "checkpoint changed"):
                    resume.preserve_check(output, manifest)

    def test_worker_and_resume_have_no_holdout_loading_or_lock_write(self):
        for name in (
            "modeling_process_support.py",
            "modeling_candidate_worker.py",
            "modeling_development_resume.py",
        ):
            tree = ast.parse((REPO / "scripts" / name).read_text())
            calls = [
                n
                for n in ast.walk(tree)
                if isinstance(n, ast.Call)
                and isinstance(n.func, ast.Attribute)
                and n.func.attr == "load"
            ]
            self.assertTrue(all(n.args[0].value in ("training", "tuning") for n in calls))
            self.assertNotIn("FrozenPartitions", (REPO / "scripts" / name).read_text())
        # Existing poison-value firewall tests also run unmodified in the full suite.
        with tempfile.TemporaryDirectory() as folder:
            data = synthetic_data(self.plan["feature_sets"]["full"], 6)
            path = Path(folder) / "pred.csv"
            from intraop.evaluation.development import export_predictions

            export_predictions(
                data, np.full(6, 0.2), model="fake", candidate_id="fixed", probability_output=True
            ).to_csv(path, index=False)
            support.validate_prediction(path, data, "fake", "fixed")
            rows = __import__("pandas").read_csv(path).iloc[::-1]
            rows.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "alignment"):
                support.validate_prediction(path, data, "fake", "fixed")

    def test_numeric_looking_ids_remain_strings_in_checkpoint_reader(self):
        data = synthetic_data(self.plan["feature_sets"]["full"], 6)
        data.metadata["window_id"] = [str(1000 + i) for i in range(6)]
        data.metadata["case_id"] = [str(2000 + i) for i in range(6)]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "pred.csv"
            export_predictions(
                data, np.full(6, 0.2), model="fake", candidate_id="fixed", probability_output=True
            ).to_csv(path, index=False)
            rows = support.validate_prediction(path, data, "fake", "fixed")
            self.assertEqual(rows.window_id.tolist(), data.metadata.window_id.tolist())
            self.assertTrue(all(isinstance(value, str) for value in rows.window_id))

    def test_checkpoint_continuation_completes_only_missing_work_and_can_resume_again(
        self,
        indeterminate_map=False,
    ):
        training = synthetic_data(self.plan["feature_sets"]["full"], 13975)
        tuning = synthetic_data(self.plan["feature_sets"]["full"], 3193)
        scores = np.linspace(0.1, 0.3, 3193)
        records, selected = [], {}
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            output = root / "artifacts/modeling-v01/development"
            output.mkdir(parents=True)
            (output / "candidates").mkdir()
            (output / "models").mkdir()
            for model in ("prevalence", "current_map", "logistic_map", "logistic_full"):
                definition = self.plan["models"][model]
                selected[model] = {
                    "original_configuration": True,
                    "selected_configuration": definition["candidates"][0],
                }
                for candidate in definition["candidates"]:
                    cid = candidate["candidate_id"]
                    export_predictions(
                        tuning, scores, model=model, candidate_id=cid, probability_output=True
                    ).to_csv(output / "candidates" / f"{model}__{cid}.csv", index=False)
                    records.append(
                        {
                            "model": model,
                            "candidate_id": cid,
                            "simplicity_rank": candidate["simplicity_rank"],
                            "hyperparameters": json.dumps(
                                candidate["hyperparameters"], sort_keys=True
                            ),
                            **development_metrics(tuning.y, scores, probability_output=True),
                        }
                    )
                if model != "current_map":
                    (output / "models" / f"{model}.joblib").write_bytes(b"synthetic-model")
            __import__("pandas").DataFrame(records).to_csv(
                output / "candidate_metrics.csv", index=False
            )
            write_json(output / "selected_configs.json", selected)
            write_json(
                output / "external_training_smoke.json",
                {"status": "PASS", "classes": [0, 1], "positive_probability_column": 1},
            )
            write_json(
                output / "started_manifest.json",
                {
                    "development_plan_sha256": "originalhash",
                    "run_id": "synthetic",
                    "timestamp_utc": "synthetic",
                },
            )
            baseline = root / "preserved"
            __import__("shutil").copytree(output, baseline)
            hashes = {
                str(p.relative_to(output)): sha256(p) for p in output.rglob("*") if p.is_file()
            }
            manifest = {
                "preservation_directory": "preserved",
                "original_plan_sha256": "originalhash",
                "partial_files_sha256": hashes,
                "validation_directory": "gates",
            }
            for mode in ("synthetic", "training"):
                write_json(
                    root / "gates" / mode / "provenance.json",
                    {"status": "PASS", "forbidden_runtime_modules_absent": True},
                )
                write_json(root / "gates" / f"{mode}_logs" / "exit.json", {"returncode": 0})
            plan_path, manifest_path = root / "plan.json", root / "manifest.json"
            write_json(plan_path, self.plan)
            write_json(manifest_path, manifest)
            called = []

            def supervised(command, logs, **kwargs):
                model = command[command.index("--model") + 1]
                cid = command[command.index("--candidate") + 1]
                called.append((model, cid))
                bundle = Path(command[command.index("--output") + 1])
                bundle.mkdir()
                candidate = support.candidate_definition(self.plan, model, cid)[1]
                export_predictions(
                    tuning, scores, model=model, candidate_id=cid, probability_output=True
                ).to_csv(bundle / "prediction.csv", index=False)
                write_json(
                    bundle / "record.json",
                    {
                        "model": model,
                        "candidate_id": cid,
                        "simplicity_rank": candidate["simplicity_rank"],
                        "hyperparameters": json.dumps(candidate["hyperparameters"], sort_keys=True),
                        **development_metrics(tuning.y, scores, probability_output=True),
                    },
                )
                write_json(
                    bundle / "provenance.json",
                    {
                        "training_rows": 13975,
                        "query_rows": 3193,
                        "partition_access_log": ["training", "tuning"],
                        "classes": [0, 1],
                        "positive_probability_column": 1,
                        "forbidden_runtime_modules_absent": model == "xgboost",
                    },
                )
                if model == "xgboost":
                    (bundle / "model.joblib").write_bytes(b"synthetic-model")
                write_json(
                    bundle / "bundle_manifest.json",
                    {
                        "status": "COMPLETE",
                        "files_sha256": {p.name: sha256(p) for p in bundle.iterdir()},
                    },
                )
                write_json(logs / "exit.json", {"returncode": 0})

            loader = SimpleNamespace(access_log=["training", "tuning"])
            original = SimpleNamespace(
                validate_environment=lambda *a, **k: {},
                validate_inputs=lambda *a: (loader, training, tuning),
            )
            with (
                patch.object(resume, "REPO", root),
                patch.object(resume, "PLAN", plan_path),
                patch.object(resume, "REMEDIATION", manifest_path),
                patch.object(resume, "verified_plan", return_value=(self.plan, manifest)),
                patch.object(resume, "original_runner", return_value=original),
                patch.object(resume, "supervise", side_effect=supervised),
                patch.object(attempts, "inspect_processes", return_value=[]),
                redirect_stdout(io.StringIO()),
            ):
                resume.resume(conventional_only=True)
                self.assertEqual(len(called), 8)
                self.assertTrue(all(model == "xgboost" for model, _ in called))
                self.assertFalse((output / "development_completion_manifest.json").exists())
                if indeterminate_map:
                    old_logs = output / "process_candidates" / f"{attempts.MAP_KEY}_logs"
                    old_logs.mkdir()
                    attempts.preserve_empty_attempt(
                        output,
                        self.plan,
                        "originalhash",
                        process_evidence={"status": "NO_RELEVANT_ACTIVE_PROCESS"},
                    )
                resume.resume(retry_indeterminate_tabpfn_map=indeterminate_map)
                if indeterminate_map:
                    self.assertEqual(list(old_logs.iterdir()), [])
                    ledger = attempts.ledger_dir(output, attempts.MAP_KEY)
                    self.assertEqual(
                        read_json(ledger / "attempt_1.json")["status"], "indeterminate"
                    )
                    self.assertEqual(
                        read_json(ledger / "attempt_2_completion.json")["status"], "complete"
                    )
                self.assertEqual(len(called), 10)
                self.assertEqual([model for model, _ in called[-2:]], ["tabpfn_map", "tabpfn_full"])
                with self.assertRaisesRegex(PermissionError, "already complete"):
                    resume.resume()
            for name, digest in hashes.items():
                if name not in ("candidate_metrics.csv", "selected_configs.json"):
                    self.assertEqual(sha256(output / name), digest)
            completion = read_json(output / "development_completion_manifest.json")
            self.assertEqual(completion["candidate_count"], 20)
            self.assertFalse(completion["smoke_rerun"])
            self.assertEqual(completion["test_evaluation_count"], 0)
            with zipfile.ZipFile(output / "development_review.zip") as archive:
                self.assertNotIn("tuning_predictions.csv", archive.namelist())
                self.assertFalse(any(name.endswith(".joblib") for name in archive.namelist()))


if __name__ == "__main__":
    unittest.main()
