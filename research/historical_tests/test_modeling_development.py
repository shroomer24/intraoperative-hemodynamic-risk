"""Synthetic firewall/orientation/selection tests; never call the hosted API."""

import ast
import importlib.util
import io
import os
import sys
import tempfile
import unittest
import zipfile
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from intraop.data.datasets import ClassificationDataset
from intraop.data.modeling import (
    DevelopmentPartitions,
    FrozenPartitions,
    predictor_array,
    read_json,
    sha256,
    write_json,
)
from intraop.evaluation.development import (
    development_metrics,
    export_predictions,
    select_candidate,
)
from intraop.evaluation.secure_errors import sanitized_traceback
from intraop.models.benchmarks import PrevalenceClassifier, build_estimator, positive_probabilities

REPO = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "development_runner", REPO / "scripts/modeling_development.py"
)
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


def synthetic_data(names, n=8):
    index = pd.Index(np.arange(n)[::-1] * 17 + 3, name="source_row")
    X = pd.DataFrame(
        np.arange(n * len(names)).reshape(n, len(names)) / 100 + 70, columns=names, index=index
    )
    metadata = pd.DataFrame(
        {
            "window_id": [f"synthetic_w{i}" for i in range(n)],
            "case_id": [f"synthetic_c{i}" for i in range(n)],
            "subject_id": [f"synthetic_s{i}" for i in range(n)],
            "anchor_time_seconds": np.arange(n) * 60.0 + 300,
            "history_start_seconds": np.arange(n) * 60.0,
            "history_end_seconds": np.arange(n) * 60.0 + 300,
            "future_observation_end_seconds": np.arange(n) * 60.0 + 660,
        },
        index=index,
    )
    return ClassificationDataset(X, pd.Series(np.arange(n) % 2, index=index), metadata)


class ReversedClassesModel:
    classes_ = np.array([1, 0])

    def predict_proba(self, X):
        p = np.arange(len(X)) / (len(X) + 1)
        return np.column_stack([p, 1 - p])


class FakeEstimator:
    """Deterministic synthetic model; fits are recorded for partition assertions."""

    classes_ = np.array([0, 1])

    def fit(self, X, y):
        self.fit_X = X.copy()
        self.fit_y = y.copy()
        return self

    def predict_proba(self, X):
        p = np.clip(X[:, 0] / 100, 0.01, 0.99) if X.shape[1] else np.full(len(X), 0.5)
        return np.column_stack([1 - p, p])


class DevelopmentFirewallTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.cohort = self.root / "cohort"
        self.modeling = self.root / "modeling"
        self.cohort.mkdir()
        self.modeling.mkdir()
        features = read_json(REPO / "artifacts/modeling-v01/feature_sets.json")
        self.features = features["full"]
        write_json(self.modeling / "feature_sets.json", features)
        write_json(self.cohort / "feature_schema.json", {"feature_names": self.features})
        partitions = ["training"] * 90 + ["tuning"] * 23 + ["calibration"] * 15 + ["test"] * 22
        self.split = {
            "subject_to_partition": {f"fake{i}": p for i, p in enumerate(partitions)},
            "counts": {"training": 90, "tuning": 23, "calibration": 15, "test": 22},
        }
        write_json(self.cohort / "realized_split_manifest.json", self.split)
        meta = [
            "window_id,case_id,subject_id,anchor_time_seconds,history_start_seconds,"
            "history_end_seconds,future_observation_end_seconds\n"
        ]
        Xlines, ylines = [",".join(self.features) + "\n"], ["label\n"]
        stats = {}
        for i, partition in enumerate(partitions):
            allowed = partition in ("training", "tuning")
            # Disallowed predictor, label and non-routing metadata fields are poison.
            # The reader must exclude them before any parsing/numeric conversion.
            value = "70" if allowed else "DO_NOT_PARSE_HOLDOUT_VALUE"
            Xlines.append(",".join([value] * 74) + "\n")
            ylines.append(str(i % 2) + "\n" if allowed else "DO_NOT_PARSE_HOLDOUT_LABEL\n")
            times = "300,0,300,660" if allowed else "FORBIDDEN,FORBIDDEN,FORBIDDEN,FORBIDDEN"
            meta.append(f"w{i},c{i},fake{i},{times}\n")
            if allowed:
                stats.setdefault(partition, {"eligible_windows": 0, "positive_windows": 0})
                stats[partition]["eligible_windows"] += 1
                stats[partition]["positive_windows"] += i % 2
        for name, lines in (
            ("features.csv", Xlines),
            ("labels.csv", ylines),
            ("metadata.csv", meta),
        ):
            (self.cohort / name).write_text("".join(lines))
        write_json(
            self.cohort / "integrity_report.json",
            {
                "split_statistics": {"partitions": stats},
            },
        )
        write_json(
            self.cohort / "table_manifest.json",
            {
                "tables": {
                    name: sha256(self.cohort / name)
                    for name in ("features.csv", "labels.csv", "metadata.csv")
                },
            },
        )
        self.hashes = {
            name: sha256(self.cohort / name)
            for name in ("realized_split_manifest.json", "feature_schema.json")
        }

    def loader(self):
        return DevelopmentPartitions(self.cohort, self.modeling, self.hashes)

    def test_holdout_values_are_never_parsed_and_ids_never_enter_X(self):
        loader = self.loader()
        train, tune = loader.load("training"), loader.load("tuning")
        self.assertEqual(train.X.shape, (90, 74))
        self.assertEqual(tune.X.shape, (23, 74))
        self.assertEqual(loader.access_log, ["training", "tuning"])
        self.assertFalse(set(train.groups) & set(tune.groups))
        matrix = predictor_array(train, self.features)
        self.assertIsInstance(matrix, np.ndarray)
        self.assertEqual(matrix.dtype, np.dtype("float64"))
        with self.assertRaises(ValueError):
            predictor_array(train, ["subject_id"])

    def test_calibration_and_test_denied_before_table_IO_even_with_lock(self):
        loader = self.loader()
        (self.modeling / "model_lock_manifest.json").write_text("{}")
        for partition in ("calibration", "test", "unknown"):
            with self.subTest(partition=partition), patch.object(Path, "open") as opened:
                with self.assertRaises(PermissionError):
                    loader.load(partition)
                opened.assert_not_called()
        self.assertFalse(hasattr(loader, "validate_lock"))

    def test_general_test_guard_still_requires_lock(self):
        loader = FrozenPartitions(self.cohort, self.modeling)
        with self.assertRaises(PermissionError):
            loader.load("test")

    def test_split_hash_mismatch_fails_before_model_tables(self):
        (self.cohort / "realized_split_manifest.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "manifest differs"):
            self.loader()

    def test_exact_74_feature_order_is_enforced(self):
        write_json(self.cohort / "feature_schema.json", {"feature_names": self.features[::-1]})
        with self.assertRaisesRegex(ValueError, "74-feature"):
            DevelopmentPartitions(self.cohort, self.modeling, {})

    def test_exact_18_MAP_feature_order_is_enforced(self):
        feature_sets = read_json(self.modeling / "feature_sets.json")
        feature_sets["map_only"] = feature_sets["map_only"][::-1]
        write_json(self.modeling / "feature_sets.json", feature_sets)
        with self.assertRaisesRegex(ValueError, "18-feature"):
            self.loader()

    def test_model_table_hash_mismatch_fails(self):
        with (self.cohort / "labels.csv").open("a") as stream:
            stream.write("0\n")
        with self.assertRaisesRegex(ValueError, "table hash"):
            self.loader()


class DevelopmentContractTests(unittest.TestCase):
    def test_class_1_is_extracted_even_for_reversed_classes(self):
        X = np.ones((4, 2))
        np.testing.assert_equal(positive_probabilities(ReversedClassesModel(), X), np.arange(4) / 5)
        with self.assertRaisesRegex(ValueError, "classes must"):
            positive_probabilities(ReversedClassesModel(), X, require_ordered_classes=True)

    def test_probability_shape_and_range_fail_fast(self):
        for probabilities in (
            np.zeros((3, 2)),
            np.full((4, 2), np.nan),
            np.ones((4, 2)),
            np.tile([-0.1, 1.1], (4, 1)),
        ):
            model = SimpleNamespace(
                classes_=np.array([0, 1]), predict_proba=lambda X, p=probabilities: p
            )
            with self.subTest(), self.assertRaises(ValueError):
                positive_probabilities(model, np.ones((4, 2)))

    def test_export_preserves_unsorted_source_order_and_labels(self):
        data = synthetic_data(["map_latest"], 6)
        scores = np.arange(6) / 10
        export = export_predictions(
            data, scores, model="fake", candidate_id="fixed", probability_output=True
        )
        self.assertEqual(export.source_row.tolist(), data.X.index.tolist())
        self.assertEqual(export.true_label.tolist(), data.y.tolist())
        self.assertEqual(export.window_id.tolist(), data.metadata.window_id.tolist())
        self.assertEqual(export.predicted_probability.tolist(), scores.tolist())
        self.assertEqual(set(export.split), {"tuning"})
        with self.assertRaises(ValueError):
            export_predictions(
                data, scores[:-1], model="fake", candidate_id="fixed", probability_output=True
            )

    def test_training_only_imputation_scaling_and_no_tuning_refit(self):
        model = build_estimator("logistic", {"C": 1, "max_iter": 1000})
        train = np.array([[1, np.nan], [2, 1], [3, 3], [4, 5], [5, 7], [6, 9]], dtype=float)
        model.fit(train, np.array([0, 1, 0, 1, 0, 1]))
        medians = model.named_steps["imputer"].statistics_.copy()
        means = model.named_steps["scaler"].mean_.copy()
        self.assertEqual(medians.tolist(), [3.5, 5])
        positive_probabilities(model, np.full((2, 2), 999999.0))
        np.testing.assert_equal(model.named_steps["imputer"].statistics_, medians)
        np.testing.assert_equal(model.named_steps["scaler"].mean_, means)

    def test_prevalence_is_deterministic_training_mean(self):
        model = PrevalenceClassifier().fit(np.zeros((4, 0)), np.array([0, 0, 0, 1]))
        np.testing.assert_equal(positive_probabilities(model, np.zeros((7, 0))), np.full(7, 0.25))

    def test_current_MAP_has_discrimination_only(self):
        metrics = development_metrics([0, 0, 1, 1], [-80, -75, -70, -65], probability_output=False)
        self.assertEqual(metrics["average_precision"], 1)
        self.assertEqual(metrics["prevalence"], 0.5)
        self.assertIsNone(metrics["log_loss"])
        self.assertIsNone(metrics["brier_score"])

    def test_AP_selection_then_loss_then_prespecified_simplicity(self):
        base = {
            "candidate_id": "a",
            "average_precision": 0.2,
            "log_loss": 0.1,
            "simplicity_rank": 0,
        }
        greater_AP = {**base, "candidate_id": "b", "average_precision": 0.3, "log_loss": 100}
        self.assertEqual(select_candidate([base, greater_AP]), greater_AP)
        lower_loss = {**base, "candidate_id": "c", "log_loss": 0.09, "simplicity_rank": 1}
        self.assertEqual(select_candidate([base, lower_loss]), lower_loss)
        simpler = {**base, "candidate_id": "d", "simplicity_rank": -1}
        self.assertEqual(select_candidate([base, simpler]), simpler)

    def test_sanitized_error_contains_traceback_but_no_credentials_or_payloads(self):
        with patch.dict(os.environ, {"TABPFN_TOKEN": "synthetic_secret_for_test_only"}):
            try:
                raise RuntimeError(
                    "Authorization: Bearer synthetic_secret_for_test_only\n"
                    "subject_id=synthetic123\n X=[70.1, 81.2, 90.3]"
                )
            except RuntimeError as exc:
                trace = sanitized_traceback(exc)
        self.assertIn("RuntimeError", trace)
        self.assertIn("Traceback", trace)
        self.assertNotIn("synthetic_secret_for_test_only", trace)
        self.assertNotIn("synthetic123", trace)
        self.assertNotIn("70.1", trace)

    def test_TABPFN_factory_explicit_identifier_and_no_model_fallback(self):
        fake = FakeEstimator()
        constructor = unittest.mock.Mock(return_value=fake)
        with patch.dict(
            sys.modules, {"tabpfn_client": SimpleNamespace(TabPFNClassifier=constructor)}
        ):
            result = build_estimator("tabpfn", {"model_path": "v3.5_default", "random_state": 42})
            self.assertIs(result, fake)
            constructor.assert_called_once_with(model_path="v3.5_default", random_state=42)
            with self.assertRaises(ValueError):
                build_estimator("tabpfn", {"model_path": "v3-fast_default"})

    def test_static_entrypoint_loads_only_training_tuning_and_writes_no_lock(self):
        tree = ast.parse((REPO / "scripts/modeling_development.py").read_text())
        loads = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "load"
        ]
        self.assertEqual([node.args[0].value for node in loads], ["training", "tuning"])
        self.assertNotIn("FrozenPartitions", (REPO / "scripts/modeling_development.py").read_text())
        paths = [
            node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and "model_lock_manifest" in node.value
        ]
        self.assertEqual(paths, ["artifacts/modeling-v01/model_lock_manifest.json"])

    def test_validation_only_cannot_call_hosted_or_fit(self):
        with (
            patch.object(RUNNER, "build_estimator", side_effect=AssertionError("No fit allowed")),
            patch.object(RUNNER, "run_development", side_effect=AssertionError("No execution")),
            patch.object(sys, "argv", ["modeling_development.py", "--validate-only"]),
            redirect_stdout(io.StringIO()),
        ):
            # The historical prepared source is superseded by the sealed held-out guard.
            # Closure/lock must stop legacy development without constructing any model.
            closed = (
                REPO / "artifacts/tabpfn-representation-study-v01/study_closure_manifest.json"
            ).exists()
            self.assertEqual(RUNNER.main(), 2 if closed else 0)

    def test_immutable_model_lock_blocks_any_new_development(self):
        plan = read_json(REPO / "artifacts/modeling-v01/development_plan.json")
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path = root / "artifacts/modeling-v01"
            path.mkdir(parents=True)
            (path / "model_lock_manifest.json").write_text("{}")
            with self.assertRaises(PermissionError):
                RUNNER.validate_inputs(root, plan)

    def test_external_smoke_exact_sampling_and_predictors_only_without_hosted_call(self):
        names = read_json(REPO / "artifacts/modeling-v01/feature_sets.json")["full"]
        training = synthetic_data(names, 300)
        model = FakeEstimator()
        with patch.object(RUNNER, "build_estimator", return_value=model) as factory:
            evidence = RUNNER.external_smoke(training, names)
        factory.assert_called_once_with(
            "tabpfn", {"model_path": "v3.5_default", "random_state": 42}
        )
        positions = np.sort(np.random.default_rng(42).choice(300, 256, replace=False))
        np.testing.assert_equal(model.fit_X, training.X.iloc[positions].to_numpy())
        np.testing.assert_equal(model.fit_y, training.y.iloc[positions].to_numpy())
        self.assertEqual(evidence["status"], "PASS")
        self.assertEqual(evidence["positive_probability_column"], 1)
        self.assertEqual(evidence["classes"], [0, 1])
        self.assertTrue(evidence["row_order_verified_by_reversal_and_duplicates"])

    def test_external_failure_preserves_sanitized_evidence_and_no_completion(self):
        plan = read_json(REPO / "artifacts/modeling-v01/development_plan.json")
        data = synthetic_data(plan["feature_sets"]["full"])
        loader = SimpleNamespace(
            features=plan["feature_sets"]["full"], access_log=["training", "tuning"]
        )
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "artifacts/modeling-v01").mkdir(parents=True)
            plan_path = root / "plan.json"
            write_json(plan_path, plan)
            with (
                patch.object(RUNNER, "validate_environment", return_value={}),
                patch.object(RUNNER, "validate_inputs", return_value=(loader, data, data)),
                patch.object(RUNNER, "external_smoke", side_effect=RuntimeError("network failure")),
                patch.object(RUNNER, "build_estimator") as factory,
                redirect_stdout(io.StringIO()),
                self.assertRaises(SystemExit),
            ):
                RUNNER.run_development(root, plan_path, sha256(plan_path))
            factory.assert_not_called()
            output = root / "artifacts/modeling-v01/development"
            self.assertTrue((output / "failure_traceback.txt").exists())
            failure = read_json(output / "failure_manifest.json")
            self.assertEqual(failure["failed_stage"], "external_training_smoke")
            self.assertEqual(failure["test_evaluation_count"], 0)
            self.assertFalse((output / "development_completion_manifest.json").exists())

    def test_complete_synthetic_execution_saves_all_seven_models_and_alignment(self):
        plan = read_json(REPO / "artifacts/modeling-v01/development_plan.json")
        training = synthetic_data(plan["feature_sets"]["full"], 8)
        tuning = synthetic_data(plan["feature_sets"]["full"], 6)
        loader = SimpleNamespace(
            features=plan["feature_sets"]["full"], access_log=["training", "tuning"]
        )
        fitted = []

        def factory(kind, config):
            model = FakeEstimator()
            fitted.append(model)
            return model

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "artifacts/modeling-v01").mkdir(parents=True)
            plan_path = root / "plan.json"
            write_json(plan_path, plan)
            with (
                patch.object(RUNNER, "validate_environment", return_value={}),
                patch.object(RUNNER, "validate_inputs", return_value=(loader, training, tuning)),
                patch.object(RUNNER, "external_smoke", return_value={"status": "SYNTHETIC_PASS"}),
                patch.object(RUNNER, "build_estimator", side_effect=factory),
                patch.object(
                    RUNNER.joblib, "dump", side_effect=lambda model, path: path.write_text("fake")
                ),
                redirect_stdout(io.StringIO()),
            ):
                manifest = RUNNER.run_development(root, plan_path, sha256(plan_path))
            self.assertEqual(manifest["status"], "COMPLETE")
            self.assertEqual(len(manifest["models_completed"]), 7)
            self.assertEqual(manifest["candidate_count"], 20)
            self.assertEqual(manifest["test_evaluation_count"], 0)
            self.assertFalse((root / "artifacts/modeling-v01/model_lock_manifest.json").exists())
            for model in fitted:
                self.assertEqual(len(model.fit_y), 8)
                np.testing.assert_equal(model.fit_y, training.y.to_numpy())
                self.assertIn(model.fit_X.shape[1], (0, 18, 74))
                self.assertTrue(np.issubdtype(model.fit_X.dtype, np.number))
            output = root / "artifacts/modeling-v01/development"
            predictions = pd.read_csv(output / "tuning_predictions.csv")
            self.assertEqual(len(predictions), 7 * 6)
            for _, rows in predictions.groupby("model_identifier", sort=False):
                self.assertEqual(rows.source_row.tolist(), tuning.X.index.tolist())
                self.assertEqual(rows.true_label.tolist(), tuning.y.tolist())
            for name, digest in manifest["files_sha256"].items():
                self.assertEqual(sha256(output / name), digest)
            with zipfile.ZipFile(output / "development_review.zip") as archive:
                self.assertIn("development_completion_manifest.json", archive.namelist())
                self.assertNotIn("tuning_predictions.csv", archive.namelist())
                self.assertFalse(any(name.startswith("candidates/") for name in archive.namelist()))
                self.assertFalse(any(name.endswith(".joblib") for name in archive.namelist()))


if __name__ == "__main__":
    unittest.main()
