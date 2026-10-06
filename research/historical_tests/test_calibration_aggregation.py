"""Synthetic/mocked-only calibration aggregation tests; no clinical labels loaded."""

import ast
import contextlib
import importlib.util
import io
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

REPO = Path(__file__).resolve().parents[1]
BASE = REPO / "artifacts/modeling-v01/calibration-aggregation-v01"


def module(name, filename):
    spec = importlib.util.spec_from_file_location(name, BASE / filename)
    obj = importlib.util.module_from_spec(spec)
    sys.modules[name] = obj
    spec.loader.exec_module(obj)
    return obj


s = module("aggregation_support", "aggregation_support.py")
worker = module("calibration_aggregate_worker", "aggregation_worker.py")
parent = module("calibration_aggregate_parent", "run_aggregation.py")
from intraop.evaluation import calibration  # noqa: E402


def raw_fixture(n=8):
    values = {name: np.linspace(0.01, 0.97, n) for name in s.MODEL_NAMES}
    values["prevalence"] = np.full(n, 0.035)
    values["current_map"] = -np.arange(70, 70 + n, dtype=float)
    return values, np.arange(n) % 2


class ProcedureTests(unittest.TestCase):
    def test_logit_clips_exact_bounds(self):
        p = np.array([0, 1e-9, 0.5, 1 - 1e-9, 1])
        expected = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
        self.assertTrue(np.array_equal(calibration.logit_input(p)[:, 0], expected))

    def test_one_fixed_LR_configuration_with_CAL_labels(self):
        raw, y = raw_fixture()
        fake = MagicMock()
        fake.coef_ = np.array([[0.4]])
        fake.intercept_ = np.array([-0.2])
        with patch.object(calibration, "LogisticRegression", return_value=fake) as constructor:
            calibration.fit_platt(raw["logistic_map"], y)
        constructor.assert_called_once_with(
            C=1.0,
            solver="lbfgs",
            max_iter=2000,
            tol=1e-4,
            fit_intercept=True,
            class_weight=None,
            random_state=42,
        )
        X, target = fake.fit.call_args.args
        self.assertEqual(fake.fit.call_count, 1)
        self.assertTrue(np.array_equal(target, y))
        self.assertTrue(np.array_equal(X, calibration.logit_input(raw["logistic_map"])))
        self.assertEqual(X.shape, (len(y), 1))
        from sklearn.linear_model import LogisticRegression

        model = LogisticRegression(**s.CALIBRATION_PROTOCOL["parameters"])
        self.assertEqual(model.get_params()["l1_ratio"], 0.0)

    def test_exactly_five_fits_precede_any_metric(self):
        raw, y = raw_fixture()
        events = []

        def fit(p, target):
            events.append("fit")
            self.assertTrue(np.array_equal(target, y))
            return calibration.fit_platt(p, target)

        def metrics(*args, **kwargs):
            events.append("metric")
            return worker.development_metrics(*args, **kwargs)

        mappings, calibrated, _, _ = worker.compute(raw, y, fit=fit, metrics=metrics)
        self.assertEqual(list(mappings), list(s.CALIBRATED_MODELS))
        self.assertEqual(set(calibrated), set(s.CALIBRATED_MODELS))
        self.assertEqual(events[:5], ["fit"] * 5)
        self.assertEqual(events.count("fit"), 5)

    def test_fixed_baselines_untransformed_and_raw_unchanged(self):
        raw, y = raw_fixture()
        before = {name: values.copy() for name, values in raw.items()}
        mappings, calibrated, metrics, tables = worker.compute(raw, y)
        self.assertNotIn("prevalence", mappings)
        self.assertNotIn("current_map", mappings)
        self.assertNotIn("current_map", calibrated)
        self.assertNotIn("current_map/raw", tables)
        self.assertNotIn("brier_score", metrics["current_map/raw"])
        self.assertNotIn("log_loss", metrics["current_map/raw"])
        for name in raw:
            self.assertTrue(np.array_equal(raw[name], before[name]))
        self.assertEqual(len(tables), 11)
        self.assertTrue(all(len(table) == 10 for table in tables.values()))

    def test_diagnostics_cannot_change_fit_or_model_membership(self):
        raw, y = raw_fixture()
        _, reference, _, _ = worker.compute(raw, y)

        def extreme(*args, **kwargs):
            return {
                "windows": len(y),
                "positives": int(y.sum()),
                "average_precision": -999,
                "auroc": 999,
                "probability_output": kwargs["probability_output"],
            }

        mappings, altered, _, _ = worker.compute(raw, y, metrics=extreme)
        self.assertEqual(set(mappings), set(s.CALIBRATED_MODELS))
        for name in reference:
            self.assertTrue(np.array_equal(reference[name], altered[name]))

    def test_single_label_class_stops(self):
        raw, y = raw_fixture()
        with self.assertRaises(ValueError):
            worker.compute(raw, np.zeros_like(y))

    def test_missing_model_stops_without_fit(self):
        raw, y = raw_fixture()
        del raw["tabpfn_full"]
        fit = MagicMock()
        with self.assertRaises(ValueError):
            worker.compute(raw, y, fit=fit)
        fit.assert_not_called()


class LabelFirewallTests(unittest.TestCase):
    def test_ALL_noncalibration_routes_denied_before_authorization(self):
        loader = object.__new__(s.CalibrationLabels)
        with patch.object(loader, "authorize") as authorize:
            for name in ["test", "training", "tuning", "TEST", "../test"]:
                with self.assertRaises(PermissionError):
                    loader.load(name)
            authorize.assert_not_called()

    def label_files(self, root, *, bad_label=False):
        (root / "metadata.csv").write_text(
            "window_id,subject_id,case_id\n"
            "fake_1,cal1,case1\nfake_2,test1,case2\nfake_3,cal1,case1\n"
            "fake_4,cal2,case3\nfake_5,train1,case4\nfake_6,cal2,case3\n"
        )
        # Excluded rows are intentionally unparsable as binary labels.
        (root / "labels.csv").write_text(
            "label\n0\nDO_NOT_PARSE_TEST\n"
            + ("bad\n" if bad_label else "1\n")
            + "0\nDO_NOT_PARSE_TRAIN\n1\n"
        )
        return {
            "subject_to_partition": {
                "cal1": "calibration",
                "cal2": "calibration",
                "test1": "test",
                "train1": "training",
            }
        }

    def test_only_frozen_calibration_label_lines_are_parsed_no_features(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            split = self.label_files(root)
            index = np.array([0, 2, 3, 5])
            with patch.multiple(
                s, ROWS=4, POSITIVES=2, SUBJECTS=2, QUERY_ORDER_HASH=s.row_hash(index)
            ):
                y, chosen = s.read_calibration_labels(root, split)
            self.assertTrue(np.array_equal(y, [0, 1, 0, 1]))
            self.assertTrue(np.array_equal(chosen, index))
            self.assertFalse((root / "features.csv").exists())

    def test_label_mismatch_stops(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            split = self.label_files(root, bad_label=True)
            with patch.multiple(
                s, ROWS=4, POSITIVES=2, SUBJECTS=2, QUERY_ORDER_HASH=s.row_hash([0, 2, 3, 5])
            ):
                with self.assertRaises(ValueError):
                    s.read_calibration_labels(root, split)

    def test_order_hash_mismatch_stops(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            split = self.label_files(root)
            with patch.multiple(s, ROWS=4, POSITIVES=2, SUBJECTS=2, QUERY_ORDER_HASH="wrong"):
                with self.assertRaises(ValueError):
                    s.read_calibration_labels(root, split)

    def test_network_unavailable(self):
        for event in ["socket.connect", "socket.getaddrinfo", "socket.sendto"]:
            with self.assertRaises(PermissionError):
                s.deny_network(event, ())
        s.deny_network("open", ())

    def test_raw_model_and_hosted_imports_blocked(self):
        blocker = s.RawRuntimeBlocker()
        for name in ["tabpfn_client", "tabpfn", "torch", "xgboost", "intraop.models.benchmarks"]:
            with self.assertRaises(PermissionError):
                blocker.find_spec(name)
        self.assertIsNone(blocker.find_spec("sklearn.linear_model"))


class RegistryTests(unittest.TestCase):
    def setUp(self):
        self.registry = json.loads((BASE / "source_registry.json").read_text())
        self.lock = json.loads((s.MODELING / "model_lock_manifest.json").read_text())

    def test_seven_sources_share_exact_row_order_feature_contracts(self):
        s.validate_registry(self.registry, self.lock)
        self.assertEqual(set(self.registry["sources"]), set(s.MODEL_NAMES))
        self.assertTrue(
            all(
                r["rows"] == 2393 and r["query_source_order_sha256"] == s.QUERY_ORDER_HASH
                for r in self.registry["sources"].values()
            )
        )

    def test_source_row_order_mismatch_stops(self):
        self.registry["sources"]["tabpfn_map"]["query_source_order_sha256"] = "wrong"
        with self.assertRaises(PermissionError):
            s.validate_registry(self.registry, self.lock)

    def test_source_result_hash_mismatch_stops(self):
        self.registry["sources"]["tabpfn_full"]["result_sha256"] = "wrong"
        with self.assertRaises(PermissionError):
            s.validate_registry(self.registry, self.lock)

    def test_source_registry_hash_mismatch_stops(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "registry.json"
            s.seal_json(path, self.registry)
            with self.assertRaises(PermissionError):
                s.verify_seal(path, "0" * 64)

    def test_feature_order_mismatch_stops(self):
        self.registry["sources"]["logistic_full"]["ordered_features"].reverse()
        with self.assertRaises(PermissionError):
            s.validate_registry(self.registry, self.lock)

    def test_seven_raw_sources_remain_immutable(self):
        for record in self.registry["sources"].values():
            path = REPO / record["result_path"]
            self.assertEqual(s.file_hash(path), record["result_sha256"])
            self.assertEqual(path.stat().st_mode & 0o222, 0)


class SyntheticLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name) / "calibration"
        self.root.mkdir()
        self.attempt = self.root / "aggregation/attempt_1"
        self.registry = json.loads((BASE / "source_registry.json").read_text())
        self.lock = json.loads((s.MODELING / "model_lock_manifest.json").read_text())
        self.raw, self.y = raw_fixture()
        self.lock["training_context"]["prevalence"] = 0.035
        self.ph = "synthetic-preparation"
        self.stdout = io.StringIO()
        self.patches = []
        for name, value in {
            "CALIBRATION": self.root,
            "ATTEMPT": self.attempt,
            "RECEIPT": self.root / "completion_receipt.json",
            "ROWS": len(self.y),
            "POSITIVES": int(self.y.sum()),
            "SUBJECTS": 2,
        }.items():
            item = patch.object(s, name, value)
            item.start()
            self.patches.append(item)
        self.addCleanup(lambda: [p.stop() for p in reversed(self.patches)])

    def mocked_worker(self):
        loader = MagicMock()
        loader.preparation_hash = self.ph
        loader.registry = self.registry
        loader.lock = self.lock
        loader.access_log = ["calibration"]
        loader.load.return_value = (self.y, np.arange(len(self.y)))

        def plot(path, tables):
            s.atomic_binary(path, lambda stream: stream.write(b"\x89PNG\r\n\x1a\nsynthetic"))

        with (
            patch.object(s, "CalibrationLabels", return_value=loader),
            patch.object(s, "raw_outputs", return_value=self.raw),
            patch.object(
                s,
                "row_hash",
                side_effect=lambda index: (
                    s.QUERY_ORDER_HASH if len(index) == len(self.y) else "wrong"
                ),
            ),
            patch.object(worker, "plot_reliability", side_effect=plot),
        ):
            result = worker.execute()
        loader.load.assert_called_once_with("calibration")
        return result

    def stage(self):
        self.attempt.mkdir(parents=True)
        self.assertEqual(self.mocked_worker(), 0)
        return self.attempt / "publication"

    def validate(self, root):
        with (
            patch.object(s, "raw_outputs", return_value=self.raw),
            patch.object(s, "row_hash", return_value=s.QUERY_ORDER_HASH),
        ):
            return parent.validate_publication(root, self.registry, self.lock)

    def test_raw_and_calibrated_predictions_retained_no_identifiers(self):
        root = self.stage()
        self.validate(root)
        for name in s.MODEL_NAMES:
            with np.load(root / "predictions" / f"{name}.npz", allow_pickle=False) as arrays:
                key = "raw_score" if name == "current_map" else "raw_probability"
                self.assertTrue(np.array_equal(arrays[key], self.raw[name]))
                self.assertEqual(
                    "calibrated_probability" in arrays.files, name in s.CALIBRATED_MODELS
                )
                self.assertNotIn("subject_id", arrays.files)
                self.assertNotIn("case_id", arrays.files)
        self.assertFalse((self.root / "completion_receipt.json").exists())

    def test_wrong_label_checkpoint_stops_validation(self):
        root = self.stage()
        with patch.object(parent.np, "load", return_value=np.zeros(len(self.y), dtype=int)):
            with self.assertRaises(ValueError):
                self.validate(root)

    def test_wrong_platt_parameters_stop(self):
        root = self.stage()
        original = s.verify_seal

        def verify(path, *args):
            obj, sha = original(path, *args)
            if Path(path).name == "calibration_parameters.json":
                obj["logistic_map"]["fitting_parameters"]["C"] = 2
            return obj, sha

        with patch.object(s, "verify_seal", side_effect=verify):
            with self.assertRaises(ValueError):
                self.validate(root)

    def test_optional_figure_publication_failure_retained_no_receipt(self):
        self.attempt.mkdir(parents=True)
        loader = MagicMock()
        loader.preparation_hash = self.ph
        loader.registry = self.registry
        loader.lock = self.lock
        loader.access_log = ["calibration"]
        loader.load.return_value = (self.y, np.arange(len(self.y)))
        with (
            patch.object(s, "CalibrationLabels", return_value=loader),
            patch.object(s, "raw_outputs", return_value=self.raw),
            patch.object(worker, "plot_reliability", side_effect=RuntimeError("PRIVATE_SENTINEL")),
        ):
            self.assertEqual(worker.execute(), 2)
        self.assertTrue((self.attempt / "failure.json").exists())
        self.assertFalse((self.root / "completion_receipt.json").exists())
        self.assertNotIn("PRIVATE_SENTINEL", (self.attempt / "failure.json").read_text())

    def test_complete_synthetic_execution_receipt_immutable_and_not_duplicated(self):
        def child(*args, **kwargs):
            return types.SimpleNamespace(returncode=self.mocked_worker())

        prep = {
            "scope": {"phase": "calibration", "operation": "final_aggregation", "attempt": 1},
            "bound_files_sha256": {"synthetic": "hash"},
        }
        with (
            patch.object(
                s, "checked_state", return_value=(prep, self.ph, self.lock, self.registry)
            ),
            patch.object(parent.subprocess, "run", side_effect=child) as launch,
            patch.object(s, "raw_outputs", return_value=self.raw),
            patch.object(s, "row_hash", return_value=s.QUERY_ORDER_HASH),
            patch.object(parent, "verify_calibration_complete") as verify,
            contextlib.redirect_stdout(self.stdout),
        ):
            self.assertEqual(parent.execute(), 0)
            verify.assert_called_once()
            before = s.receipt_files(self.root)
            with self.assertRaises(FileExistsError):
                parent.execute()
            self.assertEqual(launch.call_count, 1)
            self.assertEqual(s.receipt_files(self.root), before)
        receipt, _ = s.verify_seal(self.root / "completion_receipt.json")
        self.assertEqual(receipt["models"], list(s.MODEL_NAMES))
        self.assertEqual(receipt["Platt_models"], list(s.CALIBRATED_MODELS))
        self.assertFalse(receipt["TEST_accessed"])
        self.assertEqual(receipt["raw_model_fit_or_predict_calls"], 0)
        self.assertEqual((self.root / "calibration_parameters.json").stat().st_mode & 0o222, 0)

    def test_publication_error_stops_without_receipt_or_retry(self):
        def child(*args, **kwargs):
            return types.SimpleNamespace(returncode=self.mocked_worker())

        original = s.copy_immutable

        def copy(source, destination):
            if Path(destination).parent == self.root:
                raise OSError("PRIVATE_PUBLICATION_ERROR")
            return original(source, destination)

        prep = {
            "scope": {"phase": "calibration", "operation": "final_aggregation", "attempt": 1},
            "bound_files_sha256": {"synthetic": "hash"},
        }
        with (
            patch.object(
                s, "checked_state", return_value=(prep, self.ph, self.lock, self.registry)
            ),
            patch.object(parent.subprocess, "run", side_effect=child) as launch,
            patch.object(s, "raw_outputs", return_value=self.raw),
            patch.object(s, "row_hash", return_value=s.QUERY_ORDER_HASH),
            patch.object(s, "copy_immutable", side_effect=copy),
            contextlib.redirect_stdout(self.stdout),
        ):
            self.assertEqual(parent.execute(), 2)
            self.assertFalse(s.RECEIPT.exists())
            with self.assertRaises(FileExistsError):
                parent.execute()
            self.assertEqual(launch.call_count, 1)
        self.assertNotIn(
            "PRIVATE_PUBLICATION_ERROR", (self.attempt / "supervisor_stop.json").read_text()
        )

    def test_receipt_verification_error_stops_and_retains_evidence(self):
        def child(*args, **kwargs):
            return types.SimpleNamespace(returncode=self.mocked_worker())

        prep = {
            "scope": {"phase": "calibration", "operation": "final_aggregation", "attempt": 1},
            "bound_files_sha256": {"synthetic": "hash"},
        }
        with (
            patch.object(
                s, "checked_state", return_value=(prep, self.ph, self.lock, self.registry)
            ),
            patch.object(parent.subprocess, "run", side_effect=child) as launch,
            patch.object(s, "raw_outputs", return_value=self.raw),
            patch.object(s, "row_hash", return_value=s.QUERY_ORDER_HASH),
            patch.object(parent, "verify_calibration_complete", side_effect=ValueError("PRIVATE")),
            contextlib.redirect_stdout(self.stdout),
        ):
            self.assertEqual(parent.execute(), 2)
            self.assertTrue((self.attempt / "supervisor_stop.json").exists())
            before = s.receipt_files(self.root)
            with self.assertRaises(FileExistsError):
                parent.execute()
            self.assertEqual(launch.call_count, 1)
            self.assertEqual(s.receipt_files(self.root), before)

    def test_native_crash_stops_no_retry(self):
        prep = {"scope": {}}
        with (
            patch.object(
                s, "checked_state", return_value=(prep, self.ph, self.lock, self.registry)
            ),
            patch.object(
                parent.subprocess, "run", return_value=types.SimpleNamespace(returncode=-11)
            ) as launch,
            contextlib.redirect_stdout(self.stdout),
        ):
            self.assertEqual(parent.execute(), 2)
            with self.assertRaises(FileExistsError):
                parent.execute()
            self.assertEqual(launch.call_count, 1)
        self.assertEqual(
            s.read_json(self.attempt / "supervisor_stop.json")["status"], "INDETERMINATE"
        )

    def test_existing_final_artifact_blocks_before_fitting(self):
        s.atomic_json(self.root / "metrics.json", {})
        with patch.object(s, "checked_state", return_value=({}, self.ph, self.lock, self.registry)):
            with self.assertRaises(FileExistsError):
                s.reserve()


class ScopeAndInstalledTests(unittest.TestCase):
    def test_worker_only_calibration_load_and_no_raw_runtimes_or_refits(self):
        tree = ast.parse((BASE / "aggregation_worker.py").read_text())
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
        methods = [n.func.attr for n in calls if isinstance(n.func, ast.Attribute)]
        self.assertNotIn("fit", methods)
        self.assertNotIn("predict_proba", methods)
        loads = [
            ast.literal_eval(n.args[0])
            for n in calls
            if isinstance(n.func, ast.Attribute) and n.func.attr == "load"
        ]
        self.assertEqual(loads, ["calibration"])
        text = (BASE / "aggregation_worker.py").read_text()
        for forbidden in [
            "build_estimator",
            "joblib",
            "TabPFNClassifier",
            "XGBClassifier",
            "select_candidate",
        ]:
            self.assertNotIn(forbidden, text)

    def test_launcher_one_interpreter_no_credentials_no_arguments(self):
        text = (BASE / "run_aggregation.sh").read_text()
        self.assertIn(s.PYTHON + " -I -u", text)
        self.assertIn('if [ "$#" -ne 0 ]', text)
        self.assertNotIn("tabpfn.env", text)
        self.assertNotIn("TOKEN", text)

    def test_installed_state_real_sources_all_protected_hashes(self):
        prep, _, lock, registry = s.checked_state()
        self.assertEqual(len(prep["protected_inventory"]), 199)
        self.assertEqual(registry["model_order"], list(s.MODEL_NAMES))
        self.assertEqual(lock["calibration_protocol"], s.CALIBRATION_PROTOCOL)

    def test_lock_protocol_amendments_and_all_seven_workers_unchanged(self):
        prep, _, _, _ = s.checked_state()
        for relative, record in prep["protected_inventory"].items():
            self.assertEqual(s.file_hash(REPO / relative), record["sha256"])

    def test_TEST_presence_blocks_before_any_label_loading(self):
        original = Path.exists
        for forbidden in [
            s.MODELING / "locked_execution/test",
            s.MODELING / "locked_execution/approvals/test.json",
        ]:
            with patch.object(
                Path, "exists", lambda p, target=forbidden: p == target or original(p)
            ):
                with self.assertRaises(PermissionError):
                    s.checked_state()

    def test_source_registry_swap_blocks_before_label_read(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.json"
            s.seal_json(path, {"modified": True})
            with patch.object(s, "REGISTRY", path):
                with self.assertRaises(PermissionError):
                    s.checked_state()

    def test_final_receipt_presence_blocks(self):
        original = Path.exists
        with patch.object(Path, "exists", lambda p: p == s.RECEIPT or original(p)):
            with self.assertRaises(PermissionError):
                s.checked_state()

    def test_no_real_aggregation_fitting_or_TEST_started(self):
        self.assertFalse(s.ATTEMPT.exists())
        self.assertFalse(s.RECEIPT.exists())
        self.assertFalse((s.CALIBRATION / "calibration_parameters.json").exists())
        self.assertFalse((s.MODELING / "locked_execution/test").exists())
        self.assertFalse((s.MODELING / "locked_execution/approvals/test.json").exists())


if __name__ == "__main__":
    unittest.main()
