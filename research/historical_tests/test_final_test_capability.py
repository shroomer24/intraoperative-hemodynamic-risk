"""Synthetic/mocked final TEST capability tests; never load real held-out tables."""

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
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
BASE = REPO / "artifacts/modeling-v01/final-test-v01"
sys.path.insert(0, str(BASE))


def module(name, filename):
    spec = importlib.util.spec_from_file_location(name, BASE / filename)
    obj = importlib.util.module_from_spec(spec)
    sys.modules[name] = obj
    spec.loader.exec_module(obj)
    return obj


s = module("final_test_support", "final_test_support.py")
worker = module("final_test_capability_worker", "final_test_worker.py")
reporting = module("final_test_reporting", "final_test_reporting.py")
parent = module("final_test_capability_parent", "run_final_test.py")


def fixture():
    lock = json.loads((s.MODELING / "model_lock_manifest.json").read_text())
    registry = json.loads(s.REGISTRY.read_text())
    features = lock["feature_sets"]["full"]

    def data(n, start, prefix):
        index = pd.Index(np.arange(start, start + n), name="source_row")
        X = pd.DataFrame(np.arange(n * 74).reshape(n, 74) / 100, index=index, columns=features)
        m = pd.DataFrame(
            {
                "subject_id": [prefix + str(i // (n // 2)) for i in range(n)],
                "case_id": [prefix + "case"] * n,
                "anchor_time_seconds": np.full(n, 10.0),
            },
            index=index,
        )
        return types.SimpleNamespace(
            X=X, y=pd.Series(np.arange(n) % 2, index=index), metadata=m, groups=m.subject_id
        )

    train, query = data(8, 0, "SYN_TRAIN_"), data(6, 100, "SYN_TEST_")
    lock["training_context"].update(rows=8, subjects=2, positive_labels=4, prevalence=0.5)
    registry["fixed_training_prevalence"] = 0.5
    registry["training_context"] = lock["training_context"]
    registry["training_context_sha256"] = s.a.digest(lock["training_context"])
    for name in s.MODELS:
        record = registry["sources"][name]
        record["training_X_sha256"] = s.a.array_hash(
            s.a.full_support.predictor_array(train, lock["models"][name]["features"])
        )
        record["training_y_sha256"] = s.a.array_hash(train.y)
    mappings = json.loads((s.a.CALIBRATION / "calibration_parameters.json").read_text())
    return lock, registry, train, query, mappings


def raw_fixture(query):
    raw = {name: np.linspace(0.01, 0.95, len(query.X)) for name in s.MODELS}
    raw["prevalence"] = np.full(len(query.X), 0.5)
    raw["current_map"] = -query.X.map_latest.to_numpy()
    return raw


def metadata(rows=6, features=18, path="v3.5_default"):
    return {
        "tabpfn_config": {"model_path": path, "random_state": 42},
        "billing_model_version": "v3.5",
        "execution_mode": "standard",
        "test_set_num_rows": rows,
        "test_set_num_cols": features,
        "classes": [0, 1],
    }


class FirewallTests(unittest.TestCase):
    def test_rejected_tuning_calibration_before_authorize_or_reader(self):
        loader = s.TestPartitions.__new__(s.TestPartitions)
        loader.authorize = MagicMock()
        for partition in ["tuning", "calibration", "unknown"]:
            with self.assertRaises(PermissionError):
                loader.load(partition)
        loader.authorize.assert_not_called()

    def test_TEST_cannot_load_without_approval(self):
        with patch.object(s, "checked_state", side_effect=PermissionError("No approval")):
            with self.assertRaises(PermissionError):
                s.TestPartitions()

    def test_TEST_cannot_load_before_reservation(self):
        lock, registry, _, _, mappings = fixture()
        with patch.object(
            s, "checked_state", return_value=({}, "synthetic", lock, registry, mappings)
        ):
            loader = s.TestPartitions()
            with (
                tempfile.TemporaryDirectory() as d,
                patch.object(s, "RESERVATION", Path(d) / "absent.json"),
            ):
                with self.assertRaises(PermissionError):
                    loader.load("test")

    def test_existing_empty_failed_or_complete_directory_blocks_rerun(self):
        state = ({}, "synthetic", *fixture()[:2], fixture()[-1])
        with (
            tempfile.TemporaryDirectory() as d,
            patch.object(s, "ROOT", Path(d)),
            patch.object(s, "checked_state", return_value=state),
        ):
            with self.assertRaises(FileExistsError):
                s.reserve()

    def test_reservation_precedes_any_loading_and_never_copies_TEST_values(self):
        lock, registry, _, _, mappings = fixture()
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / "test"
            approval = Path(d) / "approvals/test.json"
            template = Path(d) / "template.json"
            s.a.seal_json(template, {"synthetic": True})
            with (
                patch.multiple(
                    s,
                    ROOT=root,
                    RESERVATION=root / "execution_manifest.json",
                    APPROVAL=approval,
                    TEMPLATE=template,
                ),
                patch.object(
                    s, "checked_state", return_value=({}, "synthetic", lock, registry, mappings)
                ),
                patch.object(
                    s.a.full_support,
                    "_load_partition",
                    side_effect=AssertionError("reader forbidden"),
                ),
            ):
                s.reserve()
                reservation, _ = s.a.verify_seal(root / "execution_manifest.json")
                self.assertEqual(reservation["status"], "RESERVED")
                self.assertEqual(reservation["phase"], "test")
                self.assertTrue(approval.exists())
                with self.assertRaises(FileExistsError):
                    s.reserve()

    def test_reservation_changes_block_loading(self):
        lock, registry, _, _, mappings = fixture()
        with patch.object(
            s, "checked_state", return_value=({}, "synthetic", lock, registry, mappings)
        ):
            loader = s.TestPartitions()
            with patch.object(
                s.a,
                "verify_seal",
                return_value=({"status": "RESERVED", "phase": "test", "attempt": 2}, "hash"),
            ):
                with self.assertRaises(PermissionError):
                    loader.authorize()


class ScientificTests(unittest.TestCase):
    def setUp(self):
        self.lock, self.registry, self.train, self.query, self.mappings = fixture()
        self.loader = types.SimpleNamespace(lock=self.lock, registry=self.registry)

    def test_exact_18_and_74_feature_contracts(self):
        for name, count, h in [("tabpfn_map", 18, s.MAP_HASH), ("tabpfn_full", 74, s.FULL_HASH)]:
            X, q, c = s.input_contract(self.loader, name, self.train, self.query)
            self.assertEqual(X.shape, (8, count))
            self.assertEqual(q.shape, (6, count))
            self.assertEqual(c["feature_contract_sha256"], h)
            self.assertTrue(c["metadata_excluded_from_X"])
            self.assertEqual(c["constructor"], {"model_path": "v3.5_default", "random_state": 42})

    def test_appended_training_rows_rejected(self):
        self.train.X = pd.concat([self.train.X, self.train.X.iloc[:1]])
        with self.assertRaises(s.a.ScientificFailure):
            s.input_contract(self.loader, "tabpfn_map", self.train, self.query)

    def test_changed_training_values_or_labels_rejected(self):
        self.train.X.iloc[0, 0] += 1
        with self.assertRaises(s.a.ScientificFailure):
            s.input_contract(self.loader, "tabpfn_full", self.train, self.query)
        self.train.X.iloc[0, 0] -= 1
        self.train.y.iloc[0] = 1
        with self.assertRaises(s.a.ScientificFailure):
            s.input_contract(self.loader, "tabpfn_full", self.train, self.query)

    def test_overlapping_subjects_rejected(self):
        self.query.groups = self.query.groups.copy()
        self.query.groups.iloc[0] = self.train.groups.iloc[0]
        with self.assertRaises(s.a.ScientificFailure):
            s.input_contract(self.loader, "tabpfn_map", self.train, self.query)

    def test_changed_feature_order_or_seed_rejected(self):
        for mode in ["features", "seed"]:
            lock, registry, train, query, _ = fixture()
            if mode == "features":
                lock["models"]["tabpfn_map"]["features"].reverse()
            else:
                lock["models"]["tabpfn_map"]["selected_configuration"]["hyperparameters"][
                    "random_state"
                ] = 43
            with self.assertRaises(s.a.ScientificFailure):
                s.input_contract(
                    types.SimpleNamespace(lock=lock, registry=registry), "tabpfn_map", train, query
                )

    def test_mapping_change_rejected(self):
        self.mappings["tabpfn_map"]["coefficient"] += 1e-9
        with self.assertRaises(PermissionError):
            s.verify_mappings(self.mappings)

    def test_exact_frozen_mapping_and_clipping_no_fit(self):
        p = np.array([0, 1e-12, 0.5, 1 - 1e-12, 1])
        from scipy.special import expit

        from intraop.evaluation import calibration

        with patch.object(calibration, "fit_platt", side_effect=AssertionError("No refit")):
            for mapping in self.mappings.values():
                actual = reporting.apply_mapping(p, mapping)
                clipped = np.clip(p, 1e-6, 1 - 1e-6)
                expected = expit(
                    mapping["coefficient"] * np.log(clipped / (1 - clipped)) + mapping["intercept"]
                )
                self.assertTrue(np.array_equal(actual, expected))
                self.assertTrue(np.array_equal(actual, calibration.apply_platt(p, mapping)))

    def test_identity_accepts_only_exact_two_paths(self):
        for path in ["v3.5_default", s.a.full_support.CANONICAL]:
            required = s.a.full_support.compatible_metadata(
                metadata(path=path), rows=6, features=18
            )
            self.assertEqual(required["model_path"], path)

    def test_identity_rejects_version_mode_suffix_normalization(self):
        for field, value in [
            ("path", "v3"),
            ("path", "v3.5_fast"),
            ("path", s.a.full_support.CANONICAL + "/"),
            ("path", "/app/tabpfn_models/../tabpfn_models/tabpfn-v3.5-20260909.safetensors"),
            ("billing_model_version", "3.5"),
            ("execution_mode", "cache"),
        ]:
            raw = metadata()
            if field == "path":
                raw["tabpfn_config"]["model_path"] = value
            else:
                raw[field] = value
            with self.assertRaises(s.a.ScientificFailure):
                s.a.full_support.compatible_metadata(raw, rows=6, features=18)

    def test_identity_missing_required_fields_fails(self):
        for key in ["tabpfn_config", "billing_model_version", "execution_mode"]:
            raw = metadata()
            del raw[key]
            with self.assertRaises(s.a.ScientificFailure):
                s.a.full_support.compatible_metadata(raw, rows=6, features=18)


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.lock, self.registry, self.train, self.query, _ = fixture()
        _, _, self.c = s.input_contract(
            types.SimpleNamespace(lock=self.lock, registry=self.registry),
            "tabpfn_map",
            self.train,
            self.query,
        )
        p = np.linspace(0.01, 0.9, 6)
        self.p = np.column_stack([1 - p, p])

    def test_immediate_immutable_checkpoint_before_optional_failure(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "cp"
            stages = []
            with patch.object(
                s.a.full_support,
                "optional_metadata",
                side_effect=TypeError("secret URL arbitrary body"),
            ):
                with self.assertRaises(TypeError):
                    s.checkpoint(
                        path, self.p, [0, 1], self.c, self.c, metadata(), mark=stages.append
                    )
            self.assertTrue((path / "manifest.json").exists())
            self.assertTrue(np.array_equal(np.load(path / "probabilities.npy"), self.p))
            self.assertFalse((path / "probabilities.npy").stat().st_mode & 0o222)
            self.assertLess(
                stages.index("PROBABILITY_CHECKPOINT_PUBLISHED"),
                stages.index("OPTIONAL_METADATA_PROCESSING_STARTED"),
            )

    def test_completed_checkpoint_cannot_be_replaced(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "cp"
            s.checkpoint(path, self.p, [0, 1], self.c, self.c, metadata(), mark=lambda stage: None)
            before = s.a.receipt_files(path)
            with self.assertRaises(FileExistsError):
                s.checkpoint(
                    path, self.p, [0, 1], self.c, self.c, metadata(), mark=lambda stage: None
                )
            self.assertEqual(s.a.receipt_files(path), before)

    def test_classes_shape_nonfinite_and_probability_bounds_fail(self):
        with tempfile.TemporaryDirectory() as d:
            for p, classes in [
                (self.p, [1, 0]),
                (self.p, ["0", "1"]),
                (self.p[:-1], [0, 1]),
                (np.full((6, 2), np.nan), [0, 1]),
                (np.full((6, 2), 1.1), [0, 1]),
            ]:
                with self.assertRaises(s.a.ScientificFailure):
                    s.checkpoint(
                        Path(d) / "cp",
                        p,
                        classes,
                        self.c,
                        self.c,
                        metadata(),
                        mark=lambda stage: None,
                    )
            self.assertFalse((Path(d) / "cp").exists())

    def test_unknown_metadata_and_credentials_are_discarded(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "cp"
            raw = {
                **metadata(),
                "authorization": "secret-synthetic",
                "uuid": "private-id",
                "url": "https://forbidden.example",
            }
            _, optional = s.checkpoint(
                path, self.p, [0, 1], self.c, self.c, raw, mark=lambda stage: None
            )
            content = (path / "manifest.json").read_text() + json.dumps(optional)
            for value in ["secret-synthetic", "private-id", "forbidden.example", "authorization"]:
                self.assertNotIn(value, content)

    def test_transport_failure_bypasses_sdk_retry_and_discards_details(self):
        calls = []

        class Client:
            def send(self, *args, **kwargs):
                calls.append(1)
                raise RuntimeError("credential unknown-response-body")

        httpx = types.SimpleNamespace(Client=Client)
        with tempfile.TemporaryDirectory() as d, s.one_shot_transport(httpx, Path(d)):
            with self.assertRaises(s.a.full_support.TransportStopped):
                try:
                    Client().send()
                except Exception:
                    Client().send()
        self.assertEqual(len(calls), 1)

    def test_runtime_isolation(self):
        for name, forbidden in [
            ("xgboost", "tabpfn_client"),
            ("xgboost", "torch"),
            ("tabpfn_map", "xgboost"),
            ("logistic_full", "tabpfn_client"),
        ]:
            with self.assertRaises(PermissionError):
                s.RuntimeBlocker(name).find_spec(forbidden)


class BootstrapTests(unittest.TestCase):
    def test_draws_match_locked_generator_and_preserve_subject_multiplicity(self):
        from intraop.evaluation.locked_execution import bootstrap_subject_indices

        groups = np.array(["a", "a", "b", "b", "b", "c"])
        y = np.array([0, 1, 0, 1, 0, 1])
        _, draws, codes, _ = reporting.subject_bootstrap(
            y, groups, {"m": np.linspace(0, 1, 6)}, replicates=20, seed=42
        )
        expected = list(bootstrap_subject_indices(groups, replicates=20, seed=42))
        for draw, index in zip(draws, expected, strict=True):
            actual = np.concatenate([np.flatnonzero(codes == i) for i in draw])
            self.assertTrue(np.array_equal(actual, index))
            for ordinal in range(3):
                self.assertEqual(
                    np.isin(actual, np.flatnonzero(codes == ordinal)).sum(),
                    int((draw == ordinal).sum()) * int((codes == ordinal).sum()),
                )

    def test_same_indices_shared_across_all_outputs(self):
        y = np.array([0, 1, 0, 1])
        groups = np.array(["a", "a", "b", "b"])
        score = np.array([0.1, 0.9, 0.2, 0.8])
        result, draws, _, _ = reporting.subject_bootstrap(
            y, groups, {"raw": score, "calibrated": score}, replicates=100
        )
        self.assertEqual(result["raw"], result["calibrated"])
        self.assertEqual(draws.shape, (100, 2))

    def test_single_class_replicates_undefined_and_counts_reported(self):
        y = np.array([0, 0, 1, 1])
        groups = np.array(["a", "a", "b", "b"])
        result, _, _, defined = reporting.subject_bootstrap(
            y, groups, {"m": np.arange(4)}, replicates=1000
        )
        self.assertGreater((~defined).sum(), 0)
        for metric in result["m"].values():
            self.assertEqual(metric["valid_replicates"], int(defined.sum()))
            self.assertEqual(metric["undefined_replicates"], int((~defined).sum()))
            self.assertEqual(len(metric["percentile_95"]), 2)

    def test_all_undefined_has_no_percentile(self):
        result, _, _, _ = reporting.subject_bootstrap(
            np.zeros(4), ["a", "a", "b", "b"], {"m": np.arange(4)}, replicates=4
        )
        for metric in result["m"].values():
            self.assertIsNone(metric["percentile_95"])
            self.assertEqual(metric["undefined_replicates"], 4)


class SyntheticExecutionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "test"
        self.lock, self.registry, self.train, self.query, self.mappings = fixture()
        self.raw = raw_fixture(self.query)
        self.ph = "synthetic"
        self.loader = MagicMock()
        self.loader.lock = self.lock
        self.loader.registry = self.registry
        self.loader.preparation_hash = self.ph
        self.loader.load.side_effect = lambda p: self.train if p == "training" else self.query
        self.loader.access_log = ["training", "test"]
        self.patches = []
        for key, value in {
            "REPO": Path(self.temp.name),
            "ROOT": self.root,
            "RESERVATION": self.root / "execution_manifest.json",
            "RECEIPT": self.root / "completion_receipt.json",
        }.items():
            p = patch.object(s, key, value)
            p.start()
            self.patches.append(p)
        self.addCleanup(lambda: [p.stop() for p in reversed(self.patches)])
        self.out = io.StringIO()

    def synthetic_worker(self, name):
        if name in s.HOSTED:
            fake = MagicMock()
            fake.get_params.return_value = s.a.CONSTRUCTOR.copy()
            p = self.raw[name]
            fake.predict_proba.return_value = np.column_stack([1 - p, p])
            fake.classes_ = np.array([0, 1])
            fake._last_meta = metadata(features=len(self.lock["models"][name]["features"]))
            sdk = types.ModuleType("tabpfn_client")
            sdk.TabPFNClassifier = MagicMock(return_value=fake)
            with (
                patch.dict(sys.modules, {"tabpfn_client": sdk}),
                patch.object(s, "one_shot_transport", return_value=contextlib.nullcontext()),
                patch.object(s, "TestPartitions", return_value=self.loader),
                patch.object(sys, "meta_path", list(sys.meta_path)),
            ):
                code = worker.execute(name)
            sdk.TabPFNClassifier.assert_called_once_with(model_path="v3.5_default", random_state=42)
            fake.fit.assert_called_once()
            self.assertTrue(np.array_equal(fake.fit.call_args.args[1], self.train.y))
            fake.predict_proba.assert_called_once()
            return code
        fake = MagicMock()
        p = self.raw[name]
        fake.predict_proba.return_value = np.column_stack([1 - p, p])
        fake.classes_ = np.array([0, 1])
        with (
            patch.object(s, "TestPartitions", return_value=self.loader),
            patch.object(worker, "conventional_model", return_value=fake),
            patch.object(sys, "meta_path", list(sys.meta_path)),
            patch.object(sys, "addaudithook"),
        ):
            code = worker.execute(name)
        fake.fit.assert_not_called()
        return code

    def reserve(self):
        if self.root.exists():
            raise FileExistsError("Prior TEST retained")
        self.root.mkdir()
        s.a.seal_json(s.RESERVATION, {"synthetic": True})
        return (
            {
                "bound_files_sha256": {"synthetic": "hash"},
                "python_version": "synthetic",
                "package_versions": {},
            },
            self.ph,
            self.lock,
            self.registry,
            self.mappings,
        )

    def fake_plot(self, path, tables):
        s.a.atomic_binary(path, lambda stream: stream.write(b"\x89PNG\r\n\x1a\nsynthetic"))

    def test_complete_mocked_one_time_execution_and_no_refits(self):
        def child(args, **kwargs):
            return types.SimpleNamespace(returncode=self.synthetic_worker(args[-1]))

        with (
            patch.object(s, "reserve", side_effect=self.reserve),
            patch.object(parent.subprocess, "run", side_effect=child),
            patch.object(s, "TestPartitions", return_value=self.loader),
            patch.object(reporting, "plot", side_effect=self.fake_plot),
            contextlib.redirect_stdout(self.out),
        ):
            self.assertEqual(parent.execute(), 0)
            before = s.a.receipt_files(self.root)
            receipt, _ = s.a.verify_seal(s.RECEIPT)
            self.assertEqual(receipt["Platt_refit_calls"], 0)
            self.assertEqual(receipt["models"], list(s.MODELS))
            self.assertEqual(
                receipt["model_fit_calls"],
                {name: 1 if name in s.HOSTED else 0 for name in s.MODELS},
            )
            with self.assertRaises(FileExistsError):
                parent.execute()
            self.assertEqual(before, s.a.receipt_files(self.root))
        for name in s.MODELS:
            with np.load(self.root / "predictions" / f"{name}.npz") as arrays:
                self.assertEqual("calibrated_probability" in arrays.files, name in s.LEARNED)
                for key in arrays.files:
                    self.assertNotIn("subject", key)
                    self.assertNotIn("case", key)

    def test_native_failure_preserves_reservation_and_stops_later_models(self):
        with (
            patch.object(s, "reserve", side_effect=self.reserve),
            patch.object(
                parent.subprocess, "run", return_value=types.SimpleNamespace(returncode=-11)
            ) as child,
            contextlib.redirect_stdout(self.out),
        ):
            self.assertEqual(parent.execute(), 2)
            self.assertEqual(child.call_count, 1)
            self.assertTrue(s.RESERVATION.exists())
            failure, _ = s.a.verify_seal(self.root / "failure.json")
            self.assertEqual(failure["signal"], 11)
            self.assertFalse(s.RECEIPT.exists())
            with self.assertRaises(FileExistsError):
                parent.execute()

    def test_probability_publication_failure_preserves_checkpoints(self):
        def child(args, **kwargs):
            return types.SimpleNamespace(returncode=self.synthetic_worker(args[-1]))

        with (
            patch.object(s, "reserve", side_effect=self.reserve),
            patch.object(parent.subprocess, "run", side_effect=child),
            patch.object(s, "TestPartitions", return_value=self.loader),
            patch.object(
                reporting, "plot", side_effect=TypeError("unknown patient secret response")
            ),
            contextlib.redirect_stdout(self.out),
        ):
            self.assertEqual(parent.execute(), 2)
        self.assertTrue(
            (self.root / "workers/tabpfn_full/probability_checkpoint/manifest.json").exists()
        )
        self.assertTrue(s.RESERVATION.exists())
        self.assertFalse(s.RECEIPT.exists())
        self.assertEqual(s.a.read_json(self.root / "failure.json")["exception_class"], "TypeError")
        self.assertNotIn("secret", (self.root / "failure.json").read_text())

    def test_metrics_cannot_change_models_mappings_or_baselines(self):
        before = s.a.digest(self.registry), s.a.digest(self.mappings)
        calibrated, metrics, tables, *_ = reporting.diagnostics(
            self.raw, self.query.y, self.query.groups, self.mappings
        )
        self.assertEqual(before, (s.a.digest(self.registry), s.a.digest(self.mappings)))
        self.assertEqual(set(calibrated), set(s.LEARNED))
        self.assertEqual(len(metrics), 12)
        self.assertEqual(len(tables), 11)
        self.assertNotIn("brier_score", metrics["current_map/raw"])
        self.assertNotIn("log_loss", metrics["current_map/raw"])
        self.assertTrue(np.array_equal(self.raw["prevalence"], np.full(6, 0.5)))
        self.assertTrue(np.array_equal(self.raw["current_map"], -self.query.X.map_latest))


class StaticAndInstalledTests(unittest.TestCase):
    def test_no_calibration_fit_call_and_only_TabPFN_fit(self):
        for filename in ["run_final_test.py", "final_test_reporting.py", "final_test_support.py"]:
            tree = ast.parse((BASE / filename).read_text())
            calls = [
                n.func.attr
                for n in ast.walk(tree)
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
            ]
            self.assertNotIn("fit", calls)
            self.assertNotIn("fit_platt", (BASE / filename).read_text())
        tree = ast.parse((BASE / "final_test_worker.py").read_text())
        fits = [
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr == "fit"
        ]
        self.assertEqual(len(fits), 1)
        self.assertEqual(ast.unparse(fits[0]), "model.fit(X, training.y.to_numpy(dtype=int))")

    def test_no_configuration_override_or_interpretation(self):
        for filename in ["run_final_test.py", "final_test_reporting.py", "final_test_worker.py"]:
            text = (BASE / filename).read_text().lower()
            for phrase in [
                "tabpfn wins",
                "clinically superior",
                "statistically significant",
                "best model",
            ]:
                self.assertNotIn(phrase, text)
        parent_text = (BASE / "run_final_test.py").read_text()
        self.assertIn("len(sys.argv) != 1", parent_text)

    def test_launcher_exact_interpreter_secure_source_and_duplicate_guard(self):
        text = (BASE / "run_final_test.sh").read_text()
        self.assertIn(s.a.PYTHON + " -I -u", text)
        self.assertIn('source "$HOME/.tabpfn.env" >/dev/null 2>&1', text)
        self.assertLess(text.index("locked_execution/test"), text.index("source "))
        self.assertIn("set +x", text)
        self.assertIn("set -a", text)
        self.assertIn("set +a", text)

    def test_installed_preparation_and_all_calibration_evidence_unchanged(self):
        prep, _, lock, registry, mappings = s.checked_state()
        self.assertEqual(len(prep["protected_inventory"]), 273)
        self.assertEqual(registry["models"], list(s.MODELS))
        self.assertEqual(lock["calibration_protocol"], s.a.CALIBRATION_PROTOCOL)
        s.verify_mappings(mappings)

    def test_live_approval_is_required_and_not_installed_by_Codex(self):
        self.assertFalse(s.APPROVAL.exists())
        with self.assertRaises(PermissionError):
            s.checked_state(live=True)

    def test_lock_mapping_source_or_protocol_hash_mismatch_blocks(self):
        with patch.object(s, "MAPPINGS_HASH", "0" * 64):
            with self.assertRaises(PermissionError):
                s.checked_state()
        with patch.object(s, "REGISTRY_HASH", "0" * 64):
            with self.assertRaises(PermissionError):
                s.checked_state()

    def test_completion_or_failure_blocks_every_future_execution(self):
        original = Path.exists
        for forbidden in [s.RECEIPT, s.ROOT / "failure.json"]:
            with patch.object(
                Path, "exists", lambda p, target=forbidden: p == target or original(p)
            ):
                with self.assertRaises(PermissionError):
                    s.checked_state()

    def test_no_real_TEST_directory_reservation_receipt_or_values_loaded(self):
        for path in [s.ROOT, s.RESERVATION, s.RECEIPT, s.APPROVAL]:
            self.assertFalse(path.exists())
        _, _, _, registry, _ = s.checked_state()
        self.assertFalse(registry["TEST_predictors_labels_identifiers_parsed_during_preparation"])


if __name__ == "__main__":
    unittest.main()
