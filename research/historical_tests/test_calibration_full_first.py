"""Offline regression coverage for the one first-full raw calibration execution."""

import ast
import contextlib
import copy
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
BASE = REPO / "artifacts/modeling-v01/calibration-full-first-v01"


def module(name, filename):
    spec = importlib.util.spec_from_file_location(name, BASE / filename)
    obj = importlib.util.module_from_spec(spec)
    sys.modules[name] = obj
    spec.loader.exec_module(obj)
    return obj


support = module("full_recovery_support", "full_recovery_support.py")
worker = module("full_identity_worker", "full_worker.py")
parent = module("full_identity_parent", "run_full_first.py")


def metadata(path=support.CANONICAL):
    return {
        "tabpfn_config": {"model_path": path, "random_state": 42},
        "billing_model_version": "v3.5",
        "execution_mode": "standard",
        "classes": [0, 1],
        "test_set_num_rows": 3,
        "test_set_num_cols": 74,
    }


def contract():
    features = json.loads((support.MODELING / "model_lock_manifest.json").read_text())[
        "feature_sets"
    ]["full"]
    return {
        "candidate": "tabpfn_full",
        "kind": "tabpfn",
        "constructor": support.CONSTRUCTOR.copy(),
        "model_lock_sha256": support.LOCK_HASH,
        "training_context_sha256": support.CONTEXT_HASH,
        "model_definition_sha256": support.MODEL_HASH,
        "features": features,
        "feature_count": 74,
        "feature_contract_sha256": support.FEATURE_HASH,
        "query_rows": 3,
        "query_source_order_sha256": support.QUERY_ORDER_HASH,
        "metadata_excluded_from_X": True,
        "query_order_verified": True,
        "positive_class": 1,
        "probability_column": 1,
        "positive_class_semantics": support.SEMANTICS,
    }


class IdentityTests(unittest.TestCase):
    def validate(self, raw):
        return support.compatible_metadata(raw, rows=3, features=74)

    def test_exact_alias_accepted(self):
        self.assertEqual(self.validate(metadata("v3.5_default"))["model_path"], "v3.5_default")

    def test_exact_canonical_accepted_unchanged(self):
        raw = metadata()
        previous = copy.deepcopy(raw)
        self.assertEqual(self.validate(raw)["model_path"], support.CANONICAL)
        self.assertEqual(raw, previous)

    def test_missing_billing_fails_for_both_paths(self):
        for path in ["v3.5_default", support.CANONICAL]:
            raw = metadata(path)
            del raw["billing_model_version"]
            with self.assertRaises(support.ScientificFailure):
                self.validate(raw)

    def test_wrong_billing_fails(self):
        for value in ["v3.5-plus", "3.5", "v3", None, 3.5, {"secret": "sentinel"}]:
            raw = metadata()
            raw["billing_model_version"] = value
            with self.assertRaises(support.ScientificFailure):
                self.validate(raw)

    def test_fast_fails(self):
        for value in ["v3.5-fast", "v3.5-fast_default"]:
            with self.assertRaises(support.ScientificFailure):
                self.validate(metadata(value))

    def test_other_checkpoint_fails(self):
        with self.assertRaises(support.ScientificFailure):
            self.validate(metadata("/app/tabpfn_models/tabpfn-v3.5-20260910.safetensors"))

    def test_v3_fails(self):
        with self.assertRaises(support.ScientificFailure):
            self.validate(metadata("v3"))

    def test_v2_fails(self):
        with self.assertRaises(support.ScientificFailure):
            self.validate(metadata("v2.5_default"))

    def test_custom_basename_normalization_suffix_fail(self):
        for value in [
            "/custom/tabpfn-v3.5-20260909.safetensors",
            "tabpfn-v3.5-20260909.safetensors",
            "/app/tabpfn_models/./tabpfn-v3.5-20260909.safetensors",
            support.CANONICAL + "?secret=sentinel",
        ]:
            with self.assertRaises(support.ScientificFailure):
                self.validate(metadata(value))

    def test_missing_and_unsupported_path_fail(self):
        for value in [None, 35, [], {"secret": "sentinel"}]:
            with self.assertRaises(support.ScientificFailure):
                self.validate(metadata(value))

    def test_only_standard_mode(self):
        for value in ["cache", "fast", None, 1, ["standard"]]:
            raw = metadata()
            raw["execution_mode"] = value
            with self.assertRaises(support.ScientificFailure):
                self.validate(raw)

    def test_required_nonidentity_checks_retained(self):
        for section, key, value in [
            ("root", "test_set_num_rows", 4),
            ("root", "test_set_num_cols", 18),
            ("root", "classes", [1, 0]),
            ("config", "random_state", 43),
            ("config", "balance_probabilities", True),
            ("config", "ignore_pretraining_limits", True),
        ]:
            raw = metadata()
            (raw if section == "root" else raw["tabpfn_config"])[key] = value
            with self.assertRaises(support.ScientificFailure):
                self.validate(raw)

    def test_safe_diagnostic_no_raw_values(self):
        value = "https://credential-sentinel.invalid/patient-sentinel"
        try:
            self.validate(metadata(value))
        except support.ScientificFailure as exc:
            self.assertNotIn(value, str(exc))
            self.assertNotIn("sentinel", json.dumps(exc.diagnostic))
        else:
            self.fail("Must reject unknown identity")

    def test_unknown_metadata_discarded(self):
        raw = metadata()
        raw.update(token="secret-sentinel", url="https://sentinel", request_id="sentinel")
        result = support.optional_metadata(raw)
        self.assertNotIn("sentinel", json.dumps(result))


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "checkpoint"
        self.p = np.array([[0.8, 0.2], [0.3, 0.7], [0.6, 0.4]])
        self.c = contract()
        self.stages = []

    def persist(self, *, p=None, classes=None, observed=None, raw=None):
        return support.persist_approved_result(
            self.path,
            self.p if p is None else p,
            [0, 1] if classes is None else classes,
            self.c,
            self.c if observed is None else observed,
            metadata() if raw is None else raw,
            amendment_hash="mock-amendment",
            mark=self.stages.append,
        )

    def test_publication_order_and_both_identities(self):
        scores, _ = self.persist()
        self.assertTrue(np.array_equal(scores, self.p[:, 1]))
        self.assertEqual(
            self.stages,
            [
                "PROBABILITY_VALIDATION_STARTED",
                "PROBABILITY_VALIDATION_PASSED",
                "PROBABILITY_CHECKPOINT_PUBLISHED",
                "OPTIONAL_METADATA_PROCESSING_STARTED",
                "OPTIONAL_METADATA_PROCESSED",
            ],
        )
        manifest = json.loads((self.path / "manifest.json").read_text())
        identity = manifest["identity_compatibility"]
        self.assertEqual(identity["requested_constructor"], support.CONSTRUCTOR)
        self.assertEqual(identity["actual_returned_model_path"], support.CANONICAL)
        self.assertFalse(identity["checkpoint_identical_development_continuity_claimed"])
        self.assertEqual(self.path.stat().st_mode & 0o222, 0)
        self.assertTrue(all(p.stat().st_mode & 0o222 == 0 for p in self.path.iterdir()))

    def test_optional_exception_cannot_erase_checkpoint(self):
        with patch.object(support, "optional_metadata", side_effect=RuntimeError("sentinel")):
            with self.assertRaises(RuntimeError):
                self.persist()
        self.assertTrue((self.path / "probabilities.npy").is_file())
        self.assertIn("PROBABILITY_VALIDATION_PASSED", self.stages)
        self.assertIn("PROBABILITY_CHECKPOINT_PUBLISHED", self.stages)
        self.assertNotIn("OPTIONAL_METADATA_PROCESSED", self.stages)

    def test_wrong_classes_fail(self):
        for value in [[1, 0], [0], [0, 1, 2], ["0", "1"], [0.0, 1.0], [False, True]]:
            with self.assertRaises(support.ScientificFailure):
                self.persist(classes=value)
        self.assertFalse(self.path.exists())

    def test_invalid_matrix_fails(self):
        for value in [
            np.ones((2, 2)),
            np.ones((3, 1)),
            [[np.nan, 0.2]] * 3,
            [[np.inf, 0.2]] * 3,
            [[1.1, -0.1]] * 3,
            [[0.8, 0.8]] * 3,
        ]:
            with self.assertRaises(support.ScientificFailure):
                self.persist(p=value)
        self.assertFalse(self.path.exists())

    def test_missing_billing_never_publishes(self):
        raw = metadata()
        del raw["billing_model_version"]
        with self.assertRaises(support.ScientificFailure):
            self.persist(raw=raw)
        self.assertFalse(self.path.exists())

    def test_input_identity_constructor_feature_order_and_semantics_fail(self):
        for key, value in [
            ("candidate", "tabpfn_map"),
            ("constructor", {"model_path": "v3", "random_state": 42}),
            ("features", list(reversed(self.c["features"]))),
            ("model_lock_sha256", "different"),
            ("feature_count", 18),
            ("query_order_verified", False),
            ("metadata_excluded_from_X", False),
            ("training_context_sha256", "changed-context"),
            ("model_definition_sha256", "changed-model"),
            ("positive_class", 0),
            ("probability_column", 0),
        ]:
            with patch.dict(self.c, {key: value}):
                with self.assertRaises(support.ScientificFailure):
                    self.persist()
        observed = dict(self.c, query_rows=4)
        with self.assertRaises(support.ScientificFailure):
            self.persist(observed=observed)

    def test_completed_checkpoint_is_immutable(self):
        self.persist()
        before = support.receipt_files(self.path)
        with self.assertRaises(FileExistsError):
            self.persist()
        self.assertEqual(support.receipt_files(self.path), before)


class AttemptTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "first_full"
        self.patch = patch.object(support, "ATTEMPT", self.path)
        self.patch.start()
        self.addCleanup(self.patch.stop)
        self.clear = {"status": "NO_RELEVANT_WORKER_ACTIVE"}
        self.output = io.StringIO()

    def test_only_first_attempt(self):
        self.assertEqual(support.resolve_attempt(), self.path)
        for number in [0, 2, 3, True, "1"]:
            with self.assertRaises(PermissionError):
                support.resolve_attempt(number=number)

    def test_other_model_cannot_reserve(self):
        with self.assertRaises(PermissionError):
            support.resolve_attempt(model="tabpfn_map")
        self.assertFalse(self.path.exists())

    def test_active_worker_blocks_before_reservation(self):
        with patch.object(
            support.subprocess,
            "run",
            return_value=types.SimpleNamespace(
                returncode=0, stdout="919191 python /fixed/full_worker.py --secret=sentinel\n"
            ),
        ):
            with self.assertRaises(PermissionError) as caught:
                support.inspect_processes()
        self.assertNotIn("sentinel", str(caught.exception))
        self.assertFalse(self.path.exists())

    def test_process_inspection_failure_blocks(self):
        with patch.object(
            support.subprocess, "run", return_value=types.SimpleNamespace(returncode=1, stdout="")
        ):
            with self.assertRaises(PermissionError):
                support.inspect_processes()

    def test_duplicate_complete_failed_or_empty_attempt_blocks(self):
        for state in ["empty", "COMPLETE", "FAILED", "INDETERMINATE"]:
            directory = Path(self.directory.name) / state
            directory.mkdir()
            if state != "empty":
                support.atomic_json(directory / "state.json", {"status": state})
            with patch.object(support, "ATTEMPT", directory):
                before = support.receipt_files(directory)
                with self.assertRaises(FileExistsError):
                    support.reserve_attempt("mock", self.clear)
                self.assertEqual(support.receipt_files(directory), before)

    def test_native_crash_retained_indeterminate_no_retry(self):
        with (
            patch.object(support, "checked_state", return_value=({}, "mock", {})),
            patch.object(support, "inspect_processes", return_value=self.clear),
            patch.object(
                parent.subprocess, "run", return_value=types.SimpleNamespace(returncode=-11)
            ) as run,
            contextlib.redirect_stdout(self.output),
        ):
            self.assertEqual(parent.execute(), 2)
            self.assertEqual(run.call_count, 1)
            self.assertEqual(run.call_args.args[0][0], support.PYTHON)
            before = support.receipt_files(self.path)
            with self.assertRaises(FileExistsError):
                parent.execute()
            self.assertEqual(run.call_count, 1)
            self.assertEqual(support.receipt_files(self.path), before)
        stop = support.read_json(self.path / "supervisor_stop.json")
        self.assertEqual(stop["status"], "INDETERMINATE")
        self.assertEqual(stop["signal"], 11)

    def test_child_failure_retained_failed_no_retry(self):
        def child(*args, **kwargs):
            support.atomic_json(self.path / "failure.json", {"status": "FAILED"})
            return types.SimpleNamespace(returncode=2)

        with (
            patch.object(support, "checked_state", return_value=({}, "mock", {})),
            patch.object(support, "inspect_processes", return_value=self.clear),
            patch.object(parent.subprocess, "run", side_effect=child) as run,
            contextlib.redirect_stdout(self.output),
        ):
            self.assertEqual(parent.execute(), 2)
            self.assertEqual(run.call_count, 1)
        self.assertEqual(support.read_json(self.path / "supervisor_stop.json")["status"], "FAILED")

    def test_success_stops_before_platt_and_test(self):
        def child(*args, **kwargs):
            support.seal_json(self.path / "completion_receipt.json", {"mock": "completed"})
            return types.SimpleNamespace(returncode=0)

        with (
            patch.object(support, "checked_state", return_value=({}, "mock", {})),
            patch.object(support, "inspect_processes", return_value=self.clear),
            patch.object(parent.subprocess, "run", side_effect=child) as run,
            patch.object(parent, "validate_completion") as validate,
            contextlib.redirect_stdout(self.output),
        ):
            self.assertEqual(parent.execute(), 0)
            self.assertEqual(run.call_count, 1)
            validate.assert_called_once()
        receipt = support.read_json(self.path / "supervisor_receipt.json")
        self.assertEqual(receipt["status"], "COMPLETE_FULL_ONLY_STOP_FOR_REVIEW")
        self.assertTrue(receipt["tabpfn_full_first_worker_complete"])
        self.assertFalse(receipt["TEST_accessed"])
        self.assertFalse(receipt["platt_started"])

    def test_test_tuning_denied_before_any_loader(self):
        loader = object.__new__(support.RecoveryPartitions)
        with (
            patch.object(loader, "authorize") as authorize,
            patch.object(support, "_load_partition") as read,
        ):
            for partition in ["test", "tuning", "../test", "TEST"]:
                with self.assertRaises(PermissionError):
                    loader.load(partition)
            authorize.assert_not_called()
            read.assert_not_called()

    def test_sdk_transport_error_bypasses_retry_handlers(self):
        client = type(
            "Client", (), {"send": MagicMock(side_effect=RuntimeError("secret-sentinel"))}
        )
        httpx = types.SimpleNamespace(Client=client)
        original = client.send
        with support.one_shot_transport(httpx):
            with self.assertRaises(support.TransportStopped) as caught:
                client().send()
        self.assertNotIn("sentinel", str(caught.exception))
        self.assertFalse(issubclass(support.TransportStopped, Exception))
        self.assertEqual(original.call_count, 1)
        self.assertIs(client.send, original)

    def test_safe_http_status_only_retained(self):
        client = type(
            "Client",
            (),
            {
                "send": MagicMock(
                    return_value=types.SimpleNamespace(
                        status_code=401, url="https://secret-sentinel", request_id="secret-sentinel"
                    )
                )
            },
        )
        with support.one_shot_transport(types.SimpleNamespace(Client=client)):
            with self.assertRaises(support.TransportStopped):
                client().send()
        contents = "".join(p.read_text() for p in self.path.rglob("*.json"))
        self.assertIn("401", contents)
        self.assertNotIn("sentinel", contents)


class WorkerTests(unittest.TestCase):
    def mocked_execution(self, *, optional_error=False, bad_identity=False):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "first_full"
            output.mkdir()
            loader = MagicMock()
            loader.amendment_hash = "mock-amendment"
            loader.access_log = ["training", "calibration"]
            training = types.SimpleNamespace(
                y=types.SimpleNamespace(to_numpy=lambda **kwargs: np.array([0, 1, 0], dtype=int))
            )
            query = object()
            loader.load.side_effect = [training, query]
            model = MagicMock()
            model.get_params.return_value = support.CONSTRUCTOR.copy()
            model.classes_ = np.array([0, 1])
            model.predict_proba.return_value = np.array([[0.8, 0.2], [0.3, 0.7], [0.6, 0.4]])
            model._last_meta = metadata("v3" if bad_identity else support.CANONICAL)
            ctor = MagicMock(return_value=model)
            sdk = types.SimpleNamespace(TabPFNClassifier=ctor)
            c = contract()
            real_optional = support.optional_metadata
            saved = sys.modules.pop("xgboost", None)
            try:
                with (
                    patch.object(support, "ATTEMPT", output),
                    patch.object(support, "RecoveryPartitions", return_value=loader),
                    patch.object(
                        support,
                        "input_contract",
                        return_value=(np.zeros((3, 74)), np.zeros((3, 74)), c),
                    ),
                    patch.object(support, "quiet_sdk", return_value=contextlib.nullcontext()),
                    patch.object(
                        support, "one_shot_transport", return_value=contextlib.nullcontext()
                    ),
                    patch.dict(sys.modules, {"tabpfn_client": sdk}),
                    patch.object(
                        support,
                        "optional_metadata",
                        side_effect=RuntimeError("secret-sentinel")
                        if optional_error
                        else real_optional,
                    ),
                ):
                    result = worker.execute()
            finally:
                if saved is not None:
                    sys.modules["xgboost"] = saved
            ctor.assert_called_once_with(model_path="v3.5_default", random_state=42)
            model.fit.assert_called_once()
            model.predict_proba.assert_called_once()
            self.assertEqual(
                [c.args[0] for c in loader.load.call_args_list], ["training", "calibration"]
            )
            if optional_error:
                self.assertEqual(result, 2)
                self.assertTrue((output / "probability_checkpoint/probabilities.npy").exists())
                self.assertFalse((output / "completion_receipt.json").exists())
                failure = support.read_json(output / "failure.json")
                self.assertEqual(
                    failure["last_completed_stage"], "OPTIONAL_METADATA_PROCESSING_STARTED"
                )
                self.assertNotIn("sentinel", json.dumps(failure))
            elif bad_identity:
                self.assertEqual(result, 2)
                self.assertFalse((output / "probability_checkpoint").exists())
                self.assertEqual(
                    support.read_json(output / "failure.json")["diagnostic"]["field"], "model_path"
                )
            else:
                self.assertEqual(result, 0)
                receipt, _ = support.verify_seal(output / "completion_receipt.json")
                self.assertEqual(receipt["review_boundary"], "STOP_BEFORE_PLATT_OR_TEST")
                self.assertEqual(receipt["partition_access"], ["training", "calibration"])
                self.assertEqual(receipt["hosted_fit_calls"], 1)
                self.assertEqual(receipt["predict_proba_calls"], 1)

    def test_mocked_worker_success_one_call_each(self):
        self.mocked_execution()

    def test_mocked_worker_optional_bug_preserves_checkpoint(self):
        self.mocked_execution(optional_error=True)

    def test_mocked_worker_identity_failure_stops(self):
        self.mocked_execution(bad_identity=True)


class FrozenFullContractTests(unittest.TestCase):
    def fixtures(self):
        lock = json.loads((support.MODELING / "model_lock_manifest.json").read_text())
        names = lock["feature_sets"]["full"]
        n = lock["training_context"]["rows"]
        train_index = pd.RangeIndex(n)
        query_index = pd.RangeIndex(2393)
        y = np.zeros(n, dtype=int)
        y[: lock["training_context"]["positive_labels"]] = 1
        training = types.SimpleNamespace(
            X=pd.DataFrame(np.zeros((n, 74)), index=train_index, columns=names),
            y=pd.Series(y, index=train_index),
            groups=pd.Series(["synthetic_train_" + str(k % 90) for k in range(n)]),
            metadata=pd.DataFrame({"subject_id": ["synthetic"] * n}, index=train_index),
        )
        query = types.SimpleNamespace(
            X=pd.DataFrame(np.zeros((2393, 74)), index=query_index, columns=names),
            y=pd.Series(np.zeros(2393, dtype=int), index=query_index),
            groups=pd.Series(["synthetic_cal_" + str(k % 23) for k in range(2393)]),
            metadata=pd.DataFrame({"subject_id": ["synthetic"] * 2393}, index=query_index),
        )
        loader = types.SimpleNamespace(lock=lock)
        return loader, training, query

    def test_synthetic_frozen_contract_has_exact_74_ordered_columns(self):
        loader, training, query = self.fixtures()
        with patch.object(support, "row_hash", return_value=support.QUERY_ORDER_HASH):
            X, q, result = support.input_contract(loader, training, query)
        self.assertEqual(X.shape, (13975, 74))
        self.assertEqual(q.shape, (2393, 74))
        self.assertEqual(result["features"], loader.lock["feature_sets"]["full"])
        self.assertEqual(result["feature_contract_sha256"], support.FEATURE_HASH)
        self.assertEqual(result["training_context_sha256"], support.CONTEXT_HASH)
        self.assertEqual(result["model_definition_sha256"], support.MODEL_HASH)
        self.assertTrue(result["metadata_excluded_from_X"])

    def test_changed_baseline_feature_order_context_or_model_rejected(self):
        for kind in ["order", "subset", "seed", "context", "representation"]:
            loader, training, query = self.fixtures()
            if kind == "order":
                loader.lock["models"]["tabpfn_full"]["features"].reverse()
            elif kind == "subset":
                loader.lock["models"]["tabpfn_full"]["features"] = loader.lock["feature_sets"][
                    "map_only"
                ]
            elif kind == "seed":
                loader.lock["models"]["tabpfn_full"]["selected_configuration"]["hyperparameters"][
                    "random_state"
                ] = 43
            elif kind == "context":
                loader.lock["training_context"]["enrichment"] = True
            else:
                loader.lock["models"]["tabpfn_full"]["preprocessing"] = "changed"
            with patch.object(support, "row_hash", return_value=support.QUERY_ORDER_HASH):
                with self.assertRaises(support.ScientificFailure):
                    support.input_contract(loader, training, query)

    def test_query_population_or_order_hash_mismatch_rejected(self):
        loader, training, query = self.fixtures()
        with patch.object(support, "row_hash", return_value="wrong-query-order"):
            with self.assertRaises(support.ScientificFailure):
                support.input_contract(loader, training, query)
        query.X = query.X.iloc[:-1]
        with patch.object(support, "row_hash", return_value=support.QUERY_ORDER_HASH):
            with self.assertRaises(support.ScientificFailure):
                support.input_contract(loader, training, query)

    def test_same_immutable_identity_validator_as_completed_map(self):
        self.assertIs(support.compatible_metadata, support.map_support.compatible_metadata)
        for path in ["v3.5_default", support.CANONICAL, "v3.5-fast", "v3", None]:
            for billing in ["v3.5", "v3", None]:
                raw = metadata(path)
                raw["billing_model_version"] = billing
                for function in [
                    support.compatible_metadata,
                    support.map_support.compatible_metadata,
                ]:
                    if path in ["v3.5_default", support.CANONICAL] and billing == "v3.5":
                        self.assertEqual(function(raw, rows=3, features=74)["model_path"], path)
                    else:
                        with self.assertRaises(support.ScientificFailure):
                            function(raw, rows=3, features=74)


class ScopeTests(unittest.TestCase):
    def test_literal_one_fit_and_one_predict_only(self):
        tree = ast.parse((BASE / "full_worker.py").read_text())
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
        methods = [n.func.attr for n in calls if isinstance(n.func, ast.Attribute)]
        self.assertEqual(methods.count("fit"), 1)
        self.assertEqual(methods.count("predict_proba"), 1)
        ctor = [
            n for n in calls if isinstance(n.func, ast.Name) and n.func.id == "TabPFNClassifier"
        ]
        self.assertEqual(len(ctor), 1)
        self.assertEqual(
            {k.arg: ast.literal_eval(k.value) for k in ctor[0].keywords}, support.CONSTRUCTOR
        )
        loads = [
            ast.literal_eval(n.args[0])
            for n in calls
            if isinstance(n.func, ast.Attribute) and n.func.attr == "load"
        ]
        self.assertEqual(loads, ["training", "calibration"])
        text = (BASE / "run_full_first.py").read_text()
        self.assertNotIn("build_estimator", text)
        self.assertNotIn("joblib", text)
        self.assertNotIn("platt.fit", text)

    def test_launcher_exact_interpreter_no_args_no_tracing(self):
        text = (BASE / "run_full_first.sh").read_text()
        self.assertIn(support.PYTHON + " -I -u", text)
        self.assertIn('source "$HOME/.tabpfn.env" >/dev/null 2>&1', text)
        self.assertLess(text.index("set +x"), text.index('source "$HOME/.tabpfn.env"'))
        self.assertIn('if [ "$#" -ne 0 ]', text)
        self.assertNotIn("run_locked_evaluation", text)


class InstalledStateTests(unittest.TestCase):
    def test_installed_seals_environment_and_all_protected_hashes(self):
        with patch.object(
            support, "_load_partition", side_effect=AssertionError("No table parsing")
        ):
            amendment, _, lock = support.checked_state()
        self.assertEqual(len(amendment["protected_inventory"]), 145)
        self.assertEqual(lock["package_versions"]["tabpfn-client"], "0.6.1")
        self.assertEqual(lock["python_executable"], support.PYTHON)
        self.assertEqual(lock["python_version"], "3.14.6")

    def test_lock_protocol_original_validator_and_failure_unchanged(self):
        amendment, _, _ = support.checked_state()
        for path in [
            support.MODELING / "model_lock_manifest.json",
            support.MODELING / "calibration_protocol.json",
            REPO / "src/evaluation/result_checkpoint.py",
            support.CALIBRATION / "workers/tabpfn_map/failure.json",
            support.CALIBRATION / "failure.json",
        ]:
            self.assertEqual(
                support.file_hash(path),
                amendment["protected_inventory"][str(path.relative_to(REPO))]["sha256"],
            )

    def test_six_workers_hash_verified_not_deserialized(self):
        amendment, _, _ = support.checked_state()
        for name in support.PRESERVED:
            paths = [
                p
                for p in amendment["protected_inventory"]
                if (
                    f"/workers/{name}/" in p
                    or (name == "tabpfn_map" and "/attempts/tabpfn_map/attempt_2/" in p)
                )
            ]
            self.assertGreater(len(paths), 3)
            for relative in paths:
                self.assertEqual(
                    support.file_hash(REPO / relative),
                    amendment["protected_inventory"][relative]["sha256"],
                )

    def test_full_test_unstarted_map_complete(self):
        self.assertFalse(support.ATTEMPT.exists())
        self.assertEqual(support.verify_map_completion()["status"], "COMPLETE")
        self.assertEqual(
            support.file_hash(support.MAP_ATTEMPT / "probability_checkpoint/probabilities.npy"),
            support.MAP_PROBABILITY_HASH,
        )
        self.assertFalse((support.MODELING / "locked_execution/test").exists())
        self.assertFalse((support.MODELING / "locked_execution/approvals/test.json").exists())

    def test_pinned_python_and_client_mismatch_block(self):
        with patch.object(support.sys, "executable", "/other/python"):
            with self.assertRaises(PermissionError):
                support.checked_state()
        from intraop.evaluation import locked_execution

        original = locked_execution.importlib.metadata.version

        def version(name):
            return "0.3.3" if name == "tabpfn-client" else original(name)

        with patch.object(locked_execution.importlib.metadata, "version", side_effect=version):
            with self.assertRaises(PermissionError):
                support.checked_state()

    def test_new_unreviewed_failure_still_blocks(self):
        unknown = support.CALIBRATION / "unreviewed_failure.json"
        original = Path.is_file

        def is_file(path):
            return True if path == unknown else original(path)

        with (
            patch.object(Path, "rglob", return_value=[unknown]),
            patch.object(Path, "is_file", is_file),
        ):
            with self.assertRaises(PermissionError):
                support.checked_state()

    def test_second_full_platt_and_test_presence_still_blocks(self):
        original = Path.exists
        for forbidden in [
            support.CALIBRATION / "attempts/tabpfn_full",
            support.CALIBRATION / "calibrators",
            support.CALIBRATION / "completion_receipt.json",
            support.MODELING / "locked_execution/test",
            support.MODELING / "locked_execution/approvals/test.json",
        ]:
            with patch.object(
                Path, "exists", lambda p, target=forbidden: p == target or original(p)
            ):
                with self.assertRaises(PermissionError):
                    support.checked_state()

    def test_original_production_canonical_rejection_unchanged(self):
        from intraop.evaluation.result_checkpoint import required_metadata

        with self.assertRaises(support.ScientificFailure):
            required_metadata(metadata(), rows=3, features=74)


if __name__ == "__main__":
    unittest.main()
