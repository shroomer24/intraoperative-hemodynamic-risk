"""Offline final-path regression tests: synthetic rows and mocked hosted results only."""

import copy
import importlib.util
import json
import os
import sys
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from intraop.evaluation import locked_execution as locked
from intraop.evaluation import result_checkpoint as result
from intraop.evaluation.calibration import apply_platt, bootstrap_intervals, fit_platt

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "tests"))


def module(name):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / (name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


worker = module("locked_model_worker")
runner = module("run_locked_evaluation")


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def seal(path, value):
    write(path, value)
    path.with_suffix(".sha256").write_text(result.file_hash(path) + "\n")


def contract():
    return {
        "kind": "tabpfn",
        "candidate": "tabpfn_full",
        "constructor": result.CONSTRUCTOR.copy(),
        "query_rows": 3,
        "feature_count": 2,
        "features": ["a", "b"],
        "positive_class": 1,
        "probability_column": 1,
        "metadata_excluded_from_X": True,
        "query_order_verified": True,
        "model_lock_sha256": "synthetic_lock",
        "query_source_order_sha256": "synthetic_order",
        "query_X_sha256": "synthetic_X",
        "feature_contract_sha256": "synthetic_features",
        "software_versions": {"client": "0.6.1"},
    }


class FakeModel:
    classes_ = np.array([0, 1])

    def __init__(self, config=None):
        self.config = config or {}
        self._last_meta = {
            "package_version": {"private": "NEVER_PERSIST_THIS"},
            "unknown_server": "NEVER_PERSIST_THIS",
        }

    def get_params(self, deep=False):
        return self.config

    def fit(self, X, y):
        return self

    def predict_proba(self, X):
        p = 0.2 + 0.6 / (1 + np.exp(-X[:, 0])) if X.shape[1] else np.full(len(X), 0.5)
        return np.column_stack([1 - p, p])


def fixture(root):
    modeling = root / "artifacts/modeling-v01"
    cohort = root / "artifacts/vitaldb-cohort-v03"
    modeling.mkdir(parents=True)
    cohort.mkdir(parents=True)
    plan = json.loads((REPO / "artifacts/modeling-v01/development_plan.json").read_text())
    selected = json.loads(
        (REPO / "artifacts/modeling-v01/development/selected_configs.json").read_text()
    )
    parts = ["training", "tuning", "calibration", "test"]
    mapping = {f"s{i}": parts[i // 2] for i in range(8)}
    split = {"subject_to_partition": mapping, "counts": dict.fromkeys(parts, 2)}
    write(cohort / "realized_split_manifest.json", split)
    features = plan["feature_sets"]["full"]
    write(cohort / "feature_schema.json", {"feature_names": features})
    write(modeling / "feature_sets.json", {**plan["feature_sets"]})
    pd.DataFrame(np.random.default_rng(42).normal(size=(32, 74)), columns=features).to_csv(
        cohort / "features.csv", index=False
    )
    pd.DataFrame({"label": np.arange(32) % 2}).to_csv(cohort / "labels.csv", index=False)
    pd.DataFrame(
        {
            "subject_id": [f"s{i // 4}" for i in range(32)],
            "case_id": [f"c{i // 4}" for i in range(32)],
            "window_id": [f"w{i}" for i in range(32)],
            "anchor_time_seconds": np.arange(32) * 60,
            "history_start_seconds": np.arange(32) * 60 - 300,
            "history_end_seconds": np.arange(32) * 60,
            "future_observation_end_seconds": np.arange(32) * 60 + 360,
        }
    ).to_csv(cohort / "metadata.csv", index=False)
    write(
        cohort / "integrity_report.json",
        {
            "split_statistics": {
                "partitions": {p: {"eligible_windows": 8, "positive_windows": 4} for p in parts}
            }
        },
    )
    table_hashes = {
        n: result.file_hash(cohort / n) for n in ["features.csv", "labels.csv", "metadata.csv"]
    }
    write(cohort / "table_manifest.json", {"tables": table_hashes})
    plan["training_context"] = {
        "rows": 8,
        "subjects": 2,
        "positive_labels": 4,
        "partition": "training",
    }
    write(modeling / "development_plan.json", plan)
    write(modeling / "development/selected_configs.json", selected)
    write(
        modeling / "development/development_completion_manifest.json",
        {"test_evaluation_count": 0, "calibration_accessed": False},
    )
    study = root / "artifacts/tabpfn-representation-study-v01"
    closure = {
        "status": "CLOSED",
        "additional_optimization_permitted": False,
        "historical_attempt_files_sha256": {},
    }
    seal(study / "study_closure_manifest.json", closure)
    value = {
        "models": selected,
        "feature_sets": plan["feature_sets"],
        "training_context": plan["training_context"],
        "selection_metric": plan["selection_metric"],
        "tie_break": plan["tie_break"],
        "tabpfn_representation": "A_frozen74_retained_control",
        "calibration_protocol": locked.CALIBRATION_PROTOCOL,
        "test_protocol": locked.TEST_PROTOCOL,
        "test_evaluations_before_lock": 0,
        "TEST_never_evaluated_before_lock": True,
        "cohort_checkpoint": "artifacts/vitaldb-cohort-v03",
        "dataset_manifest_hashes": {
            n: result.file_hash(cohort / n)
            for n in ["feature_schema.json", "realized_split_manifest.json"]
        },
        "table_hashes": table_hashes,
        "split_manifest_hash": result.file_hash(cohort / "realized_split_manifest.json"),
        "evidence_hashes": {
            str(p.relative_to(root)): result.file_hash(p)
            for p in [
                modeling / "development_plan.json",
                modeling / "development/selected_configs.json",
            ]
        },
        "code_hashes": {},
        "package_versions": plan["package_versions"],
        "optimization_study_closure_sha256": result.file_hash(
            study / "study_closure_manifest.json"
        ),
    }
    seal(modeling / "model_lock_manifest.json", value)
    lock_hash = result.file_hash(modeling / "model_lock_manifest.json")
    for phase, protocol in [
        ("calibration", locked.CALIBRATION_PROTOCOL),
        ("test", locked.TEST_PROTOCOL),
    ]:
        seal(modeling / f"{phase}_protocol.json", protocol)
    seal(
        modeling / "final_test_execution_manifest.json",
        {"model_lock_sha256": lock_hash, "protocol": locked.TEST_PROTOCOL},
    )
    return modeling, value


def approve(modeling, phase):
    write(
        modeling / "locked_execution/approvals" / f"{phase}.json",
        {
            "status": "APPROVED",
            "phase": phase,
            "model_lock_sha256": result.file_hash(modeling / "model_lock_manifest.json"),
            "protocol_sha256": result.file_hash(modeling / f"{phase}_protocol.json"),
            "one_execution_only": True,
        },
    )


class ResultTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)
        self.p = np.array([[0.8, 0.2], [0.4, 0.6], [0.7, 0.3]])
        self.c = contract()

    def persist(self, *, p=None, classes=None, observed=None, metadata=None, mark=None):
        return result.persist_result(
            self.root / "checkpoint",
            self.p if p is None else p,
            [0, 1] if classes is None else classes,
            self.c,
            self.c if observed is None else observed,
            {} if metadata is None else metadata,
            mark=mark,
        )

    def test_validation_precedes_checkpoint_and_optional_processing(self):
        stages = []
        self.persist(mark=stages.append)
        self.assertEqual(
            stages,
            [
                "PROBABILITY_VALIDATION_STARTED",
                "PROBABILITY_VALIDATION_PASSED",
                "PROBABILITY_CHECKPOINT_PUBLISHED",
                "OPTIONAL_METADATA_PROCESSED",
            ],
        )

    def test_invalid_shape_never_checkpoints(self):
        with self.assertRaises(result.ScientificFailure):
            self.persist(p=self.p[:2])
        self.assertFalse((self.root / "checkpoint").exists())

    def test_invalid_probabilities_never_checkpoints(self):
        for p in [
            self.p * 2,
            np.full((3, 2), np.nan),
            np.full((3, 2), np.inf),
            np.full((3, 2), -0.1),
        ]:
            with self.assertRaises(result.ScientificFailure):
                self.persist(p=p)
        self.assertFalse((self.root / "checkpoint").exists())

    def test_invalid_classes_never_checkpoints(self):
        for classes in [[1, 0], [0, 2], [[0, 1]]]:
            with self.assertRaises(result.ScientificFailure):
                self.persist(classes=classes)

    def test_feature_or_order_hash_change_never_checkpoints(self):
        for key in ["features", "query_source_order_sha256", "query_X_sha256", "model_lock_sha256"]:
            altered = copy.deepcopy(self.c)
            altered[key] = "changed"
            with self.assertRaises(result.ScientificFailure):
                self.persist(observed=altered)
        self.assertFalse((self.root / "checkpoint").exists())

    def test_optional_rejection_preserves_valid_checkpoint(self):
        scores, meta = self.persist(metadata={"package_version": {"token": "DO_NOT_SAVE"}})
        np.testing.assert_array_equal(scores, self.p[:, 1])
        self.assertEqual(meta["fields"]["package_version"], "UNKNOWN")
        self.assertEqual(
            meta["diagnostics"][0]["category"], "OPTIONAL_METADATA_UNSUPPORTED_REPRESENTATION"
        )
        self.assertNotIn("DO_NOT_SAVE", json.dumps(meta))
        self.assertTrue((self.root / "checkpoint/probabilities.npy").exists())

    def test_unexpected_optional_adapter_bug_preserves_checkpoint_and_stops(self):
        with patch.object(result, "optional_metadata", side_effect=ValueError("private")):
            with self.assertRaises(ValueError):
                self.persist()
        self.assertTrue((self.root / "checkpoint/probabilities.npy").exists())

    def test_required_remote_metadata_failure_fatal_before_checkpoint(self):
        cases = [
            {"classes": [1, 0]},
            {"test_set_num_rows": 4},
            {"test_set_num_cols": 3},
            {"execution_mode": "thinking"},
            {"tabpfn_config": {"model_path": "another"}},
            {"tabpfn_config": {"random_state": 43}},
            {"tabpfn_config": "secret"},
        ]
        for metadata in cases:
            with self.assertRaises(result.ScientificFailure) as caught:
                self.persist(metadata=metadata)
            self.assertEqual(
                caught.exception.diagnostic["category"], "REQUIRED_METADATA_CONTRACT_FAILURE"
            )
        self.assertFalse((self.root / "checkpoint").exists())

    def test_unknown_metadata_never_persisted(self):
        _, meta = self.persist(
            metadata={
                "private_signed_url": "DO_NOT_SAVE",
                "tabpfn_config": {"unknown_server": "DO_NOT_SAVE"},
            }
        )
        self.assertNotIn("DO_NOT_SAVE", json.dumps(meta))
        self.assertNotIn("DO_NOT_SAVE", (self.root / "checkpoint/manifest.json").read_text())

    def test_checkpoint_cannot_be_overwritten(self):
        self.persist()
        h = result.file_hash(self.root / "checkpoint/probabilities.npy")
        with self.assertRaises(FileExistsError):
            self.persist()
        self.assertEqual(h, result.file_hash(self.root / "checkpoint/probabilities.npy"))

    def test_required_metadata_diagnostics_contain_only_safe_type_reason(self):
        with self.assertRaises(result.ScientificFailure) as caught:
            self.persist(metadata={"test_set_num_rows": "DO_NOT_SAVE"})
        d = caught.exception.diagnostic
        self.assertEqual(d["field"], "test_set_num_rows")
        self.assertEqual(d["field_type"], "string")
        self.assertNotIn("DO_NOT_SAVE", json.dumps(d))

    def test_invalid_model_or_semantics_blocks(self):
        for key, value in [
            ("constructor", {"model_path": "v3.5_fast", "random_state": 42}),
            ("positive_class", 0),
            ("probability_column", 0),
            ("metadata_excluded_from_X", False),
            ("query_order_verified", False),
        ]:
            c = copy.deepcopy(self.c)
            c[key] = value
            with self.assertRaises(result.ScientificFailure):
                result.persist_result(self.root / "checkpoint", self.p, [0, 1], c, c, {})


class FirewallTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)
        self.modeling, self.lock = fixture(self.root)

    def test_lock_retains_A_and_exact_configs(self):
        value, _ = locked.validate_lock(self.modeling)
        self.assertEqual(value["tabpfn_representation"], "A_frozen74_retained_control")
        self.assertEqual(
            value["models"]["logistic_map"]["selected_configuration"]["hyperparameters"]["C"], 10
        )
        self.assertEqual(len(value["models"]["tabpfn_full"]["features"]), 74)

    def test_changed_feature_order_invalidates_even_resealed_lock(self):
        self.lock["models"]["tabpfn_full"]["features"].reverse()
        seal(self.modeling / "model_lock_manifest.json", self.lock)
        with self.assertRaises(PermissionError):
            locked.validate_lock(self.modeling)

    def test_changed_hyperparameters_invalidates_even_resealed_lock(self):
        self.lock["models"]["xgboost"]["selected_configuration"]["hyperparameters"]["max_depth"] = 3
        seal(self.modeling / "model_lock_manifest.json", self.lock)
        with self.assertRaises(PermissionError):
            locked.validate_lock(self.modeling)

    def test_bad_lock_hash_blocks(self):
        self.modeling.joinpath("model_lock_manifest.sha256").write_text("wrong")
        with self.assertRaises(PermissionError):
            locked.validate_lock(self.modeling)

    def test_TEST_cannot_load_without_lock(self):
        (self.modeling / "model_lock_manifest.json").unlink()
        with patch.object(locked, "_load_partition") as load:
            with self.assertRaises(PermissionError):
                locked.LockedPartitions(self.modeling, "test", {}, check_environment=False)
            load.assert_not_called()

    def test_approval_required(self):
        with self.assertRaises(PermissionError):
            locked.validate_approval(
                self.modeling,
                "calibration",
                result.file_hash(self.modeling / "model_lock_manifest.json"),
            )

    def test_TEST_cannot_load_before_calibration_complete(self):
        approve(self.modeling, "test")
        reservation = {
            "status": "RESERVED",
            "model_lock_sha256": result.file_hash(self.modeling / "model_lock_manifest.json"),
        }
        root, path, _ = locked.phase_paths(self.modeling, "test")
        write(path, reservation)
        reader = locked.LockedPartitions(
            self.modeling, "test", reservation, check_environment=False
        )
        with patch.object(locked, "_load_partition") as load:
            with self.assertRaises(PermissionError):
                reader.load("test")
            load.assert_not_called()

    def test_wrong_phase_partition_denied_before_values(self):
        reader = locked.LockedPartitions(self.modeling, "calibration", {}, check_environment=False)
        with patch.object(locked, "_load_partition") as load:
            for p in ["test", "tuning"]:
                with self.assertRaises(PermissionError):
                    reader.load(p)
            load.assert_not_called()

    def test_completed_phase_cannot_reload(self):
        approve(self.modeling, "calibration")
        root, _, receipt = locked.phase_paths(self.modeling, "calibration")
        write(receipt, {"status": "COMPLETE"})
        reader = locked.LockedPartitions(self.modeling, "calibration", {}, check_environment=False)
        with patch.object(locked, "_load_partition") as load:
            with self.assertRaises(PermissionError):
                reader.load("calibration")
            load.assert_not_called()

    def test_calibration_cannot_change_protocol_or_models(self):
        for key in ["calibration_protocol", "models"]:
            lock = copy.deepcopy(self.lock)
            lock[key] = {}
            seal(self.modeling / "model_lock_manifest.json", lock)
            with self.assertRaises((PermissionError, KeyError)):
                locked.validate_lock(self.modeling)

    def test_partition_rosters_are_disjoint(self):
        split = json.loads(
            (self.root / "artifacts/vitaldb-cohort-v03/realized_split_manifest.json").read_text()
        )
        rosters = locked.validate_subject_disjointness(split)
        for p in rosters:
            for q in rosters:
                if p != q:
                    self.assertFalse(rosters[p] & rosters[q])
        split["counts"]["test"] = 3
        with self.assertRaises(PermissionError):
            locked.validate_subject_disjointness(split)

    def test_current_MAP_probability_mapping_absent(self):
        definition = self.lock["models"]["current_map"]
        self.assertEqual(definition["preprocessing"]["score"], "-map_latest")
        self.assertNotIn("current_map", locked.CALIBRATED_MODELS)
        self.assertEqual(
            locked.TEST_PROTOCOL["current_map_metrics"], ["average_precision", "auroc"]
        )

    def test_closed_study_blocks_new_attempts(self):
        study = self.root / "artifacts/tabpfn-representation-study-v01"
        before = set(study.rglob("*"))
        for _variant in ["B_attempt_3", "C", "D"]:
            with self.assertRaises(PermissionError):
                locked.ensure_study_open(study)
        self.assertEqual(before, set(study.rglob("*")))

    def test_real_historical_entrypoints_block_closed_study_before_SDK(self):
        from tests.test_tabpfn_B_recovery import recovery
        from tests.test_tabpfn_B_recovery import worker as historical_worker
        from tests.test_tabpfn_representation_study import runner as historical_runner

        with self.assertRaises(PermissionError):
            historical_runner.run()
        for name, attempt in [
            ("B_structural71", 3),
            ("C_causal_contrasts78", 1),
            ("D_positive_preserving_negative_thinning74", 1),
        ]:
            with self.assertRaises(PermissionError):
                historical_worker.execute(name, attempt)
        with self.assertRaises(PermissionError):
            recovery.reserve_B_attempt_2({}, {}, None, None, scanner=lambda: [])


class CalibrationTests(unittest.TestCase):
    def test_single_prespecified_platt_mapping_preserves_raw(self):
        p = np.array([0.1, 0.2, 0.7, 0.8])
        before = p.copy()
        parameters = fit_platt(p, [0, 0, 1, 1])
        cal = apply_platt(p, parameters)
        np.testing.assert_array_equal(p, before)
        self.assertTrue(((cal >= 0) & (cal <= 1)).all())
        self.assertFalse(parameters["configuration_changed"])

    def test_bootstrap_resamples_complete_subjects_with_multiplicity(self):
        groups = np.array(["a", "a", "b", "b", "b", "c"])
        draws = list(locked.bootstrap_subject_indices(groups, replicates=30, seed=42))
        for index in draws:
            for subject in ["a", "b", "c"]:
                original = np.flatnonzero(groups == subject)
                counts = [int((index == position).sum()) for position in original]
                self.assertEqual(len(set(counts)), 1)
        self.assertTrue(any(len(np.unique(index)) < len(index) for index in draws))

    def test_bootstrap_undefined_replicates_are_reported(self):
        intervals = bootstrap_intervals(
            [0, 0, 1, 1], ["a", "a", "b", "b"], {"model": [0.1, 0.2, 0.8, 0.9]}, replicates=30
        )
        result = intervals["model"]["auroc"]
        self.assertGreater(result["undefined_replicates"], 0)
        self.assertEqual(result["valid_replicates"] + result["undefined_replicates"], 30)

    def test_calibration_single_class_stops_without_selection(self):
        with self.assertRaises(ValueError):
            fit_platt([0.1, 0.2], [0, 0])


class FullMockedExecutionTests(unittest.TestCase):
    def test_calibration_then_test_and_no_automatic_repeat(self):
        with tempfile.TemporaryDirectory() as temp, ExitStack() as stack:
            root = Path(temp)
            modeling, _ = fixture(root)
            approve(modeling, "calibration")
            approve(modeling, "test")
            original_validation = locked.validate_lock
            stack.enter_context(
                patch.object(
                    locked,
                    "validate_lock",
                    side_effect=lambda *a, **k: original_validation(
                        *a, **{**k, "check_environment": False}
                    ),
                )
            )
            stack.enter_context(patch.object(worker, "MODELING", modeling))
            stack.enter_context(patch.object(runner, "MODELING", modeling))
            stack.enter_context(
                patch.object(
                    worker, "build_estimator", side_effect=lambda kind, config: FakeModel(config)
                )
            )
            stack.enter_context(
                patch.dict(
                    os.environ,
                    {
                        "INTRAOP_LOCKED_EXTERNAL_EXECUTION": "normal_terminal",
                        "TABPFN_TOKEN": "synthetic",
                    },
                )
            )

            def child(command, **kwargs):
                worker.execute(command[command.index("--model") + 1], command[-1])
                return type("SafeExit", (), {"returncode": 0})()

            stack.enter_context(patch.object(runner.subprocess, "run", side_effect=child))
            stack.enter_context(patch.object(runner, "plot_reliability"))
            # 20 draws exercise integration; production call is pinned to 1000 in code/protocol.
            original_bootstrap = runner.bootstrap_intervals
            stack.enter_context(
                patch.object(
                    runner,
                    "bootstrap_intervals",
                    side_effect=lambda *a, **k: original_bootstrap(*a, **{**k, "replicates": 20}),
                )
            )
            runner.execute("calibration")
            locked.verify_calibration_complete(
                modeling, result.file_hash(modeling / "model_lock_manifest.json")
            )
            runner.execute("test")
            _, _, receipt = locked.phase_paths(modeling, "test")
            self.assertTrue(receipt.exists())
            predictions = pd.read_csv(receipt.parent / "predictions/current_map.csv")
            self.assertTrue(predictions.raw_probability.isna().all())
            self.assertTrue(predictions.calibrated_probability.isna().all())
            with self.assertRaises(PermissionError):
                runner.execute("test")
