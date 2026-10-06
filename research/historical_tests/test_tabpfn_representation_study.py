"""Approved bounded study safeguards, synthetic inputs and mocked SDK only."""

import importlib.util
import io
import json
import sys
import tempfile
import unittest
import zipfile
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from intraop.data.datasets import ClassificationDataset
from intraop.data.modeling import DevelopmentPartitions
from intraop.evaluation.development import export_predictions

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "artifacts/tabpfn-representation-study-v01"
sys.path.insert(0, str(STUDY))


def module(name):
    spec = importlib.util.spec_from_file_location(name, STUDY / f"{name}.py")
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


support = module("study_support")
worker = module("study_worker")
runner = module("run_study")
PLAN = json.loads((STUDY / "study_plan.json").read_text())


ORIGINAL_FILE_HASH = support.sha256


def historical_source_hash(path):
    # Historical guards test their archived source against the unchanged original plan.
    # Modern held-out authorization is tested separately by test_locked_final_execution.
    closure = STUDY / "study_closure_manifest.json"
    if closure.exists():
        manifest = json.loads(closure.read_text())
        relative = str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else None
        if relative in manifest["historical_modified_sources_sha256"]:
            archived = STUDY / manifest["historical_source_snapshot"] / relative
            return ORIGINAL_FILE_HASH(archived)
    return ORIGINAL_FILE_HASH(path)


def fixture(rows=12, subjects=2, prefix="synthetic", positives=None):
    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.normal(size=(rows, 74)), columns=PLAN["variants"][0]["feature_names"])
    X.index.name = "source_row"
    sizes = np.array_split(np.arange(rows), subjects)
    m = pd.DataFrame(index=X.index)
    m["subject_id"] = [f"{prefix}_{i}" for i, indices in enumerate(sizes) for _ in indices]
    m["case_id"] = m.subject_id
    m["window_id"] = [f"{prefix}_window_{i}" for i in range(rows)]
    m["anchor_time_seconds"] = np.concatenate([np.arange(len(v)) * 60 for v in sizes])
    m["history_start_seconds"] = m.anchor_time_seconds - 300
    m["history_end_seconds"] = m.anchor_time_seconds
    m["future_observation_end_seconds"] = m.anchor_time_seconds + 600
    m["matched_episode_onset_seconds"] = m.anchor_time_seconds + 120
    y = pd.Series(np.arange(rows) % 2, index=X.index, dtype="int64")
    if positives is not None:
        y[:] = 0
        for positions in positives:
            y.iloc[positions] = 1
    return ClassificationDataset(X, y, m)


class StudyTests(unittest.TestCase):
    def test_spec_and_baseline_hash_are_pinned(self):
        if (support.ORIGINAL / "model_lock_manifest.json").exists():
            with self.assertRaises(PermissionError):
                support.verified_spec()
            return
        plan = support.verified_spec()
        self.assertEqual(plan["variant_count"], 4)
        self.assertEqual(support.sha256(support.AUDIT_PLAN), support.SPEC_HASH)
        with patch.object(support, "sha256", return_value="changed"):
            with self.assertRaises(ValueError):
                support.verified_spec()

    def test_baseline_hash_mismatch_stops(self):
        original = support.sha256
        with patch.object(
            support,
            "sha256",
            side_effect=lambda p: "changed" if p == support.BASELINE else original(p),
        ):
            with self.assertRaisesRegex(ValueError, "baseline"):
                support.verified_spec()

    def test_b_is_exact_ordered_71(self):
        data = fixture()
        actual = support.represented_features(data, PLAN, support.IDS[1])
        names = PLAN["variants"][1]["feature_names"]
        self.assertEqual(actual.shape, (12, 71))
        np.testing.assert_array_equal(actual, data.X[names].to_numpy())
        self.assertEqual(len(set(data.X) - set(names)), 3)

    def test_c_arithmetic_order_nan_and_dtype(self):
        data = fixture()
        data.X.loc[0, "map_slope_60s"] = np.nan
        data.X.loc[1, "sbp_latest"] = np.nan
        before = data.X.copy(deep=True)
        actual = support.represented_features(data, PLAN, support.IDS[2])
        self.assertEqual(actual.shape, (12, 78))
        self.assertEqual(actual.dtype, np.float64)
        for i, (_, left, right) in enumerate(support.DERIVED, 74):
            np.testing.assert_array_equal(actual[:, i], (data.X[left] - data.X[right]).to_numpy())
        self.assertTrue(np.isnan(actual[0, 75]))
        self.assertTrue(np.isnan(actual[1, 76]))
        pd.testing.assert_frame_equal(data.X, before)

    def test_d_exact_clock_positive_not_reset(self):
        data = fixture(rows=8, subjects=1)
        data.metadata["anchor_time_seconds"] = [0, 120, 180, 240, 300, 420, 600, 900]
        data.y[:] = 0
        data.y.iloc[2] = 1
        positions = support.negative_thinning_positions(data)
        self.assertEqual(positions.tolist(), [0, 2, 4, 6, 7])
        report = support.context_manifest(data, positions)
        self.assertTrue(report["all_original_positive_rows_retained"])
        self.assertFalse(report["positive_resets_negative_clock"])

    def test_d_named_source_index_is_unambiguous_and_positions_unchanged(self):
        data = fixture(rows=8, subjects=1)
        named = support.negative_thinning_positions(data)
        self.assertEqual(data.X.index.name, "source_row")
        data.X.index.name = None
        data.metadata.index.name = None
        plain = support.negative_thinning_positions(data)
        np.testing.assert_array_equal(named, plain)

    def test_d_case_reset_ties_and_original_order(self):
        data = fixture(rows=8, subjects=2)
        data.metadata["anchor_time_seconds"] = [300, 0, 300, 600, 0, 300, 300, 600]
        data.y[:] = 0
        positions = support.negative_thinning_positions(data)
        self.assertEqual(positions.tolist(), [0, 1, 3, 4, 5, 7])
        self.assertTrue(np.all(np.diff(positions) > 0))

    def test_d_retains_498_positives_and_all_positive_states(self):
        sizes = np.array_split(np.arange(13975), 90)
        positives = [v[:12] for v in sizes[:41]] + [sizes[41][:6]]
        data = fixture(13975, 90, positives=positives)
        positions = support.negative_thinning_positions(data)
        report = support.context_manifest(data, positions)
        self.assertEqual(report["retained_positives"], 498)
        self.assertEqual(report["represented_positive_subjects"], 42)
        self.assertEqual(report["retained_subjects"], 90)
        self.assertTrue(np.isin(np.flatnonzero(data.y == 1), positions).all())
        self.assertEqual(report["selected_position_sha256"], support.positions_hash(positions))

    def test_all_variants_query_full_population(self):
        train, tune = fixture(), fixture(rows=7, prefix="query")
        for variant in support.IDS[1:]:
            _, y, query, _ = support.matrices(train, tune, PLAN, variant)
            self.assertEqual(y.dtype, np.int64)
            self.assertEqual(len(query), 7)
            np.testing.assert_array_equal(query, support.represented_features(tune, PLAN, variant))

    def test_A_extra_variants_reordered_features_metadata_and_inf_rejected(self):
        data = fixture()
        for bad in [support.IDS[0], "E", "B+C"]:
            with self.assertRaises(PermissionError):
                support.represented_features(data, PLAN, bad)
        original = data.X.copy()
        object.__setattr__(data, "X", data.X.iloc[:, ::-1])
        with self.assertRaises(ValueError):
            support.represented_features(data, PLAN, support.IDS[1])
        object.__setattr__(data, "X", original)
        data.metadata["map_latest"] = 0
        with self.assertRaisesRegex(ValueError, "Metadata"):
            support.represented_features(data, PLAN, support.IDS[1])
        object.__setattr__(data, "metadata", data.metadata.drop(columns="map_latest"))
        data.X.iloc[0, 0] = np.inf
        with self.assertRaises(ValueError):
            support.represented_features(data, PLAN, support.IDS[1])

    def test_frozen_source_feature_manifest_and_table_hashes_block_changes(self):
        original = historical_source_hash
        source = next(
            iter(support.read_json(support.ORIGINAL / "development_plan.json")["code_hashes"])
        )
        for target in [support.REPO / source, support.ORIGINAL / "feature_sets.json"]:
            with patch.object(
                support,
                "sha256",
                side_effect=lambda p, target=target: "changed" if p == target else original(p),
            ):
                with patch.object(support, "DevelopmentPartitions") as reader:
                    with self.assertRaises(ValueError):
                        support.load_development(PLAN)
                    reader.assert_not_called()
        import intraop.data.modeling as modeling

        table = support.REPO / PLAN["frozen_checkpoint"] / "features.csv"
        with (
            patch.object(support, "sha256", side_effect=historical_source_hash),
            patch.object(
                modeling, "sha256", side_effect=lambda p: "changed" if p == table else original(p)
            ),
        ):
            with patch.object(modeling, "_load_partition") as reader:
                with self.assertRaisesRegex(ValueError, "model table hash"):
                    support.load_development(PLAN)
                reader.assert_not_called()

    def test_training_tuning_membership_and_feature_order_guard(self):
        sizes = np.array_split(np.arange(13975), 90)
        training = fixture(
            13975, 90, prefix="train", positives=[v[:12] for v in sizes[:41]] + [sizes[41][:6]]
        )
        qsizes = np.array_split(np.arange(3193), 23)
        tuning = fixture(
            3193, 23, prefix="query", positives=[v[:10] for v in qsizes[:9]] + [qsizes[0][10:14]]
        )
        loader = SimpleNamespace(
            load=lambda partition: {"training": training, "tuning": tuning}[partition]
        )
        with (
            patch.object(support, "sha256", side_effect=historical_source_hash),
            patch.object(support, "DevelopmentPartitions", return_value=loader),
        ):
            support.load_development(PLAN)
            tuning.metadata.loc[tuning.groups == "query_0", "subject_id"] = "train_0"
            with self.assertRaisesRegex(ValueError, "subjects overlap"):
                support.load_development(PLAN)
            tuning.metadata.loc[tuning.groups == "train_0", "subject_id"] = "query_0"
            object.__setattr__(tuning, "X", tuning.X.iloc[:, ::-1])
            with self.assertRaisesRegex(ValueError, "74-feature"):
                support.load_development(PLAN)

    def test_duplicate_worker_start_stops_before_data_or_SDK(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            variant = support.IDS[1]
            expected = {"unchanged": True}
            candidate = root / "execution/candidates" / variant
            support.atomic_json(
                candidate / "attempt_1/reservation.json",
                {"scientific_identity_sha256": support.digest_json(expected)},
                exclusive=True,
            )
            marker = candidate / "attempt_1/worker_started.json"
            support.atomic_json(marker, {"status": "STARTED"}, exclusive=True)
            retained = marker.read_bytes()
            with (
                patch.object(worker, "STUDY", root),
                patch.object(worker, "verified_spec", return_value=PLAN),
                patch.object(worker, "environment"),
                patch.object(
                    worker, "verify_preparation", return_value={"candidates": {variant: expected}}
                ),
                patch.object(worker, "load_development") as reader,
            ):
                with self.assertRaises(FileExistsError):
                    worker.execute(variant)
                reader.assert_not_called()
            self.assertEqual(marker.read_bytes(), retained)

    def test_partition_firewall_before_reader(self):
        loader = DevelopmentPartitions.__new__(DevelopmentPartitions)
        with patch("intraop.data.modeling._load_partition") as value_reader:
            for partition in ["calibration", "test"]:
                with self.assertRaises(PermissionError):
                    loader.load(partition)
            value_reader.assert_not_called()

    def test_probability_orientation_shape_and_validation(self):
        p = np.array([[0.9, 0.1], [0.2, 0.8]])
        np.testing.assert_array_equal(support.class_one_probability(p, [0, 1], 2), [0.1, 0.8])
        for probabilities, classes, n in [
            (p, [1, 0], 2),
            (p, [0, 1], 3),
            (np.array([[0.2, 0.2]]), [0, 1], 1),
            (np.array([[np.nan, 1]]), [0, 1], 1),
        ]:
            with self.assertRaises(ValueError):
                support.class_one_probability(probabilities, classes, n)

    def test_prediction_alignment_and_membership(self):
        data = fixture()
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "pred.csv"
            frame = export_predictions(
                data,
                np.full(12, 0.1),
                model="tabpfn_representation_study",
                candidate_id=support.IDS[1],
                probability_output=True,
            )
            frame.to_csv(path, index=False)
            support.validate_predictions(path, data, variant=support.IDS[1])
            for bad_column in ["subject_id", "case_id", "window_id", "source_row", "true_label"]:
                bad = frame.copy()
                bad.loc[0, bad_column] = (
                    bad.loc[1, bad_column] if bad_column != "subject_id" else "outside"
                )
                if bad_column == "case_id":
                    bad.loc[0, bad_column] = "outside"
                bad.to_csv(path, index=False)
                with self.assertRaises(ValueError):
                    support.validate_predictions(path, data, variant=support.IDS[1])
            frame.iloc[::-1].to_csv(path, index=False)
            with self.assertRaises(ValueError):
                support.validate_predictions(path, data)

    def test_metadata_allowlist_discards_payloads_secrets_and_ids(self):
        raw = {
            "classes": [0, 1],
            "test_set_num_rows": 7,
            "test_set_num_cols": 71,
            "package_version": "9.0.0",
            "n_estimators": 8,
            "billing_model_version": "v3.5",
            "execution_mode": "standard",
            "cache_outcome": "miss",
            "authorization": "secret-A",
            "signed_url": "https://example/?token=secret-B",
            "fitted_train_set_id": "secret-C",
            "subject_id": "private-ID",
            "X": [1, 2, 3],
            "tabpfn_config": {
                "model_path": "v3.5_default",
                "random_state": 42,
                "authorization": "secret-D",
                "patient_payload": [3, 4],
                "inference_config": {"N_ESTIMATORS": 8, "unknown": "secret-E"},
            },
        }
        clean = support.safe_metadata(raw, query_rows=7, feature_count=71)
        rendered = json.dumps(clean)
        for forbidden in [
            "secret-",
            "private-ID",
            "https://",
            "fitted_train_set_id",
            '"patient_payload":',
        ]:
            self.assertNotIn(forbidden, rendered)
        for field, value in [
            ("classes", [1, 0]),
            ("test_set_num_rows", 8),
            ("billing_model_version", "v3.5-fast"),
            ("execution_mode", "thinking"),
        ]:
            with self.assertRaises(ValueError):
                support.safe_metadata({**raw, field: value}, query_rows=7, feature_count=71)
        with self.assertRaises(ValueError):
            support.safe_config({"model_path": "secret-token"})

    def test_metadata_drift_stops_but_cache_and_dimensions_can_differ(self):
        a = support.safe_metadata(
            {"package_version": "9.0.0", "cache_outcome": "hit", "test_set_num_cols": 71},
            query_rows=7,
            feature_count=71,
        )
        b = support.safe_metadata(
            {"package_version": "9.0.0", "cache_outcome": "miss", "test_set_num_cols": 78},
            query_rows=7,
            feature_count=78,
        )
        support.require_consistent_metadata([a, b])
        b["metadata"]["package_version"] = "9.0.1"
        with self.assertRaises(RuntimeError):
            support.require_consistent_metadata([a, b])

    def test_attempt_states_never_implicitly_retry(self):
        with tempfile.TemporaryDirectory() as d:
            candidate = Path(d) / "candidate"
            self.assertEqual(support.candidate_state(candidate), "NEW")
            with self.assertRaises(RuntimeError):
                support.reserve_attempt(candidate, "identity", scanner=lambda: [123])
            self.assertFalse(candidate.exists())
            support.reserve_attempt(candidate, "identity", scanner=lambda: [])
            self.assertEqual(support.candidate_state(candidate), "INDETERMINATE")
            before = (candidate / "attempt_1/reservation.json").read_bytes()
            with self.assertRaises(RuntimeError):
                support.reserve_attempt(candidate, "identity", scanner=lambda: [])
            self.assertEqual(before, (candidate / "attempt_1/reservation.json").read_bytes())
            logs = candidate / "attempt_1/logs"
            logs.mkdir()
            support.atomic_json(logs / "exit.json", {"returncode": 0}, exclusive=True)
            self.assertEqual(support.candidate_state(candidate), "INDETERMINATE")
            (candidate / "bundle").mkdir()
            self.assertEqual(support.candidate_state(candidate), "COMPLETE")
            with self.assertRaises(RuntimeError):
                support.reserve_attempt(candidate, "changed", scanner=lambda: [])
            support.atomic_json(logs / "failure_marker.json", {"status": "FAILED"}, exclusive=True)
            self.assertEqual(support.candidate_state(candidate), "FAILED")

    def test_empty_directory_is_indeterminate_and_preserved(self):
        with tempfile.TemporaryDirectory() as d:
            candidate = Path(d) / "candidate"
            candidate.mkdir()
            self.assertEqual(support.candidate_state(candidate), "INDETERMINATE")
            with self.assertRaises(RuntimeError):
                support.reserve_attempt(candidate, "x", scanner=lambda: [])
            self.assertEqual(list(candidate.iterdir()), [])

    def test_environment_no_fallback_and_no_external_launch_without_boundary(self):
        with patch.object(support.sys, "executable", "other-python"):
            with self.assertRaises(RuntimeError):
                support.environment(PLAN)
        with patch.object(support.os, "environ", {}):
            with self.assertRaises(PermissionError):
                support.environment(PLAN, external=True)
        with patch.object(
            support.os, "environ", {"TABPFN_STUDY_EXTERNAL_EXECUTION": "normal_terminal"}
        ):
            with self.assertRaises(RuntimeError):
                support.environment(PLAN, external=True)

    def test_transport_failure_bypasses_retry_and_restores_method(self):
        attempts = []

        class Client:
            def send(self, *args, **kwargs):
                attempts.append(1)
                raise RuntimeError("secret and patient payload must never escape")

        httpx = SimpleNamespace(Client=Client)
        original = Client.send
        with worker.no_transport_retries(httpx):
            with self.assertRaises(worker.OneShotTransportFailure):
                Client().send()
        self.assertEqual(len(attempts), 1)
        self.assertIs(Client.send, original)
        Client.send = lambda *_a, **_kw: SimpleNamespace(status_code=503)
        with worker.no_transport_retries(httpx):
            with self.assertRaises(worker.OneShotTransportFailure):
                Client().send()
        Client.send = lambda *_a, **_kw: SimpleNamespace(status_code=409)
        with worker.no_transport_retries(httpx):
            self.assertEqual(Client().send().status_code, 409)

    def test_frozen_selection_AP_strict_improvement_tolerance_ties(self):
        def records(aps, losses):
            return [
                dict(variant=v, average_precision=ap, log_loss=loss)
                for v, ap, loss in zip(support.IDS, aps, losses, strict=True)
            ]

        self.assertEqual(
            support.selected_representation(records([0.2, 0.2, 0.2, 0.2], [0.5, 0.1, 0.2, 0.3]))[
                "variant"
            ],
            support.IDS[0],
        )
        self.assertEqual(
            support.selected_representation(
                records([0.2, 0.2 + 5e-13, 0.1, 0.1], [0.5, 0.1, 0.2, 0.3])
            )["variant"],
            support.IDS[0],
        )
        self.assertEqual(
            support.selected_representation(records([0.2, 0.3, 0.3, 0.3], [0.5, 0.2, 0.1, 0.1]))[
                "variant"
            ],
            support.IDS[2],
        )
        self.assertEqual(
            support.selected_representation(records([0.2, 0.3, 0.3, 0.3], [0.5, 0.1, 0.1, 0.1]))[
                "variant"
            ],
            support.IDS[1],
        )

    def test_bootstrap_is_paired_subject_deterministic_and_descriptive(self):
        data = fixture(rows=12, subjects=3)
        p = {v: np.linspace(0.1, 0.9, 12) for v in support.IDS}
        first = support.paired_bootstrap(data, p)
        second = support.paired_bootstrap(data, p)
        self.assertEqual(first, second)
        self.assertEqual(first["requested_replicates"], 1000)
        self.assertEqual(first["valid_replicates"], 1000)
        self.assertFalse(first["used_for_selection"])
        self.assertTrue(
            all(value == [0.0, 0.0] for value in first["AP_difference_95pct_percentile"].values())
        )

    def test_complete_external_workflow_with_mock_SDK_no_real_requests(self):
        sizes = np.array_split(np.arange(13975), 90)
        training = fixture(
            13975, 90, prefix="fake_train", positives=[v[:12] for v in sizes[:41]] + [sizes[41][:6]]
        )
        qsizes = np.array_split(np.arange(3193), 23)
        tuning = fixture(
            3193,
            23,
            prefix="fake_query",
            positives=[v[:10] for v in qsizes[:9]] + [qsizes[0][10:14]],
        )
        fit_calls = []
        query_calls = []
        with tempfile.TemporaryDirectory() as d, ExitStack() as stack:
            temp = Path(d)
            baseline = temp / "baseline.csv"
            export_predictions(
                tuning,
                np.full(3193, 0.03),
                model="tabpfn_full",
                candidate_id="v3p5_default_seed42",
                probability_output=True,
            ).to_csv(baseline, index=False)
            baseline_before = baseline.read_bytes()
            candidates = {}
            for v in support.IDS[1:]:
                X, y, query, positions = support.matrices(training, tuning, PLAN, v)
                candidates[v] = {
                    "variant": v,
                    "training_X_sha256": support.array_hash(X),
                    "training_y_sha256": support.array_hash(y),
                    "tuning_X_sha256": support.array_hash(query),
                    "query_source_order_sha256": support.positions_hash(tuning.X.index.to_numpy()),
                }
            prep = {"candidates": candidates}
            support.atomic_json(temp / "preparation_manifest.json", prep, exclusive=True)
            support.atomic_json(
                temp / "validation_report.json", {"status": "SYNTHETIC_MOCK_TEST"}, exclusive=True
            )
            support.atomic_json(
                temp / "D_context_selection_manifest.json",
                support.context_manifest(training, positions),
                exclusive=True,
            )
            for mod in [runner, worker, support]:
                stack.enter_context(patch.object(mod, "STUDY", temp))
            for mod in [runner, worker]:
                stack.enter_context(patch.object(mod, "verified_spec", return_value=PLAN))
                stack.enter_context(patch.object(mod, "verify_preparation", return_value=prep))
                stack.enter_context(
                    patch.object(mod, "environment", return_value={"mock_only": True})
                )
                stack.enter_context(
                    patch.object(
                        mod,
                        "load_development",
                        side_effect=lambda _p: (
                            SimpleNamespace(access_log=["training", "tuning"]),
                            training,
                            tuning,
                        ),
                    )
                )
            stack.enter_context(patch.object(runner, "BASELINE", baseline))
            stack.enter_context(patch.object(runner, "require_inactive"))
            stack.enter_context(patch.object(support, "require_inactive"))

            class MockClient:
                def __init__(self, **kwargs):
                    self.assert_config = kwargs
                    if kwargs != support.CONSTRUCTOR:
                        raise AssertionError("Changed constructor")
                    self.classes_ = np.array([0, 1])

                def fit(self, X, y):
                    fit_calls.append(X.shape[1])

                def predict_proba(self, X):
                    query_calls.append(len(X))
                    prob = np.where(
                        tuning.y.to_numpy() == 1, 0.9 if X.shape[1] == 78 else 0.8, 0.01
                    )
                    self._last_meta = {
                        "package_version": "9.0.0",
                        "billing_model_version": "v3.5",
                        "n_estimators": 8,
                        "execution_mode": "standard",
                        "cache_outcome": "miss",
                        "classes": [0, 1],
                        "test_set_num_rows": len(X),
                        "test_set_num_cols": X.shape[1],
                        "tabpfn_config": support.CONSTRUCTOR,
                    }
                    return np.column_stack([1 - prob, prob])

            stack.enter_context(
                patch.dict(
                    sys.modules,
                    {
                        "tabpfn_client": SimpleNamespace(TabPFNClassifier=MockClient),
                        "httpx": SimpleNamespace(
                            Client=type("HTTPClient", (), {"send": lambda *_a, **_k: None})
                        ),
                    },
                )
            )

            def mock_supervise(v, directory, logs):
                logs.mkdir()
                worker.execute(v)
                support.atomic_json(logs / "exit.json", {"returncode": 0}, exclusive=True)

            stack.enter_context(patch.object(runner, "supervise", side_effect=mock_supervise))
            with patch("sys.stdout", io.StringIO()):
                result = runner.run()
            self.assertEqual(result["status"], "COMPLETE")
            self.assertEqual(baseline.read_bytes(), baseline_before)
            self.assertEqual(fit_calls, [71, 78, 74])
            self.assertEqual(query_calls, [3193] * 3)
            self.assertEqual(result["selected_representation"], support.IDS[2])
            self.assertEqual(result["D_context"]["retained_positives"], 498)
            self.assertFalse(result["model_lock_written"])
            self.assertTrue((temp / "tabpfn_representation_study_review.zip").exists())
            with zipfile.ZipFile(temp / "tabpfn_representation_study_review.zip") as z:
                self.assertIsNone(z.testzip())
            with patch("sys.stdout", io.StringIO()):
                runner.run()
            self.assertEqual(baseline.read_bytes(), baseline_before)
            self.assertEqual(fit_calls, [71, 78, 74])
            candidate = temp / "execution/candidates" / support.IDS[1]
            original = (candidate / "bundle/prediction.csv").read_bytes()
            with self.assertRaises(FileExistsError):
                worker.execute(support.IDS[1])
            self.assertEqual(original, (candidate / "bundle/prediction.csv").read_bytes())
            (candidate / "bundle/prediction.csv").write_bytes(original + b"corrupted")
            with self.assertRaises(ValueError):
                runner.run()
            self.assertEqual(baseline.read_bytes(), baseline_before)
            self.assertEqual(fit_calls, [71, 78, 74])
            self.assertFalse((temp / "model_lock_manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
