"""Publication invariants and a mocked hosted interface, never a real client call."""

import hashlib
import json
import sys
import unittest
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

import numpy as np

from intraop.evaluation.calibration import apply_platt, fit_platt
from intraop.features.core import FEATURE_NAMES
from intraop.models.benchmarks import build_estimator, positive_probabilities

ROOT = Path(__file__).resolve().parents[1]


class PublicationTests(unittest.TestCase):
    def test_recovered_code_hashes(self):
        manifest = json.loads((ROOT / "provenance/source_manifest.json").read_text())
        for name, row in manifest["files"].items():
            self.assertEqual(
                hashlib.sha256((ROOT / name).read_bytes()).hexdigest(), row["published_sha256"]
            )
            if row["byte_identical"]:
                self.assertEqual(row["original_sha256"], row["published_sha256"])

    def test_feature_order_and_map_subset(self):
        schema = json.loads((ROOT / "reference/feature_schema.json").read_text())
        features = json.loads((ROOT / "reference/feature_sets.json").read_text())
        self.assertEqual(list(FEATURE_NAMES), schema["feature_names"])
        self.assertEqual(features["full"], list(FEATURE_NAMES))
        self.assertEqual(features["map_only"], [x for x in FEATURE_NAMES if x.startswith("map_")])
        self.assertEqual(len(features["map_only"]), 18)

    def test_mocked_tabpfn_constructor_and_class_one(self):
        fake = ModuleType("tabpfn_client")
        calls = []

        class Client:
            classes_ = np.array([0, 1])

            def __init__(self, **kw):
                calls.append(kw)

            def predict_proba(self, X):
                return np.tile([0.8, 0.2], (len(X), 1))

        fake.TabPFNClassifier = Client
        with patch.dict(sys.modules, {"tabpfn_client": fake}):
            m = build_estimator("tabpfn", {"model_path": "v3.5_default", "random_state": 42})
            np.testing.assert_array_equal(
                positive_probabilities(m, np.zeros((3, 74)), require_ordered_classes=True),
                [0.2] * 3,
            )
        self.assertEqual(calls, [{"model_path": "v3.5_default", "random_state": 42}])

    def test_model_substitution_rejected(self):
        fake = ModuleType("tabpfn_client")
        fake.TabPFNClassifier = lambda **kw: None
        with patch.dict(sys.modules, {"tabpfn_client": fake}):
            with self.assertRaises(ValueError):
                build_estimator("tabpfn", {"model_path": "different", "random_state": 42})

    def test_calibration_exact_frozen_protocol(self):
        from intraop.evaluation.locked_execution import CALIBRATION_PROTOCOL

        p = json.loads((ROOT / "reference/calibration_protocol.json").read_text())
        self.assertEqual(p, CALIBRATION_PROTOCOL)
        raw = np.array([0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9])
        y = np.array([0, 0, 0, 0, 1, 0, 1, 1])
        a = fit_platt(raw, y)
        b = fit_platt(raw, y)
        self.assertEqual(a, b)
        calibrated = apply_platt(raw, a)
        self.assertTrue(np.isfinite(calibrated).all())
        self.assertTrue(((calibrated >= 0) & (calibrated <= 1)).all())

    def test_safe_projection_is_not_a_lock(self):
        p = json.loads((ROOT / "reference/selected_models.json").read_text())
        self.assertEqual(p["provenance"]["type"], "PUBLIC_SAFE_PROJECTION_NOT_ORIGINAL_LOCK")
        self.assertFalse((ROOT / "artifacts/modeling-v01/model_lock_manifest.json").exists())

    def test_private_roster_and_raw_data_absent(self):
        self.assertFalse((ROOT / "private/cohort-selection.json").exists())
        self.assertFalse(list(ROOT.rglob("*.vital")))
        self.assertFalse(list(ROOT.rglob("*.joblib")))

    def test_development_firewall_rejects_heldout_without_IO(self):
        from intraop.data.modeling import DevelopmentPartitions

        loader = object.__new__(DevelopmentPartitions)
        for name in ["calibration", "test"]:
            with self.assertRaises(PermissionError):
                loader.load(name)

    def test_checkpoint_probability_contract(self):
        class Client:
            classes_ = np.array([1, 0])

            def predict_proba(self, X):
                return np.tile([0.2, 0.8], (len(X), 1))

        with self.assertRaises(ValueError):
            positive_probabilities(Client(), np.zeros((2, 18)), require_ordered_classes=True)

        class Broken(Client):
            classes_ = np.array([0, 1])

            def predict_proba(self, X):
                return np.array([[np.nan, 1], [0, 1]])

        with self.assertRaises(ValueError):
            positive_probabilities(Broken(), np.zeros((2, 18)))
