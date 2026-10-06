"""Exact-client environment gate and one training-only repository smoke test."""

import contextlib
import importlib.metadata
import io
import logging
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from intraop.data.modeling import FrozenPartitions, predictor_array, write_json


def main():
    repo = Path(__file__).resolve().parents[1]
    modeling = repo / "artifacts/modeling-v01"
    artifact = modeling / "validation/access_gate_resolved.json"
    if artifact.exists():
        raise RuntimeError("Smoke-test evidence already exists; do not repeat automatically")
    evidence = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "python_executable": sys.executable,
        "python_version": sys.version.split()[0],
        "tabpfn_client_version": importlib.metadata.version("tabpfn-client"),
        "TABPFN_TOKEN_present": bool(os.environ.get("TABPFN_TOKEN")),
        "model_identifier": "v3.5_default",
        "hosted_inference": True,
        "secret_recorded": False,
        "test_partition_accessed": False,
        "seed": 42,
        "status": "RUNNING",
    }
    stage = "environment"
    try:
        if evidence["tabpfn_client_version"] != "0.6.1" or not evidence["TABPFN_TOKEN_present"]:
            raise RuntimeError("Exact environment prerequisite failed")
        partitions = FrozenPartitions(repo / "artifacts/vitaldb-cohort-v03", modeling)
        evidence["integrity"] = partitions.verify_integrity()
        training = partitions.load("training")
        rng = np.random.default_rng(42)
        fit_positions = np.sort(rng.choice(len(training.X), 256, replace=False))
        remaining = np.setdiff1d(np.arange(len(training.X)), fit_positions)
        query_positions = rng.choice(remaining, 12, replace=False)
        query_positions = np.concatenate([query_positions, query_positions[:4]])
        fit = training.subset(fit_positions)
        query = training.subset(query_positions[:12])
        Xfit = predictor_array(fit, partitions.features)
        Xquery = predictor_array(training, partitions.features)[query_positions]
        if np.unique(fit.y).tolist() != [0, 1]:
            raise ValueError("Smoke training sample lacks a class")
        # Only numeric X and binary TRAINING labels reach the hosted estimator.
        logging.getLogger("tabpfn_client").setLevel(logging.CRITICAL)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            from tabpfn_client import TabPFNClassifier

            stage = "initialize_v3.5_default"
            model = TabPFNClassifier(model_path="v3.5_default", random_state=42)
            evidence["initializes"] = True
            stage = "training_only_fit"
            model.fit(Xfit, fit.y.to_numpy(dtype=int))
            evidence["fit_succeeded"] = True
            stage = "training_only_predict_proba"
            probabilities = np.asarray(model.predict_proba(Xquery))
            classes = np.asarray(model.classes_)
            if classes.tolist() != [0, 1] or probabilities.shape != (16, 2):
                raise ValueError("Binary probability shape/classes mismatch")
            if not np.isfinite(probabilities).all() or (probabilities < 0).any():
                raise ValueError("Invalid probabilities")
            if not np.allclose(probabilities.sum(axis=1), 1, atol=1e-6):
                raise ValueError("Probabilities do not sum to one")
            if not np.allclose(probabilities[:4], probabilities[12:], atol=1e-5):
                raise ValueError("Duplicate-query row alignment failed")
            stage = "query_order_verification"
            reversed_probabilities = np.asarray(model.predict_proba(Xquery[::-1].copy()))
            if not np.allclose(probabilities, reversed_probabilities[::-1], atol=1e-5):
                raise ValueError("Reversed-query row alignment failed")
        evidence.update(
            {
                "status": "PASS",
                "classes": classes.tolist(),
                "positive_probability_column": 1,
                "predict_proba_shape": list(probabilities.shape),
                "row_count_verified": True,
                "row_order_verified_by_reversal_and_duplicates": True,
                "fit_rows": len(fit.X),
                "fit_positives": int(fit.y.sum()),
                "query_rows": 16,
                "query_unique_rows": len(query.X),
                "fit_source_rows": fit.X.index.tolist(),
                "query_source_rows": training.X.index[query_positions].tolist(),
                "predictors": partitions.features,
                "positive_class_semantics": (
                    "1 = new sustained hypotension episode beginning within 5 minutes"
                ),
                "uploaded_fields": ["frozen numeric predictor array", "binary training labels"],
                "partition_access_log": partitions.access_log,
            }
        )
    except Exception as exc:
        # Exception text and SDK logs are deliberately not persisted.
        evidence.update(
            {
                "status": "FAIL",
                "failed_stage": stage,
                "non_secret_exception_type": type(exc).__name__,
            }
        )
        write_json(artifact, evidence, exclusive=True)
        print(f"Access gate FAIL: stage={stage}; exception_type={type(exc).__name__}", flush=True)
        return 2
    write_json(artifact, evidence, exclusive=True)
    print(
        "Access gate PASS: token present; client 0.6.1; v3.5_default fit/predict succeeded; "
        "classes [0, 1]; positive column 1; shape/order verified; TRAINING only",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
