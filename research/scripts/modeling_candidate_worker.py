"""Narrow candidate execution. Atomic bundles appear only after successful fit."""

import argparse
import faulthandler
import importlib.metadata
import json
import os
import platform
import resource
import shutil
import sys
import tempfile
from pathlib import Path

import joblib
import numpy as np
from modeling_process_support import (
    assert_isolated,
    candidate_definition,
    load_training,
    verified_plan,
)

from intraop.data.modeling import predictor_array, sha256, write_json
from intraop.evaluation.development import development_metrics, export_predictions
from intraop.evaluation.secure_errors import sanitized_traceback
from intraop.models.benchmarks import build_estimator, positive_probabilities, quiet_hosted_call


def fit_scores(kind, model, X, y, query):
    if kind == "xgboost":
        assert_isolated()
    with quiet_hosted_call():
        model.fit(X, y)
        scores = positive_probabilities(model, query, require_ordered_classes=True)
    if kind == "xgboost":
        assert_isolated()
    return scores


def execute(kind, args):
    plan, _ = verified_plan()
    model_name = args.model or "xgboost"
    definition, candidate = candidate_definition(plan, model_name, args.candidate)
    if definition["kind"] != kind:
        raise ValueError("Worker kind/candidate mismatch")
    if kind == "xgboost":
        assert_isolated()
        if importlib.metadata.version("xgboost") != "3.3.0" or np.__version__ != "2.4.6":
            raise ValueError("XGBoost/NumPy version mismatch")
    elif not os.environ.get("TABPFN_TOKEN"):
        raise RuntimeError("TABPFN_TOKEN unavailable; no credential values recorded")
    if args.validation and kind != "xgboost":
        raise PermissionError("No repeated hosted smoke is permitted")
    output = Path(args.output)
    if output.exists():
        raise FileExistsError("Worker output already exists")
    if args.validation == "synthetic":
        X = np.random.default_rng(42).normal(size=(256, 74))
        y = (X[:, :3].sum(axis=1) > 0).astype(int)
        query = X[:16].copy()
        loader = None
    else:
        loader, training = load_training(plan)
        X = predictor_array(training, definition["features"])
        y = training.y.to_numpy(dtype=int)
        if args.validation == "training":
            positions = np.sort(np.random.default_rng(42).choice(len(X), 256, replace=False))
            queries = np.setdiff1d(np.arange(len(X)), positions)[:16]
            query, X, y = X[queries], X[positions], y[positions]
        else:
            tuning = loader.load("tuning")
            if len(tuning.X) != 3193 or set(training.groups) & set(tuning.groups):
                raise ValueError("Frozen TUNING count/subject disjointness differs")
            query = predictor_array(tuning, definition["features"])
    if kind == "xgboost":
        assert_isolated()
    with quiet_hosted_call():
        model = build_estimator(kind, candidate["hyperparameters"])
    scores = fit_scores(kind, model, X, y, query)
    provenance = {
        "model": model_name,
        "candidate_id": args.candidate,
        "constructor_arguments": candidate["hyperparameters"],
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "versions": plan["package_versions"],
        "classes": [0, 1],
        "positive_probability_column": 1,
        "positive_class": 1,
        "positive_class_semantics": "new sustained hypotension episode beginning within 5 minutes",
        "training_rows": len(X),
        "predictor_count": X.shape[1],
        "query_rows": len(query),
        "partition_access_log": loader.access_log if loader else [],
        "row_alignment": "source order; numeric positional queries and exports",
        "forbidden_runtime_modules_absent": kind == "xgboost",
        "credential_values_recorded": False,
    }
    if args.validation:
        provenance.update(
            {
                "status": "PASS",
                "validation": args.validation,
                "X_shape": list(X.shape),
                "y_shape": list(y.shape),
                "y_dtype": str(y.dtype),
                "y_class_counts": {str(c): int((y == c).sum()) for c in np.unique(y)},
                "NaN_count": int(np.isnan(X).sum()),
                "positive_inf_count": int(np.isposinf(X).sum()),
                "negative_inf_count": int(np.isneginf(X).sum()),
                "probability_shape": [len(scores), 2],
            }
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".candidate-", dir=output.parent))
    try:
        write_json(temporary / "provenance.json", provenance, exclusive=True)
        if not args.validation:
            prediction = export_predictions(
                tuning,
                scores,
                model=model_name,
                candidate_id=args.candidate,
                probability_output=True,
            )
            prediction.to_csv(temporary / "prediction.csv", index=False)
            record = {
                "model": model_name,
                "candidate_id": args.candidate,
                "simplicity_rank": candidate["simplicity_rank"],
                "hyperparameters": json.dumps(candidate["hyperparameters"], sort_keys=True),
                **development_metrics(tuning.y, scores, probability_output=True),
            }
            write_json(temporary / "record.json", record, exclusive=True)
            if kind == "xgboost":
                joblib.dump(model, temporary / "model.joblib")
        write_json(
            temporary / "bundle_manifest.json",
            {
                "status": "COMPLETE",
                "files_sha256": {p.name: sha256(p) for p in temporary.iterdir()},
            },
            exclusive=True,
        )
        os.rename(temporary, output)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    print(
        json.dumps(
            {
                "status": "PASS",
                "model": model_name,
                "candidate_id": args.candidate,
                "validation": args.validation,
            }
        ),
        flush=True,
    )


def main(kind):
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    faulthandler.enable()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--model")
    parser.add_argument("--output", required=True)
    parser.add_argument("--validation", choices=["synthetic", "training"])
    args = parser.parse_args()
    try:
        execute(kind, args)
    except Exception as exc:
        print(sanitized_traceback(exc), file=sys.stderr, flush=True)
        return 2
    return 0
