"""One locked model in a fresh process; hosted calls only in the approved terminal."""

from locked_bootstrap import REPO  # isort: skip

import argparse
import importlib.abc
import json
import os
import resource
import sys


class RuntimeBlocker(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in ("torch", "tabpfn", "tabpfn_client"):
            raise RuntimeError("Forbidden runtime in isolated XGBoost process")
        return None


if "--model" in sys.argv and sys.argv[sys.argv.index("--model") + 1] == "xgboost":
    sys.meta_path.insert(0, RuntimeBlocker())

import joblib  # noqa: E402
import numpy as np  # noqa: E402

from intraop.data.modeling import predictor_array, read_json  # noqa: E402
from intraop.evaluation.locked_execution import (  # noqa: E402
    MODEL_NAMES,
    LockedPartitions,
    phase_paths,
)
from intraop.evaluation.result_checkpoint import (  # noqa: E402
    SEMANTICS,
    ScientificFailure,
    array_hash,
    atomic_json,
    digest,
    persist_result,
    row_hash,
)
from intraop.models.benchmarks import build_estimator, quiet_hosted_call  # noqa: E402

MODELING = REPO / "artifacts/modeling-v01"


def safe_exception(exc):
    names = {
        "ValueError",
        "RuntimeError",
        "PermissionError",
        "FileExistsError",
        "OSError",
        "ConnectError",
        "HTTPStatusError",
        "TimeoutException",
        "ScientificFailure",
    }
    return type(exc).__name__ if type(exc).__name__ in names else "OTHER_EXCEPTION"


def contract_for(loader, name, training, query, X, query_X):
    definition = loader.lock["models"][name]
    return {
        "candidate": name,
        "kind": definition["kind"],
        "constructor": definition["selected_configuration"]["hyperparameters"],
        "features": definition["features"],
        "feature_count": len(definition["features"]),
        "feature_contract_sha256": digest(definition["features"]),
        "training_X_sha256": array_hash(X),
        "training_y_sha256": array_hash(training.y),
        "query_X_sha256": array_hash(query_X),
        "query_rows": len(query.X),
        "query_source_order_sha256": row_hash(query.X.index),
        "query_order_verified": bool(
            query.X.index.is_unique
            and query.X.index.is_monotonic_increasing
            and query.y.index.equals(query.X.index)
            and query.metadata.index.equals(query.X.index)
        ),
        "metadata_excluded_from_X": not bool(set(definition["features"]) & set(query.metadata)),
        "model_lock_sha256": loader.lock_hash,
        "dataset_hashes": loader.lock["table_hashes"],
        "split_manifest_sha256": loader.lock["split_manifest_hash"],
        "software_versions": loader.lock["package_versions"],
        "positive_class": 1,
        "probability_column": 1,
        "positive_class_semantics": SEMANTICS,
    }


def execute(name, phase):
    if os.environ.get("INTRAOP_LOCKED_EXTERNAL_EXECUTION") != "normal_terminal":
        raise PermissionError("Only approved normal-terminal locked execution is permitted")
    root, manifest, _ = phase_paths(MODELING, phase)
    output = root / "workers" / name
    output.mkdir(parents=True, exist_ok=False)
    sequence = 0

    def mark(stage):
        nonlocal sequence
        sequence += 1
        atomic_json(output / "stages" / f"{sequence:02d}_{stage}.json", {"stage": stage})

    try:
        mark("WORKER_STARTED")
        loader = LockedPartitions(MODELING, phase, read_json(manifest))
        training, query = loader.load("training"), loader.load(phase)
        definition = loader.lock["models"][name]
        expected_context = loader.lock["training_context"]
        if (
            len(training.X) != expected_context["rows"]
            or training.groups.nunique() != expected_context["subjects"]
            or int(training.y.sum()) != expected_context["positive_labels"]
        ):
            raise ScientificFailure("input_contract", None, "TRAINING_CONTEXT_MISMATCH")
        if definition["kind"] == "xgboost" and any(
            key.split(".")[0] in {"torch", "tabpfn", "tabpfn_client"} for key in sys.modules
        ):
            raise RuntimeError("Isolated XGBoost process is contaminated")
        if definition["kind"] == "tabpfn" and "xgboost" in sys.modules:
            raise RuntimeError("Hosted worker cannot co-import XGBoost")
        features = definition["features"]
        X, query_X = predictor_array(training, features), predictor_array(query, features)
        if np.unique(training.y).tolist() != [0, 1] or set(training.groups) & set(query.groups):
            raise ScientificFailure("input_contract", None, "TRAINING_OR_SUBJECT_MISMATCH")
        contract = contract_for(loader, name, training, query, X, query_X)
        mark("INPUT_VALIDATED")
        if name == "current_map":
            scores = -query_X[:, 0]
            if not np.isfinite(scores).all():
                raise ScientificFailure("current_map", scores, "NONFINITE_SCORE")
            atomic_json(output / "score_contract.json", contract)
            with (output / "scores.npy").open("xb") as stream:
                np.save(stream, scores, allow_pickle=False)
            mark("COMPLETE")
            return
        with quiet_hosted_call():
            if definition["kind"] == "tabpfn":
                if not os.environ.get("TABPFN_TOKEN"):
                    raise PermissionError("Secure credential unavailable")
                # The existing safe transport records typed status only and blocks SDK retry.
                recovery = REPO / "artifacts/tabpfn-representation-study-v01/recovery-b-v01"
                sys.path.insert(0, str(recovery.parent))
                sys.path.insert(0, str(recovery))
                import httpx
                from recovery_worker import no_transport_retries

                with no_transport_retries(httpx, output):
                    model = build_estimator(definition["kind"], contract["constructor"])
                    mark("CLIENT_INITIALIZED")
                    params = model.get_params(deep=False)
                    if any(params.get(k) != v for k, v in contract["constructor"].items()):
                        raise ScientificFailure("constructor", None, "MODEL_MISMATCH")
                    mark("HOSTED_CALL_STARTED")
                    model.fit(X, training.y.to_numpy(dtype=int))
                    mark("HOSTED_FIT_RETURNED")
                    probabilities = model.predict_proba(query_X)
                    mark("HOSTED_CALL_RETURNED")
                raw_metadata = getattr(model, "_last_meta", {})
            else:
                if phase == "test":
                    calibration_root, _, _ = phase_paths(MODELING, "calibration")
                    model = joblib.load(calibration_root / "workers" / name / "model.joblib")
                else:
                    model = build_estimator(definition["kind"], contract["constructor"])
                    model.fit(X, training.y.to_numpy(dtype=int))
                probabilities = model.predict_proba(query_X)
                raw_metadata = {}
        # Revalidate sealed sources/data, feature order and query position after the call.
        loader.authorize()
        if definition["kind"] == "tabpfn":
            params = model.get_params(deep=False)
            if any(params.get(k) != v for k, v in contract["constructor"].items()):
                raise ScientificFailure("constructor", None, "MODEL_MISMATCH")
        observed = contract_for(loader, name, training, query, X, query_X)
        scores, optional = persist_result(
            output / "probability_checkpoint",
            probabilities,
            model.classes_,
            contract,
            observed,
            raw_metadata,
            mark=mark,
        )
        atomic_json(output / "optional_metadata.json", optional)
        if phase == "calibration" and definition["kind"] != "tabpfn":
            joblib.dump(model, output / "model.joblib")
            (output / "model.joblib").chmod(0o444)
        if not np.array_equal(scores, probabilities[:, 1]):
            raise ScientificFailure("probabilities", None, "COLUMN_MISMATCH")
        mark("COMPLETE")
    except BaseException as exc:
        safe = {
            "status": "FAILED",
            "exception_class": safe_exception(exc),
            "stage": sorted((output / "stages").glob("*.json"))[-1].stem if sequence else None,
            "diagnostic": exc.diagnostic if isinstance(exc, ScientificFailure) else None,
            "raw_exception_messages_retained": False,
        }
        http_status = getattr(exc, "http_status", None)
        if type(http_status) is int and 100 <= http_status <= 599:
            safe.update(http_status=http_status, http_category=f"HTTP_{http_status // 100}XX")
        atomic_json(output / "failure.json", safe)
        raise


def main():
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODEL_NAMES, required=True)
    parser.add_argument("--phase", choices=("calibration", "test"), required=True)
    args = parser.parse_args()
    try:
        execute(args.model, args.phase)
    except BaseException as exc:
        print(json.dumps({"status": "FAILED", "exception_class": safe_exception(exc)}))
        return 2
    print(json.dumps({"status": "COMPLETE", "model": args.model}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
