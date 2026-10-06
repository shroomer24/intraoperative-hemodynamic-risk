"""One fresh TEST worker; existing conventional fits, one hosted fit/query per TabPFN."""

import argparse
import resource
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import final_test_support as s  # noqa: E402
import numpy as np  # noqa: E402


def conventional_model(loader, name):
    import joblib

    record = loader.registry["sources"][name]
    path = s.a.safe_path(record["fitted_model_path"])
    if s.a.file_hash(path) != record["fitted_model_sha256"]:
        raise PermissionError("Preserved fitted-model hash differs before deserialization")
    model = joblib.load(path)
    constructor = record["constructor"]
    if name == "prevalence":
        if model.prevalence_ != loader.lock["training_context"]["prevalence"]:
            raise PermissionError("Fixed TRAINING prevalence differs")
    else:
        params = model.get_params(deep=True)
        prefix = "classifier__" if name.startswith("logistic") else ""
        if any(params.get(prefix + key) != value for key, value in constructor.items()):
            raise PermissionError("Preserved fitted-model parameters differ")
        if name.startswith("logistic") and (
            params.get("imputer__strategy") != "median"
            or params.get("imputer__keep_empty_features") is not True
            or params.get("scaler__with_mean") is not True
            or params.get("scaler__with_std") is not True
        ):
            raise PermissionError("Locked preprocessing differs")
    return model


def execute(name):
    if name not in s.MODELS:
        raise PermissionError("Unknown TEST model")
    blocker = s.RuntimeBlocker(name)
    if any(key.split(".")[0] in blocker.forbidden for key in sys.modules):
        raise PermissionError("Fresh TEST worker runtime is contaminated")
    sys.meta_path.insert(0, blocker)
    if name not in s.HOSTED:
        sys.addaudithook(s.a.deny_network)
    loader = s.TestPartitions()
    loader.authorize()
    output = s.ROOT / "workers" / name
    output.mkdir(parents=True, exist_ok=False)
    sequence, last_stage = 0, None

    def mark(stage):
        nonlocal sequence, last_stage
        sequence += 1
        s.a.atomic_json(
            output / "stages" / f"{sequence:02d}_{stage}.json",
            {
                "stage": stage,
                "timestamp_utc": s.a.stamp(),
            },
        )
        last_stage = stage

    s.a.atomic_json(output / "worker_claim.json", {"status": "CLAIMED"})
    try:
        mark("WORKER_STARTED")
        training, query = loader.load("training"), loader.load("test")
        X, query_X, contract = s.input_contract(loader, name, training, query)
        s.a.atomic_json(output / "input_contract.json", contract)
        mark("INPUT_VALIDATED")
        fit_calls, predict_calls = 0, 0
        if name == "current_map":
            scores = -query_X[:, 0]
            if not np.isfinite(scores).all():
                raise ValueError("Current-MAP discrimination score must be finite")
            cp = output / "score_checkpoint"
            cp.mkdir(exist_ok=False)
            s.a.atomic_binary(
                cp / "scores.npy", lambda stream: np.save(stream, scores, allow_pickle=False)
            )
            s.a.seal_json(
                cp / "manifest.json",
                {
                    "scientific_validation": "PASS",
                    "contract": contract,
                    "score_definition": "-map_latest",
                    "scores_sha256": s.a.file_hash(cp / "scores.npy"),
                    "probability_mapping": False,
                },
            )
            cp.chmod(0o555)
            mark("SCORE_CHECKPOINT_PUBLISHED")
        else:
            if name in s.HOSTED:
                if "xgboost" in sys.modules:
                    raise PermissionError("Hosted worker cannot co-import XGBoost")
                with s.a.quiet_sdk():
                    import httpx
                    from tabpfn_client import TabPFNClassifier

                    with s.one_shot_transport(httpx, output):
                        model = TabPFNClassifier(model_path="v3.5_default", random_state=42)
                        params = model.get_params(deep=False)
                        if any(
                            type(params.get(k)) is not type(v) or params.get(k) != v
                            for k, v in s.a.CONSTRUCTOR.items()
                        ):
                            raise PermissionError("Exact locked TabPFN constructor required")
                        mark("CLIENT_INITIALIZED")
                        loader.authorize()
                        mark("HOSTED_FIT_STARTED")
                        fit_calls += 1
                        model.fit(X, training.y.to_numpy(dtype=int))
                        mark("HOSTED_FIT_RETURNED")
                        mark("PREDICT_PROBA_STARTED")
                        predict_calls += 1
                        probabilities = model.predict_proba(query_X)
                        mark("PREDICT_PROBA_RETURNED")
                    raw = getattr(model, "_last_meta", {})
                params = model.get_params(deep=False)
                if any(
                    type(params.get(k)) is not type(v) or params.get(k) != v
                    for k, v in s.a.CONSTRUCTOR.items()
                ):
                    raise PermissionError("TabPFN constructor changed during hosted call")
            else:
                with s.a.quiet_sdk():
                    model = conventional_model(loader, name)
                    mark("PRESERVED_MODEL_LOADED")
                    mark("PREDICT_PROBA_STARTED")
                    predict_calls += 1
                    probabilities = model.predict_proba(query_X)
                    mark("PREDICT_PROBA_RETURNED")
                raw = {}
            loader.authorize()
            _, _, observed = s.input_contract(loader, name, training, query)
            scores, optional = s.checkpoint(
                output / "probability_checkpoint",
                probabilities,
                model.classes_,
                contract,
                observed,
                raw,
                mark=mark,
            )
            s.a.atomic_json(output / "optional_metadata.json", optional)
            if not np.array_equal(scores, np.asarray(probabilities)[:, 1]):
                raise ValueError("Positive class must remain probability column 1")
        loader.authorize()
        mark("COMPLETE")
        s.a.seal_json(
            output / "completion_receipt.json",
            {
                "status": "COMPLETE",
                "phase": "test",
                "model": name,
                "execution_preparation_sha256": loader.preparation_hash,
                "source_registry_sha256": s.REGISTRY_HASH,
                "model_lock_sha256": s.a.LOCK_HASH,
                "partition_access": loader.access_log,
                "fit_calls": fit_calls,
                "predict_proba_calls": predict_calls,
                "Platt_fit_calls": 0,
                "automatic_retry": False,
                "files_sha256": s.a.receipt_files(output),
                "timestamp_utc": s.a.stamp(),
            },
        )
        return 0
    except BaseException as exc:
        failure = {
            "status": "FAILED",
            "phase": "test",
            "model": name,
            "last_completed_stage": last_stage,
            "exception_class": s.a.safe_exception(exc),
            "diagnostic": exc.diagnostic if isinstance(exc, s.a.ScientificFailure) else None,
            "probability_checkpoint_exists": (
                output / "probability_checkpoint/manifest.json"
            ).exists(),
            "raw_exception_messages_retained": False,
            "automatic_retry": False,
        }
        status = getattr(exc, "http_status", None)
        if type(status) is int and 100 <= status <= 599:
            failure.update(http_status=status, http_category=f"HTTP_{status // 100}XX")
        s.a.seal_json(output / "failure.json", failure)
        return 2


def main():
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=s.MODELS, required=True)
    args = parser.parse_args()
    try:
        return execute(args.model)
    except BaseException:
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
