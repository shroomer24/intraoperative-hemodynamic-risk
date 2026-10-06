"""Exactly one MAP recovery worker; no full-model or TEST execution capability."""

import resource
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import map_recovery_support as support  # noqa: E402
import numpy as np  # noqa: E402


def execute():
    output = support.resolve_attempt()
    sequence = 0
    last_stage = None

    def mark(stage):
        nonlocal sequence, last_stage
        sequence += 1
        support.atomic_json(
            output / "stages" / f"{sequence:02d}_{stage}.json",
            {"stage": stage, "timestamp_utc": support.stamp()},
        )
        last_stage = stage

    loader = support.RecoveryPartitions()
    loader.authorize()
    support.atomic_json(output / "worker_claim.json", {"status": "CLAIMED"})
    try:
        mark("WORKER_STARTED")
        training = loader.load("training")
        query = loader.load("calibration")
        X, query_X, contract = support.input_contract(loader, training, query)
        mark("INPUT_VALIDATED")
        support.atomic_json(output / "input_contract.json", contract)
        if "xgboost" in sys.modules:
            raise RuntimeError("Hosted worker cannot co-import XGBoost")
        with support.quiet_sdk():
            import httpx
            from tabpfn_client import TabPFNClassifier

            with support.one_shot_transport(httpx):
                model = TabPFNClassifier(model_path="v3.5_default", random_state=42)
                params = model.get_params(deep=False)
                if any(params.get(k) != v for k, v in support.CONSTRUCTOR.items()):
                    raise support.ScientificFailure("constructor", None, "MODEL_MISMATCH")
                mark("CLIENT_INITIALIZED")
                loader.authorize()
                mark("HOSTED_FIT_STARTED")
                model.fit(X, training.y.to_numpy(dtype=int))
                mark("HOSTED_FIT_RETURNED")
                mark("PREDICT_PROBA_STARTED")
                probabilities = model.predict_proba(query_X)
                mark("PREDICT_PROBA_RETURNED")
            raw = getattr(model, "_last_meta", {})
        loader.authorize()
        params = model.get_params(deep=False)
        if any(params.get(k) != v for k, v in support.CONSTRUCTOR.items()):
            raise support.ScientificFailure("constructor", None, "MODEL_MISMATCH")
        _, _, observed = support.input_contract(loader, training, query)
        scores, optional = support.persist_approved_result(
            output / "probability_checkpoint",
            probabilities,
            model.classes_,
            contract,
            observed,
            raw,
            amendment_hash=loader.amendment_hash,
            mark=mark,
        )
        support.atomic_json(output / "optional_metadata.json", optional)
        if not np.array_equal(scores, np.asarray(probabilities)[:, 1]):
            raise support.ScientificFailure("probabilities", None, "COLUMN_MISMATCH")
        loader.authorize()
        mark("COMPLETE")
        support.seal_json(
            output / "completion_receipt.json",
            {
                "status": "COMPLETE",
                "phase": "calibration",
                "model": "tabpfn_map",
                "attempt": 2,
                "timestamp_utc": support.stamp(),
                "execution_amendment_sha256": loader.amendment_hash,
                "model_lock_sha256": support.LOCK_HASH,
                "calibration_protocol_sha256": support.PROTOCOL_HASH,
                "partition_access": loader.access_log,
                "hosted_fit_calls": 1,
                "predict_proba_calls": 1,
                "files_sha256": support.receipt_files(output),
                "review_boundary": "STOP_BEFORE_FULL_OR_PLATT",
                "automatic_retry": False,
            },
        )
        return 0
    except BaseException as exc:
        failure = {
            "status": "FAILED",
            "phase": "calibration",
            "model": "tabpfn_map",
            "attempt": 2,
            "timestamp_utc": support.stamp(),
            "last_completed_stage": last_stage,
            "exception_class": support.safe_exception(exc),
            "diagnostic": exc.diagnostic if isinstance(exc, support.ScientificFailure) else None,
            "raw_exception_messages_retained": False,
            "automatic_retry": False,
            "probability_checkpoint_exists": (
                output / "probability_checkpoint/manifest.json"
            ).exists(),
        }
        status = getattr(exc, "http_status", None)
        if type(status) is int and 100 <= status <= 599:
            failure.update(http_status=status, http_category=f"HTTP_{status // 100}XX")
        support.seal_json(output / "failure.json", failure)
        return 2


def main():
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    if len(sys.argv) != 1:
        return 2
    try:
        return execute()
    except BaseException:
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
