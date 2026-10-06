"""One external-terminal TabPFN candidate, one fit/query, atomic immutable bundle."""

import argparse
import contextlib
import io
import json
import logging
import os
import sys
import tempfile
import threading
from pathlib import Path

import recovery_support as recovery
from study_support import (
    CONSTRUCTOR,
    IDS,
    SPEC_HASH,
    STUDY,
    array_hash,
    atomic_json,
    class_one_probability,
    context_manifest,
    digest_json,
    matrices,
    read_json,
    sha256,
)

from intraop.evaluation.development import development_metrics, export_predictions


class OneShotTransportFailure(BaseException):
    """Typed HTTP status only; bypass SDK retry loops without retaining payloads."""

    def __init__(self, status=None):
        self.http_status = status if recovery.http_category(status) is not None else None
        super().__init__("Hosted transport failed; no automatic retry")


@contextlib.contextmanager
def no_transport_retries(httpx_module, attempt):
    original = httpx_module.Client.send
    lock = threading.Lock()
    sequence = 0

    def event(status, http_status=None):
        nonlocal sequence
        with lock:
            sequence += 1
            value = {"status": status, "timestamp_utc": recovery.original.timestamp()}
            if http_status is not None:
                value.update(
                    http_status=http_status, http_category=recovery.http_category(http_status)
                )
            atomic_json(attempt / "http_events" / f"{sequence:06d}.json", value, exclusive=True)

    def send(client, *args, **kwargs):
        event("REQUEST_STARTED")
        try:
            response = original(client, *args, **kwargs)
        except Exception:
            raise OneShotTransportFailure() from None
        status = response.status_code
        if recovery.http_category(status) is None:
            raise OneShotTransportFailure() from None
        event("RESPONSE_RECEIVED", status)
        if status >= 400 and status != 409:
            raise OneShotTransportFailure(status) from None
        return response

    httpx_module.Client.send = send
    try:
        yield
    finally:
        httpx_module.Client.send = original


@contextlib.contextmanager
def quiet_sdk():
    previous = logging.root.manager.disable
    logging.disable(logging.CRITICAL)
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            yield
    finally:
        logging.disable(previous)


def execute(variant, attempt_number):
    from intraop.evaluation.locked_execution import ensure_study_open

    ensure_study_open(STUDY)
    if attempt_number != recovery.attempt_number(variant):
        raise PermissionError("Only specifically approved B2/C1/D1 may execute")
    plan, prep, loader, training, tuning, auth, env = recovery.checked_context(external=True)
    attempt = recovery.attempt_path(variant)
    if (recovery.RECOVERY / "study_stop.json").exists():
        raise RuntimeError("New unresolved failure blocks hosted execution")
    if variant != recovery.B:
        from recovery_run import check_bundle

        if (
            recovery.state(recovery.B) != "COMPLETE"
            or not (recovery.attempt_path(recovery.B) / "completion.json").exists()
        ):
            raise RuntimeError("B recovery must complete before C/D")
        check_bundle(STUDY / "execution/candidates" / recovery.B, tuning, recovery.B, prep)
    directory = STUDY / "execution/candidates" / variant
    if "xgboost" in sys.modules:
        raise RuntimeError("XGBoost must not be co-loaded in hosted worker")
    reserved = read_json(attempt / "reservation.json")
    expected = prep["candidates"][variant]
    if reserved["scientific_identity_sha256"] != digest_json(expected):
        raise ValueError("Attempt scientific configuration changed")
    if variant == recovery.B and reserved.get("authorization_sha256") != sha256(
        recovery.RECOVERY / "B_attempt_2_authorization.json"
    ):
        raise PermissionError("Exact B recovery authorization is missing")
    if (directory / "bundle").exists():
        raise FileExistsError("Completed candidate bundle already exists")
    atomic_json(
        attempt / "worker_started.json",
        {
            "status": "STARTED",
            "pid": os.getpid(),
            "variant": variant,
            "scientific_identity_sha256": digest_json(expected),
        },
        exclusive=True,
    )
    recovery.mark_stage(attempt, "WORKER_STARTED")
    X, y, query, positions = matrices(training, tuning, plan, variant)
    if (
        array_hash(X) != expected["training_X_sha256"]
        or array_hash(query) != expected["tuning_X_sha256"]
        or array_hash(y) != expected["training_y_sha256"]
    ):
        raise ValueError("Prepared candidate matrix hashes differ")
    if variant == IDS[3]:
        selected = context_manifest(training, positions)
        if selected != read_json(STUDY / "D_context_selection_manifest.json"):
            raise ValueError("D context selection differs from preparation")
        if selected["retained_positives"] != 498 or selected["retained_subjects"] != 90:
            raise ValueError("D scientific context invariants failed")
        # Evidence already exists before importing/authenticating the hosted estimator.
        atomic_json(directory / "D_context_selection_manifest.json", selected, exclusive=True)
    recovery.mark_stage(attempt, "INPUT_VALIDATED")
    with quiet_sdk():
        import httpx

        with no_transport_retries(httpx, attempt):
            recovery.mark_stage(attempt, "CLIENT_INITIALIZATION_STARTED")
            from tabpfn_client import TabPFNClassifier

            model = TabPFNClassifier(model_path="v3.5_default", random_state=42)
            recovery.mark_stage(attempt, "CLIENT_INITIALIZED")
            recovery.mark_stage(attempt, "HOSTED_CALL_STARTED")
            recovery.mark_stage(attempt, "HOSTED_FIT_STARTED")
            model.fit(X, y)
            recovery.mark_stage(attempt, "HOSTED_FIT_RETURNED")
            recovery.mark_stage(attempt, "PREDICT_PROBA_STARTED")
            probabilities = model.predict_proba(query)
            recovery.mark_stage(attempt, "PREDICT_PROBA_RETURNED")
            recovery.mark_stage(attempt, "HOSTED_CALL_RETURNED")
        from study_support import safe_metadata

        recovery.mark_stage(attempt, "RESPONSE_PROCESSING_STARTED")
        remote = safe_metadata(
            getattr(model, "_last_meta", {}), query_rows=len(query), feature_count=query.shape[1]
        )
        recovery.mark_stage(attempt, "RESPONSE_PROCESSING_PASSED")
    if "xgboost" in sys.modules:
        raise RuntimeError("Hosted worker unexpectedly co-loaded XGBoost")
    recovery.mark_stage(attempt, "PROBABILITY_VALIDATION_STARTED")
    scores = class_one_probability(probabilities, model.classes_, len(query))
    recovery.mark_stage(attempt, "PROBABILITY_VALIDATION_PASSED")
    prediction = export_predictions(
        tuning,
        scores,
        model="tabpfn_representation_study",
        candidate_id=variant,
        probability_output=True,
    )
    record = {
        "variant": variant,
        "simplicity_rank": IDS.index(variant),
        **development_metrics(tuning.y, scores, probability_output=True),
    }
    provenance = {
        "variant": variant,
        "attempt_number": attempt_number,
        "execution_recovery_authorization_sha256": sha256(
            recovery.RECOVERY / "B_attempt_2_authorization.json"
        ),
        "study_plan_sha256": SPEC_HASH,
        "constructor": CONSTRUCTOR,
        "environment": env,
        "training_shape": list(X.shape),
        "query_shape": list(query.shape),
        "X_dtype": str(X.dtype),
        "y_dtype": str(y.dtype),
        "classes": [0, 1],
        "positive_probability_column": 1,
        "partition_access_log": loader.access_log,
        "prepared_candidate_sha256": digest_json(expected),
        "query_source_order_sha256": expected["query_source_order_sha256"],
        "fresh_exec_process": True,
        "xgboost_not_imported": True,
        "fit_calls": 1,
        "predict_proba_calls": 1,
        "automatic_transport_retries": False,
        "remote_order_assurance": (
            "API positional contract plus export alignment; no additional query"
        ),
        "credential_values_recorded": False,
    }
    recovery.mark_stage(attempt, "ARTIFACT_PUBLICATION_STARTED")
    temporary = Path(tempfile.mkdtemp(prefix=".unpublished-", dir=attempt))
    prediction.to_csv(temporary / "prediction.csv", index=False)
    atomic_json(temporary / "metrics.json", record, exclusive=True)
    atomic_json(temporary / "remote_metadata.json", remote, exclusive=True)
    atomic_json(temporary / "provenance.json", provenance, exclusive=True)
    atomic_json(
        temporary / "bundle_manifest.json",
        {
            "status": "COMPLETE",
            "variant": variant,
            "files_sha256": {p.name: sha256(p) for p in temporary.iterdir()},
        },
        exclusive=True,
    )
    # Reservation + run lock exclude competing writers; never replace an existing bundle.
    if (directory / "bundle").exists():
        raise FileExistsError("Bundle publication already exists")
    os.rename(temporary, directory / "bundle")
    recovery.mark_stage(attempt, "COMPLETE")
    return {"status": "COMPLETE", "variant": variant}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=IDS[1:], required=True)
    parser.add_argument("--attempt", type=int, choices=(1, 2), required=True)
    args = parser.parse_args()
    try:
        result = execute(args.variant, args.attempt)
    except BaseException as exc:
        attempt = recovery.attempt_path(args.variant)
        if (
            args.attempt == recovery.attempt_number(args.variant)
            and (attempt / "worker_started.json").exists()
            and read_json(attempt / "worker_started.json").get("pid") == os.getpid()
        ):
            failure = {
                "status": "FAILED",
                "error_type": recovery.safe_error_class(exc),
                **recovery.stage_summary(attempt),
                "http_failure_status": exc.http_status
                if isinstance(exc, OneShotTransportFailure)
                else None,
                "http_failure_category": recovery.http_category(exc.http_status)
                if isinstance(exc, OneShotTransportFailure)
                else None,
                "raw_payloads_or_messages_retained": False,
                "timestamp_utc": recovery.original.timestamp(),
            }
            atomic_json(attempt / "failure.json", failure, exclusive=True)
        # Never persist raw exception messages, frame locals, SDK streams or payloads.
        result = {
            "status": "WORKER_FAILED",
            "variant": args.variant,
            "error_type": recovery.safe_error_class(exc),
            "details": "No raw SDK payload retained",
        }
        print(json.dumps(result), flush=True)
        return 2
    print(json.dumps(result), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
