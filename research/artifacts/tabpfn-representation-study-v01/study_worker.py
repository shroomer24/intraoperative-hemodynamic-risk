"""One external-terminal TabPFN candidate, one fit/query, atomic immutable bundle."""

import argparse
import contextlib
import io
import json
import logging
import os
import sys
import tempfile
from pathlib import Path

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
    environment,
    load_development,
    matrices,
    read_json,
    sha256,
    verified_spec,
    verify_preparation,
)

from intraop.evaluation.development import development_metrics, export_predictions


class OneShotTransportFailure(BaseException):
    """Bypass SDK Exception retry loops; no original message or payload retained."""


@contextlib.contextmanager
def no_transport_retries(httpx_module):
    """Fail on the first transport/retryable/auth HTTP failure, not after SDK retries.

    Only the in-process HTTP transport boundary changes. Model parameters, installed
    library files and successful requests are untouched. HTTP 409 upload dedup is normal.
    """
    original = httpx_module.Client.send

    def send(client, *args, **kwargs):
        try:
            response = original(client, *args, **kwargs)
        except Exception:
            raise OneShotTransportFailure("Transport request failed; no retry permitted") from None
        if response.status_code >= 400 and response.status_code != 409:
            raise OneShotTransportFailure(
                f"HTTP status {response.status_code}; no retry permitted"
            ) from None
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


def execute(variant):
    from intraop.evaluation.locked_execution import ensure_study_open

    ensure_study_open(STUDY)
    plan = verified_spec()
    env = environment(plan, external=True)
    prep = verify_preparation(plan)
    directory = STUDY / "execution/candidates" / variant
    if "xgboost" in sys.modules:
        raise RuntimeError("XGBoost must not be co-loaded in hosted worker")
    reserved = read_json(directory / "attempt_1/reservation.json")
    expected = prep["candidates"][variant]
    if reserved["scientific_identity_sha256"] != digest_json(expected):
        raise ValueError("Attempt scientific configuration changed")
    if (directory / "bundle").exists():
        raise FileExistsError("Completed candidate bundle already exists")
    atomic_json(
        directory / "attempt_1/worker_started.json",
        {
            "status": "STARTED",
            "pid": os.getpid(),
            "variant": variant,
            "scientific_identity_sha256": digest_json(expected),
        },
        exclusive=True,
    )
    loader, training, tuning = load_development(plan)
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
    with quiet_sdk():
        import httpx
        from tabpfn_client import TabPFNClassifier

        with no_transport_retries(httpx):
            model = TabPFNClassifier(model_path="v3.5_default", random_state=42)
            model.fit(X, y)
            probabilities = model.predict_proba(query)
        from study_support import safe_metadata

        remote = safe_metadata(
            getattr(model, "_last_meta", {}), query_rows=len(query), feature_count=query.shape[1]
        )
    if "xgboost" in sys.modules:
        raise RuntimeError("Hosted worker unexpectedly co-loaded XGBoost")
    scores = class_one_probability(probabilities, model.classes_, len(query))
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
    temporary = Path(tempfile.mkdtemp(prefix=".unpublished-", dir=directory))
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
    return {"status": "COMPLETE", "variant": variant}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=IDS[1:], required=True)
    args = parser.parse_args()
    try:
        result = execute(args.variant)
    except BaseException as exc:
        # Never persist raw exception messages, frame locals, SDK streams or payloads.
        result = {
            "status": "WORKER_FAILED",
            "variant": args.variant,
            "error_type": type(exc).__name__,
            "details": "No raw SDK payload retained",
        }
        print(json.dumps(result), flush=True)
        return 2
    print(json.dumps(result), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
