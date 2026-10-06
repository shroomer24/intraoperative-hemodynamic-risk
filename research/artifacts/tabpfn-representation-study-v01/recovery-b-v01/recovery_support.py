"""One specifically approved B execution recovery; historical files remain immutable."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import re
import shlex
import subprocess

import numpy as np
import study_support as original

STUDY = original.STUDY
REPO = original.REPO
RECOVERY = STUDY / "recovery-b-v01"
B = original.IDS[1]
APPROVAL_SHA = "34908a30fba45aeaa28202cd56303d61819d6db4aceff28c4ca5bd9bac7133ad"
CODE_PATHS = tuple(
    f"artifacts/tabpfn-representation-study-v01/recovery-b-v01/{name}"
    for name in [
        "recovery_support.py",
        "recovery_worker.py",
        "recovery_run.py",
        "prepare_recovery.py",
    ]
) + ("scripts/run_tabpfn_representation_B_recovery.sh", "tests/test_tabpfn_B_recovery.py")
VALIDATION_FILES = ("validation_report.json", "software_tests.txt")
STAGES = (
    "WORKER_STARTED",
    "INPUT_VALIDATED",
    "CLIENT_INITIALIZATION_STARTED",
    "CLIENT_INITIALIZED",
    "HOSTED_CALL_STARTED",
    "HOSTED_FIT_STARTED",
    "HOSTED_FIT_RETURNED",
    "PREDICT_PROBA_STARTED",
    "PREDICT_PROBA_RETURNED",
    "HOSTED_CALL_RETURNED",
    "RESPONSE_PROCESSING_STARTED",
    "RESPONSE_PROCESSING_PASSED",
    "PROBABILITY_VALIDATION_STARTED",
    "PROBABILITY_VALIDATION_PASSED",
    "ARTIFACT_PUBLICATION_STARTED",
    "COMPLETE",
)


def tree_hashes(directory):
    return {
        str(p.relative_to(directory)): original.sha256(p)
        for p in sorted(directory.rglob("*"))
        if p.is_file()
    }


def attempt_number(variant):
    if variant == B:
        return 2
    if variant in original.IDS[2:]:
        return 1
    raise PermissionError("Only approved B2, C1 and D1 execution is permitted")


def attempt_path(variant):
    return STUDY / "execution/candidates" / variant / f"attempt_{attempt_number(variant)}"


def scientific_contract(plan, prep, training, tuning):
    expected = prep["candidates"][B]
    X, y, query, positions = original.matrices(training, tuning, plan, B)
    checks = {
        "training_X_sha256": original.array_hash(X),
        "training_y_sha256": original.array_hash(y),
        "tuning_X_sha256": original.array_hash(query),
        "selected_position_sha256": original.positions_hash(positions),
        "query_source_order_sha256": original.positions_hash(tuning.X.index.to_numpy()),
    }
    if any(checks[k] != expected[k] for k in checks):
        raise ValueError("B scientific matrix or source-order fingerprint changed")
    if list(X.shape) != [13975, 71] or list(query.shape) != [3193, 71]:
        raise ValueError("B frozen row/feature counts changed")
    if X.dtype != np.dtype("float64") or y.dtype != np.dtype("int64"):
        raise ValueError("B scientific input dtypes changed")
    return {
        "variant": B,
        "prepared_candidate": expected,
        "scientific_identity_sha256": original.digest_json(expected),
        "ordered_71_features": expected["feature_names"],
        "ordered_feature_sha256": original.digest_json(expected["feature_names"]),
        **checks,
        "training_source_row_order_sha256": original.positions_hash(training.X.index.to_numpy()),
        "training_subject_membership_order_sha256": original.digest_json(training.groups.tolist()),
        "tuning_subject_membership_order_sha256": original.digest_json(tuning.groups.tolist()),
        "tuning_y_sha256": original.array_hash(tuning.y.to_numpy(dtype=np.int64)),
        "frozen_checkpoint_manifest_hashes": plan["frozen_checkpoint_manifest_hashes"],
        "python_executable": plan["invariants"]["python_executable"],
        "python_version": plan["invariants"]["python_version"],
        "tabpfn_client_version": "0.6.1",
        "constructor": original.CONSTRUCTOR,
        "positive_class": 1,
        "positive_class_semantics": (
            "y=1: new sustained hypotension episode beginning within 5 minutes"
        ),
        "classes": [0, 1],
        "positive_probability_column": 1,
        "probability_semantics": "raw uncalibrated P(y=1), original TUNING source-row order",
        "allowed_value_partitions": ["training", "tuning"],
    }


def verify_authorization(plan, prep, training, tuning):
    path = RECOVERY / "B_attempt_2_authorization.json"
    if original.sha256(path) != path.with_suffix(".sha256").read_text().strip():
        raise ValueError("B recovery authorization seal differs")
    auth = original.read_json(path)
    if (
        auth.get("status") != "APPROVED"
        or auth.get("variant") != B
        or auth.get("attempt_2_authorization") != "EXECUTION_RECOVERY_ONLY"
        or auth.get("attempt_1_operational_state") != "FAILED"
        or auth.get("attempt_1_hosted_outcome") != "INDETERMINATE"
        or auth.get("maximum_B_attempt") != 2
        or auth.get("C_D_retry_authorized") is not False
        or auth.get("C_D_progression_requires_verified_B_complete") is not True
        or auth.get("automatic_retry_permitted") is not False
        or auth.get("attempt_3_permitted") is not False
        or auth.get("independent_candidate_evaluation_number") != 1
        or auth.get("approval_request_sha256") != APPROVAL_SHA
    ):
        raise PermissionError("Exact B-only architectural recovery approval is required")
    if original.sha256(RECOVERY / "approved_recovery_request.md") != APPROVAL_SHA:
        raise PermissionError("Approved recovery request differs")
    if auth["study_plan_sha256"] != original.SPEC_HASH:
        raise ValueError("Recovery scientific plan differs")
    if auth["preparation_manifest_sha256"] != original.sha256(
        STUDY / "preparation_manifest.json"
    ) or auth["preparation_seal_sha256"] != original.sha256(STUDY / "preparation_manifest.sha256"):
        raise ValueError("Original preparation manifest/seal changed")
    old = STUDY / "execution/candidates" / B / "attempt_1"
    if tree_hashes(old) != auth["attempt_1_evidence_sha256"]:
        raise ValueError("Attempt-1 evidence changed")
    if (
        sorted(str(p.relative_to(old)) for p in old.rglob("*") if p.is_dir())
        != auth["attempt_1_directory_inventory"]
    ):
        raise ValueError("Attempt-1 directory evidence changed")
    if set(auth["historical_execution_records_sha256"]) != {
        "review_required.json",
        "started_manifest.json",
    }:
        raise ValueError("Only the exact historical execution stop/start may be resolved")
    for name, digest in auth["historical_execution_records_sha256"].items():
        if original.sha256(STUDY / "execution" / name) != digest:
            raise ValueError("Historical execution stop/start evidence changed")
    if scientific_contract(plan, prep, training, tuning) != auth["scientific_contract"]:
        raise ValueError("Authorized B scientific fingerprint differs")
    if set(auth["recovery_code_hashes"]) != set(CODE_PATHS):
        raise ValueError("Exact reviewed recovery source set is required")
    if set(auth["validation_evidence_sha256"]) != set(VALIDATION_FILES):
        raise ValueError("Exact reviewed recovery validation set is required")
    for name, digest in auth["recovery_code_hashes"].items():
        if original.sha256(REPO / name) != digest:
            raise ValueError("Reviewed recovery implementation changed")
    for name, digest in auth["validation_evidence_sha256"].items():
        if original.sha256(RECOVERY / name) != digest:
            raise ValueError("Recovery validation evidence changed")
    if original.read_json(RECOVERY / "validation_report.json").get("status") != "PASS":
        raise ValueError("Recovery validation did not pass")
    return auth


def checked_context(*, external=False):
    if external:
        from intraop.evaluation.locked_execution import ensure_study_open

        ensure_study_open(STUDY)
    plan = original.verified_spec()
    environment = original.environment(plan, external=external)
    prep = original.verify_preparation(plan)
    loader, training, tuning = original.load_development(plan)
    auth = verify_authorization(plan, prep, training, tuning)
    return plan, prep, loader, training, tuning, auth, environment


def inspect_processes():
    """Inspect argv, never environments; return only relevant PIDs and fail closed."""
    try:
        result = subprocess.run(
            ["/bin/ps", "-axo", "pid=,comm=,args="],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        raise RuntimeError("Process inspection unavailable; hosted launch blocked") from None
    scripts = {
        "study_worker.py",
        "run_study.py",
        "recovery_worker.py",
        "recovery_run.py",
        "modeling_tabpfn_worker.py",
        "modeling_development_resume.py",
    }
    import os

    found = []
    for line in result.stdout.splitlines():
        fields = line.strip().split(None, 2)
        if len(fields) != 3:
            continue
        pid, executable, command = fields
        if int(pid) == os.getpid() or not Path(executable).name.lower().startswith("python"):
            continue
        try:
            words = shlex.split(command)
        except ValueError:
            raise RuntimeError("Unparseable Python process; launch blocked") from None
        if {Path(word).name for word in words} & scripts:
            found.append(int(pid))
    return found


def require_inactive(scanner=None):
    if (scanner or inspect_processes)():
        raise RuntimeError("ACTIVE worker/parent; hosted launch blocked")


def require_no_output(directory):
    if (directory / "bundle").exists():
        raise RuntimeError("Existing completed B bundle blocks duplicate execution")
    for p in directory.rglob("*"):
        if not p.is_file():
            continue
        if p.name in {"prediction.csv", "metrics.json", "completion.json"} or (
            p.suffix == ".csv" and "prediction" in p.name
        ):
            raise RuntimeError("Existing B output/completion evidence blocks recovery")
        if p.name == "exit.json" and original.read_json(p).get("returncode") == 0:
            raise RuntimeError("Successful exit evidence blocks duplicate recovery")


def reserve_B_attempt_2(plan, prep, training, tuning, *, scanner=None):
    from intraop.evaluation.locked_execution import ensure_study_open

    ensure_study_open(STUDY)
    require_inactive(scanner)
    auth = verify_authorization(plan, prep, training, tuning)
    if (RECOVERY / "study_stop.json").exists():
        raise RuntimeError("Unresolved new study stop blocks recovery")
    if any((STUDY / "execution/candidates" / v).exists() for v in original.IDS[2:]):
        raise RuntimeError("Unexpected C/D attempt before B recovery blocks allocation")
    directory = STUDY / "execution/candidates" / B
    if attempt_path(B).exists():
        raise RuntimeError("Attempt 2 already allocated; attempt 3 is forbidden")
    require_no_output(directory)
    original.atomic_json(
        attempt_path(B) / "reservation.json",
        {
            "status": "reserved",
            "attempt_number": 2,
            "variant": B,
            "authorization": "EXECUTION_RECOVERY_ONLY",
            "authorization_sha256": original.sha256(RECOVERY / "B_attempt_2_authorization.json"),
            "scientific_identity_sha256": auth["scientific_contract"]["scientific_identity_sha256"],
            "study_plan_sha256": original.SPEC_HASH,
            "reserved_utc": original.timestamp(),
            "independent_candidate_evaluation_number": 1,
            "automatic_retry_permitted": False,
        },
        exclusive=True,
    )
    return attempt_path(B) / "logs"


def state(variant):
    """Operational state of the authorized attempt, without reclassifying B1."""
    directory = STUDY / "execution/candidates" / variant
    attempt = attempt_path(variant)
    if attempt_number(variant) == 1:
        return original.candidate_state(directory)
    if (attempt / "logs/failure_marker.json").exists() or (attempt / "failure.json").exists():
        return "FAILED"
    if not attempt.exists():
        return "AUTHORIZED_NOT_ALLOCATED"
    exit_path = attempt / "logs/exit.json"
    if exit_path.exists():
        record = original.read_json(exit_path)
        if record.get("returncode") != 0:
            return "FAILED"
        if (
            record.get("worker_protocol", {}).get("status") == "COMPLETE"
            and (directory / "bundle/bundle_manifest.json").exists()
            and stage_summary(attempt)["last_completed_stage"] == "COMPLETE"
        ):
            return "COMPLETE"
    return "INDETERMINATE"


def mark_stage(attempt, stage):
    if stage not in STAGES:
        raise ValueError("Unknown execution stage")
    original.atomic_json(
        attempt / "stages" / f"{STAGES.index(stage) + 1:02d}_{stage}.json",
        {"stage": stage, "timestamp_utc": original.timestamp(), "status": "completed"},
        exclusive=True,
    )


def http_category(status):
    return f"HTTP_{status // 100}XX" if type(status) is int and 100 <= status <= 599 else None


def safe_error_class(exc):
    value = type(exc).__name__
    return value if re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]{0,80}", value) else "UNKNOWN_ERROR_CLASS"


def stage_summary(attempt):
    completed = []
    for p in sorted((attempt / "stages").glob("*.json")):
        value = original.read_json(p)
        if set(value) != {"stage", "timestamp_utc", "status"} or value["stage"] not in STAGES:
            raise ValueError("Unsafe or invalid execution stage record")
        if value["status"] != "completed":
            raise ValueError("Unexpected execution stage status")
        completed.append(value["stage"])
    responses = []
    requests = 0
    for p in sorted((attempt / "http_events").glob("*.json")):
        value = original.read_json(p)
        allowed = {"status", "timestamp_utc"}
        if value.get("status") == "RESPONSE_RECEIVED":
            allowed |= {"http_status", "http_category"}
        if set(value) != allowed:
            raise ValueError("Unsafe HTTP event fields")
        if value.get("status") == "REQUEST_STARTED":
            requests += 1
        elif value.get("status") == "RESPONSE_RECEIVED":
            status = value.get("http_status")
            if http_category(status) is None:
                raise ValueError("Invalid HTTP status evidence")
            responses.append(status)
        else:
            raise ValueError("Unknown HTTP event status")
    return {
        "last_completed_stage": completed[-1] if completed else None,
        "hosted_call_begun": "HOSTED_CALL_STARTED" in completed,
        "hosted_fit_returned": "HOSTED_FIT_RETURNED" in completed,
        "predict_proba_returned": "PREDICT_PROBA_RETURNED" in completed,
        "hosted_call_returned": "HOSTED_CALL_RETURNED" in completed,
        "http_request_started": requests > 0,
        "http_response_received": bool(responses),
        "http_response_evidence": "RECORDED" if responses else "NOT_RECORDED",
        "last_observed_http_status": responses[-1] if responses else None,
        "last_observed_http_category": http_category(responses[-1]) if responses else None,
        "last_http_status_is_failure_cause": False,
    }
