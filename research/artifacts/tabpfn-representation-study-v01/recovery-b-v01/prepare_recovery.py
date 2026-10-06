"""Local-only authorization preparation; never allocate a worker attempt or import SDK."""

import json

import recovery_support as recovery


def prepare():
    original = recovery.original
    plan = original.verified_spec()
    original.environment(plan)
    prep = original.verify_preparation(plan)
    loader, training, tuning = original.load_development(plan)
    path = recovery.RECOVERY / "B_attempt_2_authorization.json"
    if path.exists():
        return recovery.verify_authorization(plan, prep, training, tuning)
    candidate = recovery.STUDY / "execution/candidates" / recovery.B
    old = candidate / "attempt_1"
    recovery.require_no_output(candidate)
    if recovery.attempt_path(recovery.B).exists():
        raise RuntimeError("Attempt 2 already exists; cannot authorize another execution")
    for variant in original.IDS[2:]:
        if (recovery.STUDY / "execution/candidates" / variant).exists():
            raise RuntimeError("Unexpected C/D attempt precedes authorized B recovery")
    marker = original.read_json(old / "logs/failure_marker.json")
    exit_record = original.read_json(old / "logs/exit.json")
    if (
        marker.get("status") != "FAILED"
        or marker.get("returncode") != 2
        or exit_record.get("signal") is not None
        or exit_record.get("worker_protocol", {}).get("error_type") != "ValueError"
    ):
        raise ValueError("Historical B failure does not match the specifically approved attempt")
    if original.sha256(recovery.RECOVERY / "approved_recovery_request.md") != recovery.APPROVAL_SHA:
        raise PermissionError("Exact user architectural approval missing")
    validation = original.read_json(recovery.RECOVERY / "validation_report.json")
    if validation["status"] != "PASS" or validation["hosted_calls"] != 0:
        raise RuntimeError("Recovery software validation must pass before sealing")
    code = recovery.CODE_PATHS
    auth = {
        "status": "APPROVED",
        "authorized_utc": original.timestamp(),
        "variant": recovery.B,
        "approval_request_sha256": recovery.APPROVAL_SHA,
        "attempt_1_operational_state": "FAILED",
        "attempt_1_hosted_outcome": "INDETERMINATE",
        "attempt_2_authorization": "EXECUTION_RECOVERY_ONLY",
        "maximum_B_attempt": 2,
        "independent_candidate_evaluation_number": 1,
        "observed_result_available_from_attempt_1": False,
        "C_D_retry_authorized": False,
        "C_D_progression_requires_verified_B_complete": True,
        "study_plan_sha256": original.SPEC_HASH,
        "preparation_manifest_sha256": original.sha256(
            recovery.STUDY / "preparation_manifest.json"
        ),
        "preparation_seal_sha256": original.sha256(recovery.STUDY / "preparation_manifest.sha256"),
        "attempt_1_evidence_sha256": recovery.tree_hashes(old),
        "attempt_1_directory_inventory": sorted(
            str(p.relative_to(old)) for p in old.rglob("*") if p.is_dir()
        ),
        "historical_execution_records_sha256": {
            name: original.sha256(recovery.STUDY / "execution" / name)
            for name in ["review_required.json", "started_manifest.json"]
        },
        "scientific_contract": recovery.scientific_contract(plan, prep, training, tuning),
        "recovery_code_hashes": {name: original.sha256(recovery.REPO / name) for name in code},
        "validation_evidence_sha256": {
            name: original.sha256(recovery.RECOVERY / name)
            for name in ["validation_report.json", "software_tests.txt"]
        },
        "allocation_requires_live_inactive_process_check": True,
        "process_inspection_failure_policy": "FAIL_CLOSED",
        "automatic_retry_permitted": False,
        "attempt_3_permitted": False,
        "partition_access_log": loader.access_log,
        "heldout_values_parsed": False,
        "hosted_calls_during_preparation": 0,
    }
    original.atomic_json(path, auth, exclusive=True)
    original.atomic_bytes(
        path.with_suffix(".sha256"), (original.sha256(path) + "\n").encode(), exclusive=True
    )
    recovery.verify_authorization(plan, prep, training, tuning)
    return auth


if __name__ == "__main__":
    auth = prepare()
    print(
        json.dumps(
            {
                "status": auth["attempt_2_authorization"],
                "variant": recovery.B,
                "attempt_2_allocated": False,
                "hosted_calls": 0,
            }
        )
    )
